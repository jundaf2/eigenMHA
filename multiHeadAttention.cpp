// ----------------------------------------------------------------------
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ----------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include "multiHeadAttention.h"


#define COUNTOF(arr) int(sizeof(arr) / sizeof(arr[0]))
#define INIT_MEAN    0.0
#define INIT_VAR     0.5
#define WGROUP_COUNT 4

#if defined(__linux__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
inline double seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__QNX__)
#include <time.h>
inline double seconds(void) {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return ((double)tp.tv_sec + (double)tp.tv_nsec / 1000000000.0);
}
#else
#error UNSUPPORTED PLATFORM FOR TIME FUNCTION
#endif

// Returns uniformly distributed integer values between [lower,..,upper], 
// ie, with both lower and upper limits included.
inline int randRangeInt(int lower, int upper) {
    int lo  = (lower < upper ? lower : upper);
    int hi  = (lower > upper ? lower : upper);
    return lo + int(drand48() * (hi - lo + 1));
}

// Returns uniformly distributed floating point values between [bias, bias+range) 
// assuming range>0, ie, including the lower limit but excluding the upper bound.
inline double randRangeDbl(double bias, double range) {
    return range * drand48() + bias;
}

// Initializes buffer with uniformly distributed values with the given mean and variance.
template <typename T_ELEM>
void initBuffer(T_ELEM *image, size_t imageSize, double mean, double var) {
    double range = sqrt(12.0 * var);
    double bias  = mean - 0.5 * range;
    for (size_t index = 0; index < imageSize; index++) {
        image[index] = (T_ELEM) randRangeDbl(bias, range);
    }
}


template <typename T_ELEM>
void saveWeights(const char *fName, int dimA[3], int strideA[3], T_ELEM *weightAddr) {
    FILE *fp = fopen(fName, "w");
    if (fp == NULL) {
        const char *reason = (errno ? strerror(errno) : "unknown reason");
        fprintf(stderr, "ERROR: failed to open '%s' file (%s)\n\n", fName, reason);
        exit(-1);
    }

    // Number of decimal digits when saving as text.
    int decDigs = (sizeof(T_ELEM) == sizeof(double) ? 16 : 8);

    for (int h = 0; h < dimA[0]; h++) {
        for (int r = 0; r < dimA[1]; r++) {
            for (int c = 0; c < dimA[2]; c++) {
                size_t idx = size_t(h) * strideA[0] + size_t(r) * strideA[1] + size_t(c) * strideA[2];
                fprintf(fp, " % -.*e", decDigs, double(weightAddr[idx]));
            }
            fprintf(fp, "\n");
        }
    }

    if (fclose(fp) != 0) {
        const char *reason = (errno ? strerror(errno) : "unknown reason");
        fprintf(stderr, "ERROR: failed to write to '%s' file (%s)\n\n", fName, reason);
        exit(-1);
    }
}

template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<IS_TRAINING, T_ELEM, T_MATH>::setup(testOpts &opts) {
    attn_desc = NULL;
    drop_desc = NULL;
    q_desc    = NULL;
    k_desc    = NULL;
    v_desc    = NULL;
    o_desc    = NULL;

    dropoutBuf     = NULL;
    dropoutBufSize = 0;

    devQ  = NULL;
    devK  = NULL;
    devV  = NULL;
    devO  = NULL;
    devDQ = NULL;
    devDK = NULL;
    devDV = NULL;
    devDO = NULL;
    devW  = NULL;
    devDW = NULL;

    hostQ  = NULL;
    hostK  = NULL;
    hostV  = NULL;
    hostO  = NULL;
    hostDQ = NULL;
    hostDK = NULL;
    hostDV = NULL;
    hostDO = NULL;
    hostW  = NULL;
    hostDW = NULL;

    devWkspace = NULL;
    devReserve = NULL;

    maxWeights = 0;
    maxWkspace = 0;
    maxReserve = 0;

    maxElemQ = 0;
    maxElemK = 0;
    maxElemV = 0;
    maxElemO = 0;

    qSeqArray = NULL;
    kSeqArray = NULL;

    loWinIdx = NULL;
    hiWinIdx = NULL;

    mainCfg.numHeads    = opts.attnNumHeads;
    mainCfg.batchSize   = opts.attnBatchSize;
    mainCfg.beamSize    = opts.attnBeamSize;
    mainCfg.smScaler    = opts.attnSmScaler;
    mainCfg.dropoutRate = opts.attnDropoutRate;
    mainCfg.qSize       = opts.attnQsize;
    mainCfg.kSize       = opts.attnKsize;
    mainCfg.vSize       = opts.attnVsize;
    mainCfg.qProjSize   = opts.attnProjQsize;
    mainCfg.kProjSize   = opts.attnProjKsize;
    mainCfg.vProjSize   = opts.attnProjVsize;
    mainCfg.oProjSize   = opts.attnProjOsize;
    mainCfg.seqLenQ     = opts.attnSeqLenQ;
    mainCfg.seqLenK     = opts.attnSeqLenK;
    mainCfg.resLink     = opts.attnResLink == 0 ? false : true;
    mainCfg.projBias    = opts.attnProjBias == 0 ? false : true;
    mainCfg.dataType    = cudnnDataType_t(opts.attnDataType);
    mainCfg.compPrec    = cudnnDataType_t(opts.attnCompPrec);


    if (opts.attnProjBias == 0) {
        mainCfg.attnMode = (mainCfg.attnMode | CUDNN_ATTN_DISABLE_PROJ_BIASES);
    } else if (opts.attnProjBias == 1) {
        mainCfg.attnMode = (mainCfg.attnMode | CUDNN_ATTN_ENABLE_PROJ_BIASES);
    } else {
        fprintf(stderr, "ERROR: wrong -attnProjBias value\n\n");
        exit(-1);
    }

    if (mainCfg.numHeads <= 0 || mainCfg.batchSize <= 0 || mainCfg.beamSize <= 0) {
        fprintf(stderr, "ERROR: wrong attention NumHeads/BatchSize/BeamSize arguments\n\n");
        exit(-1);
    }


    int qProjLen = mainCfg.qLength();
    int kProjLen = mainCfg.kLength();
    int outLen   = mainCfg.oLength();


    CHECK_CUDNN_ERR(cudnnCreate(&handle));
    CHECK_CUDNN_ERR(cudnnCreateAttnDescriptor(&attn_desc));
    CHECK_CUDNN_ERR(cudnnCreateDropoutDescriptor(&drop_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&q_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&k_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&v_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&o_desc));

    size_t maxQTokens = size_t(mainCfg.seqLenQ) * mainCfg.batchSize * mainCfg.beamSize;
    size_t maxKTokens = size_t(mainCfg.seqLenK) * mainCfg.batchSize;

    // Buffer Q/K/V/O capacity in elements.
    maxElemQ = maxQTokens * mainCfg.qSize;
    maxElemK = maxKTokens * mainCfg.kSize;
    maxElemV = maxKTokens * mainCfg.vSize;
    maxElemO = maxQTokens * outLen;
    maxElemA = maxQTokens * mainCfg.numHeads * mainCfg.seqLenK;

    maxElemQbar = maxQTokens * mainCfg.numHeads * mainCfg.qProjSize;
    maxElemKbar = maxKTokens * mainCfg.numHeads * mainCfg.kProjSize;
    maxElemVbar = maxKTokens * mainCfg.numHeads * mainCfg.vProjSize;
    maxElemHbar = maxQTokens * mainCfg.numHeads * mainCfg.vProjSize;

    // Allocate input and output buffers (forward/inference pass).
    CHECK_CUDA_ERR(cudaMalloc((void **)&devQ, maxElemQ * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devK, maxElemK * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devV, maxElemV * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&devO, maxElemO * sizeof(T_ELEM)));

    // Allocate input and output buffers (backward/training pass).
    if (IS_TRAINING) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDQ, maxElemQ * sizeof(T_ELEM)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDK, maxElemK * sizeof(T_ELEM)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDV, maxElemV * sizeof(T_ELEM)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&devDO, maxElemO * sizeof(T_ELEM)));
    }

    CHECK_CUDNN_ERR(cudnnDropoutGetStatesSize(handle, &dropoutBufSize));
    CHECK_CUDA_ERR(cudaMalloc((void **)&dropoutBuf, dropoutBufSize));

    CHECK_CUDNN_ERR(cudnnSetDropoutDescriptor(drop_desc, handle, mainCfg.dropoutRate, dropoutBuf, dropoutBufSize, 0));

    CHECK_CUDNN_ERR(cudnnSetAttnDescriptor(attn_desc,
                                           mainCfg.attnMode,
                                           mainCfg.numHeads,
                                           mainCfg.smScaler,
                                           mainCfg.dataType,
                                           mainCfg.compPrec,
                                           CUDNN_DEFAULT_MATH,
                                           IS_TRAINING && mainCfg.dropoutRate > 0.0 ? drop_desc : NULL,
                                           NULL,
                                           mainCfg.qSize,
                                           mainCfg.kSize,
                                           mainCfg.vSize,
                                           mainCfg.qProjSize,
                                           mainCfg.kProjSize,
                                           mainCfg.vProjSize,
                                           mainCfg.oProjSize,
                                           mainCfg.seqLenQ,
                                           mainCfg.seqLenK,
                                           mainCfg.batchSize,
                                           mainCfg.beamSize));

    if (IS_TRAINING) {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &maxWeights, &maxWkspace, &maxReserve));
    } else {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &maxWeights, &maxWkspace, NULL));
    }

    if (maxWeights > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devW, maxWeights));
        if (IS_TRAINING) {
            CHECK_CUDA_ERR(cudaMalloc((void **)&devDW, maxWeights));
        }
    }
    if (maxWkspace > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devWkspace, maxWkspace));
    }
    if (maxReserve > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void **)&devReserve, maxReserve));

        // Fill with -NaN to deterct incorrect segment write for debugging.
        CHECK_CUDA_ERR(cudaMemset(devReserve, 0xff, maxReserve));
    }

    qSeqArray = (int *)calloc(mainCfg.batchSize * mainCfg.beamSize, sizeof(int));
    kSeqArray = (int *)calloc(mainCfg.batchSize, sizeof(int));

    if (loWinIdx == NULL && hiWinIdx == NULL) {
        loWinIdx = (int *)calloc(mainCfg.seqLenQ, sizeof(int));
        hiWinIdx = (int *)calloc(mainCfg.seqLenQ, sizeof(int));
    }

    // Allocate weight and data buffers on the CPU side.
    if (maxWeights > 0) {
        hostW  = (T_ELEM *)malloc(maxWeights);
        hostDW = (T_ELEM *)malloc(maxWeights);
    }

    hostQ = (T_ELEM *)malloc(maxElemQ * sizeof(T_ELEM));
    hostK = (T_ELEM *)malloc(maxElemK * sizeof(T_ELEM));
    hostV = (T_ELEM *)malloc(maxElemV * sizeof(T_ELEM));
    hostO = (T_ELEM *)malloc(maxElemO * sizeof(T_ELEM));

    // Allocate input and output buffers (backward/training pass).
    if (IS_TRAINING) {
        hostDQ = (T_ELEM *)malloc(maxElemQ * sizeof(T_ELEM));
        hostDK = (T_ELEM *)malloc(maxElemK * sizeof(T_ELEM));
        hostDV = (T_ELEM *)malloc(maxElemV * sizeof(T_ELEM));
        hostDO = (T_ELEM *)malloc(maxElemO * sizeof(T_ELEM));
    }
}

// Teardown destroys various descriptors and free memories.
template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<IS_TRAINING, T_ELEM, T_MATH>::teardown() {
    cudnnDestroyAttnDescriptor(attn_desc);
    attn_desc = NULL;

    cudnnDestroyDropoutDescriptor(drop_desc);
    drop_desc = NULL;

    cudnnDestroySeqDataDescriptor(q_desc);
    q_desc = NULL;

    cudnnDestroySeqDataDescriptor(k_desc);
    k_desc = NULL;

    cudnnDestroySeqDataDescriptor(v_desc);
    v_desc = NULL;

    cudnnDestroySeqDataDescriptor(o_desc);
    o_desc = NULL;

    cudaFree(dropoutBuf);
    dropoutBuf = NULL;

    cudaFree(devQ);
    devQ = NULL;

    cudaFree(devK);
    devK = NULL;

    cudaFree(devV);
    devV = NULL;

    cudaFree(devO);
    devO = NULL;

    if (IS_TRAINING) {
        cudaFree(devDQ);
        devDQ = NULL;

        cudaFree(devDK);
        devDK = NULL;

        cudaFree(devDV);
        devDV = NULL;

        cudaFree(devDO);
        devDO = NULL;

        cudaFree(devDW);
        devDW = NULL;
    }

    cudaFree(devW);
    devW = NULL;

    cudaFree(devWkspace);
    devWkspace = NULL;

    cudaFree(devReserve);
    devReserve = NULL;

    free(qSeqArray);
    qSeqArray = NULL;

    free(kSeqArray);
    kSeqArray = NULL;

    free(loWinIdx);
    loWinIdx = NULL;

    free(hiWinIdx);
    hiWinIdx = NULL;

    free(hostW);
    hostW = NULL;

    free(hostDW);
    hostDW = NULL;

    free(hostQ);
    hostQ = NULL;

    free(hostK);
    hostK = NULL;

    free(hostV);
    hostV = NULL;

    free(hostO);
    hostO = NULL;

    free(hostDQ);
    hostDQ = NULL;

    free(hostDK);
    hostDK = NULL;

    free(hostDV);
    hostDV = NULL;

    free(hostDO);
    hostDO = NULL;
}

template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<IS_TRAINING, T_ELEM, T_MATH>::testgen(attnConfig *testCfg) {
    *testCfg = this->mainCfg;

    // Initialize qSeqArray and kSeqArray values and attention window
    size_t qBatches = testCfg->qSeqLenCount();
    size_t kBatches = testCfg->kSeqLenCount();

    // Set random number generator seed.
    srand48(testCfg->randSeed);

    // Fixed lengths for all sequences in a batch.
    for (size_t i = 0; i < qBatches; ++i) {
        qSeqArray[i] = testCfg->seqLenQ;
    }

    for (size_t i = 0; i < kBatches; ++i) {
        kSeqArray[i] = testCfg->seqLenK;
    }

    // Set the maximum attention window in all time-steps.
    for (int i = 0; i < testCfg->seqLenQ; ++i) {
        loWinIdx[i] = 0;
        hiWinIdx[i] = testCfg->seqLenK;
    }
    

    printf("Test parameters:\n\n");
    printf("#### attnTrain       = %d (%s)\n", IS_TRAINING, IS_TRAINING ? "training" : "inference");
    printf("#### attnDataType    = %d (FP%d)\n", testCfg->dataType, int(8*sizeof(T_ELEM)));
    printf("#### attnCompPrec    = %d (FP%d)\n", testCfg->compPrec, int(8*sizeof(T_MATH)));
    printf("#### attnNumHeads    = %d\n", testCfg->numHeads);
    printf("#### attnBatchSize   = %d\n", testCfg->batchSize);
    printf("#### attnBeamSize    = %d\n", testCfg->beamSize);
    printf("#### attnSmScaler    = %.4e\n", testCfg->smScaler);
    printf("#### attnDropoutRate = %.4f\n", testCfg->dropoutRate);
    printf("#### attnQsize       = %d\n", testCfg->qSize);
    printf("#### attnKsize       = %d\n", testCfg->kSize);
    printf("#### attnVsize       = %d\n", testCfg->vSize);
    printf("#### attnProjQsize   = %d%s\n", testCfg->qProjSize, testCfg->qProjSize ? "" : " (no Q weights)");
    printf("#### attnProjKsize   = %d%s\n", testCfg->kProjSize, testCfg->kProjSize ? "" : " (no K weights)");
    printf("#### attnProjVsize   = %d%s\n", testCfg->vProjSize, testCfg->vProjSize ? "" : " (no V weights)");
    printf("#### attnProjOsize   = %d%s\n", testCfg->oProjSize, testCfg->oProjSize ? "" : " (no O weights)");
    printf("#### attnSeqLenQ     = %d\n", testCfg->seqLenQ);
    printf("#### attnSeqLenK     = %d\n", testCfg->seqLenK);
    printf("#### attnResLink     = %d\n", testCfg->resLink);
    printf("#### attnProjBias    = %d\n", testCfg->projBias);

    for (size_t i = 0; i < qBatches; ++i) {
        printf("sequence_length_q[idx=%lu]=%d\n", i, qSeqArray[i]);
    }
    printf("\n");

    for (size_t i = 0; i < kBatches; ++i) {
        printf("sequence_length_k[idx=%lu]=%d\n", i, kSeqArray[i]);
    }
    printf("\n");

    for (int i = 0; i < testCfg->seqLenQ; ++i) {
        printf("attention_window[time=%d]=%d:%d\n", i, loWinIdx[i], hiWinIdx[i]);
    }
    printf("\n");
}

template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<IS_TRAINING, T_ELEM, T_MATH>::run() {
    attnConfig testCfg;

    testgen(&testCfg);

    CHECK_CUDNN_ERR(cudnnSetDropoutDescriptor(drop_desc, handle, testCfg.dropoutRate, dropoutBuf, dropoutBufSize, 0));

    // Set attention descriptor according to generated testCfg.
    CHECK_CUDNN_ERR(cudnnSetAttnDescriptor(attn_desc,
                                           testCfg.attnMode,
                                           testCfg.numHeads,
                                           testCfg.smScaler,
                                           testCfg.dataType,
                                           testCfg.compPrec,
                                           CUDNN_DEFAULT_MATH,
                                           IS_TRAINING && testCfg.dropoutRate > 0.0 ? drop_desc : NULL,
                                           NULL,
                                           testCfg.qSize,
                                           testCfg.kSize,
                                           testCfg.vSize,
                                           testCfg.qProjSize,
                                           testCfg.kProjSize,
                                           testCfg.vProjSize,
                                           testCfg.oProjSize,
                                           testCfg.seqLenQ,
                                           testCfg.seqLenK,
                                           testCfg.batchSize,
                                           testCfg.beamSize));

    size_t sizeWeights = 0, sizeWkspace = 0, sizeReserve = 0;

    if (IS_TRAINING) {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &sizeWeights, &sizeWkspace, &sizeReserve));
    } else {
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(handle, attn_desc, &sizeWeights, &sizeWkspace, NULL));
    }

    // Sanity check so we do not over-run the allocated buffers.
    if (sizeWeights > maxWeights || sizeWkspace > maxWkspace || sizeReserve > maxReserve) {
        fprintf(stderr, "ERROR: cudnnGetMultiHeadAttnBuffers() reported inconsistent buffer sizes\n\n");
        exit(-1);
    }

    int qSeqArraySize = testCfg.beamSize * testCfg.batchSize;
    int kSeqArraySize = testCfg.batchSize;

    // host-to-device copies
    size_t size = sizeof(qSeqArray[0]) * qSeqArraySize;
    CHECK_CUDA_ERR(cudaMalloc((void **)&devQSeqArray, size));
    CHECK_CUDA_ERR(cudaMemcpy(devQSeqArray, qSeqArray, size, cudaMemcpyHostToDevice));

    size = sizeof(kSeqArray[0]) * kSeqArraySize;
    CHECK_CUDA_ERR(cudaMalloc((void **)&devKSeqArray, size));
    CHECK_CUDA_ERR(cudaMemcpy(devKSeqArray, kSeqArray, size, cudaMemcpyHostToDevice));

    // Length of output vectors.
    int o_len = testCfg.oLength();

    int dimA[CUDNN_SEQDATA_DIM_COUNT];

    dimA[CUDNN_SEQDATA_BEAM_DIM]  = testCfg.beamSize;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenQ;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = testCfg.qSize;

    cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];
    dataAxes[0] = CUDNN_SEQDATA_BEAM_DIM;
    dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
    dataAxes[2] = CUDNN_SEQDATA_TIME_DIM;
    dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        q_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, qSeqArraySize, qSeqArray, NULL));

    dimA[CUDNN_SEQDATA_BEAM_DIM]  = testCfg.beamSize;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenQ;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = o_len;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        o_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, qSeqArraySize, qSeqArray, NULL));

    // seq-k
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = testCfg.kSize;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        k_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, kSeqArraySize, kSeqArray, NULL));

    // seq-v
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = testCfg.vSize;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        v_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, dataAxes, kSeqArraySize, kSeqArray, NULL));

    size_t qNmbElem = testCfg.qAllData();
    size_t kNmbElem = testCfg.kAllData();
    size_t vNmbElem = testCfg.vAllData();
    size_t oNmbElem = testCfg.oAllData();

    size_t qNmbWeights = testCfg.qAllWeights();
    size_t kNmbWeights = testCfg.kAllWeights();
    size_t vNmbWeights = testCfg.vAllWeights();
    size_t oNmbWeights = testCfg.oAllWeights();

    // Sanity check so we do not over-run the allocated buffers.
    if (qNmbElem > maxElemQ || kNmbElem > maxElemK || vNmbElem > maxElemV || oNmbElem > maxElemO) {
        fprintf(stderr, "ERROR: inconsistent data buffer sizes\n\n");
        exit(-1);
    }

    if (qNmbElem == 0 || kNmbElem == 0 || oNmbElem == 0) {
        fprintf(stderr, "ERROR: Q/K/O data buffers cannot be zero size\n\n");
        exit(-1);
    }

    if (sizeWeights > 0) {
        initBuffer<T_ELEM>(hostW, sizeWeights / sizeof(T_ELEM), INIT_MEAN, INIT_VAR);
    }

    initBuffer<T_ELEM>(hostQ, qNmbElem, INIT_MEAN, INIT_VAR);
    initBuffer<T_ELEM>(hostK, kNmbElem, INIT_MEAN, INIT_VAR);
    initBuffer<T_ELEM>(hostV, vNmbElem, INIT_MEAN, INIT_VAR);

    // Fill output surface with NaN-s.
    CHECK_CUDA_ERR(cudaMemset(devO, 0xFF, oNmbElem * sizeof(devO[0])));

    if (IS_TRAINING) {
        initBuffer<T_ELEM>(hostDO, oNmbElem, INIT_MEAN, INIT_VAR);

        // Fill output surfaces with NaN-s.
        CHECK_CUDA_ERR(cudaMemset(devDQ, 0xFF, sizeof(devDQ[0]) * qNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDK, 0xFF, sizeof(devDK[0]) * kNmbElem));
        CHECK_CUDA_ERR(cudaMemset(devDV, 0xFF, sizeof(devDV[0]) * vNmbElem));

        // Fill the "wgrad" buffer with zeros (results are added to existing values).
        CHECK_CUDA_ERR(cudaMemset(devDW, 0, sizeWeights));
    }

    // Copy the data from GPU (device) to CPU (host)
    CHECK_CUDA_ERR(cudaMemcpy(devW, hostW, sizeWeights, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devQ, hostQ, sizeof(devQ[0]) * qNmbElem, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devK, hostK, sizeof(devK[0]) * kNmbElem, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devV, hostV, sizeof(devV[0]) * vNmbElem, cudaMemcpyHostToDevice));

    if (IS_TRAINING) {
        CHECK_CUDA_ERR(cudaMemcpy(devDO, hostDO, oNmbElem * sizeof(devO[0]), cudaMemcpyHostToDevice));
    }


    double start = seconds();

    if (IS_TRAINING) {
        if (sizeReserve == 0) {
            fprintf(stderr, "ERROR: zero reserve buffer size in training mode\n\n");
            exit(-1);
        }

        printf("Calling cudnnMultiHeadAttnForward(currIdx = -1)\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(handle,
                                                  attn_desc,
                                                  -1,
                                                  loWinIdx,
                                                  hiWinIdx,
                                                  devQSeqArray,
                                                  devKSeqArray,
                                                  q_desc,
                                                  devQ,
                                                  mainCfg.resLink ? devQ : NULL,
                                                  k_desc,
                                                  devK,
                                                  v_desc,
                                                  devV,
                                                  o_desc,
                                                  devO,
                                                  sizeWeights,
                                                  sizeWeights > 0 ? devW : NULL,
                                                  sizeWkspace,
                                                  devWkspace,
                                                  sizeReserve,
                                                  devReserve));

        printf("Calling cudnnMultiHeadAttnBackwardData()\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnBackwardData(handle,
                                                       attn_desc,
                                                       loWinIdx,
                                                       hiWinIdx,
                                                       devQSeqArray,
                                                       devKSeqArray,
                                                       o_desc,
                                                       devDO,
                                                       q_desc,
                                                       devDQ,
                                                       devQ,
                                                       k_desc,
                                                       devDK,
                                                       devK,
                                                       v_desc,
                                                       devDV,
                                                       devV,
                                                       sizeWeights,
                                                       sizeWeights > 0 ? devW : NULL,
                                                       sizeWkspace,
                                                       devWkspace,
                                                       sizeReserve,
                                                       devReserve));

        printf("Calling cudnnMultiHeadAttnBackwardWeights()\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnBackwardWeights(handle,
                                                          attn_desc,
                                                          CUDNN_WGRAD_MODE_ADD,
                                                          q_desc,
                                                          devQ,
                                                          k_desc,
                                                          devK,
                                                          v_desc,
                                                          devV,
                                                          o_desc,
                                                          devDO,
                                                          sizeWeights,
                                                          sizeWeights > 0 ? devW : NULL,
                                                          sizeWeights > 0 ? devDW : NULL,
                                                          sizeWkspace,
                                                          devWkspace,
                                                          sizeReserve,
                                                          devReserve));
    } else {
        if (sizeReserve != 0) {
            fprintf(stderr, "ERROR: non-zero reserve buffer size in inference mode\n\n");
            exit(-1);
        }

        printf("Calling cudnnMultiHeadAttnForward(currIdx = -1)\n");
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(handle,
                                                    attn_desc,
                                                    -1,
                                                    loWinIdx,
                                                    hiWinIdx,
                                                    devQSeqArray,
                                                    devKSeqArray,
                                                    q_desc,
                                                    devQ,
                                                    mainCfg.resLink ? devQ : NULL,
                                                    k_desc,
                                                    devK,
                                                    v_desc,
                                                    devV,
                                                    o_desc,
                                                    devO,
                                                    sizeWeights,
                                                    sizeWeights > 0 ? devW : NULL,
                                                    sizeWkspace,
                                                    devWkspace,
                                                    0,
                                                    NULL));
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    double stop = seconds();

    printf("Elapsed time = %g sec\n", stop - start);

    // Copy forward output to host.
    CHECK_CUDA_ERR(cudaMemcpy(hostO, devO, oNmbElem * sizeof(devO[0]), cudaMemcpyDeviceToHost));

    if (IS_TRAINING) {
        CHECK_CUDA_ERR(cudaMemcpy(hostDQ, devDQ, sizeof(devDQ[0]) * qNmbElem, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(hostDK, devDK, sizeof(devDK[0]) * kNmbElem, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(hostDV, devDV, sizeof(devDV[0]) * vNmbElem, cudaMemcpyDeviceToHost));

        // Copy wgrad results to host
        if (sizeWeights > 0) {
            CHECK_CUDA_ERR(cudaMemcpy(hostDW, devDW, sizeWeights, cudaMemcpyDeviceToHost));
        }
    }
    
}

template class MultiheadAttentionTest<false, float, float>;
template class MultiheadAttentionTest<true, float, float>;