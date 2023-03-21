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

void saveMeta(const char *fName, bool isTraining, attnConfig *testCfg) {
    FILE *fp = fopen(fName, "w");
    if (fp == NULL) {
        const char *reason = (errno ? strerror(errno) : "unknown reason");
        fprintf(stderr, "ERROR: failed to open '%s' file (%s)\n\n", fName, reason);
        exit(-1);
    }

    fprintf(fp, "# train_mode, attn_heads, softmax_scaler, residuals\n");
    fprintf(fp, "%d %d %.16e %d\n", isTraining, testCfg->numHeads, testCfg->smScaler, testCfg->resLink);

    if (fclose(fp) != 0) {
        const char *reason = (errno ? strerror(errno) : "unknown reason");
        fprintf(stderr, "ERROR: failed to write to '%s' file (%s)\n\n", fName, reason);
        exit(-1);
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

template <typename T_ELEM>
void saveAllParams(cudnnHandle_t handle, cudnnAttnDescriptor_t desc, bool isGgrad, size_t paramSize, void *paramBuf) {
    static cudnnMultiHeadAttnWeightKind_t wKind[WGROUP_COUNT] = {
        CUDNN_MH_ATTN_Q_WEIGHTS, CUDNN_MH_ATTN_K_WEIGHTS, CUDNN_MH_ATTN_V_WEIGHTS, CUDNN_MH_ATTN_O_WEIGHTS};

    static const char * baseName[WGROUP_COUNT] = { "wq.dat", "wk.dat", "wv.dat", "wo.dat" };

    cudnnTensorDescriptor_t weightDesc = NULL;
    int nbDims, dimA[3], strideA[3];
    cudnnDataType_t dataTypeUnsed;
    char fileName[64];

    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&weightDesc));

    for (int i = 0; i < WGROUP_COUNT; i++) {
        T_ELEM *weightAddr = NULL;
        CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnWeights(
            handle, desc, wKind[i], paramSize, paramBuf, weightDesc, (void **)&weightAddr));

        CHECK_CUDNN_ERR(cudnnGetTensorNdDescriptor(weightDesc, 3, &dataTypeUnsed, &nbDims, dimA, strideA));

        // cudnnGetMultiHeadAttnWeights() reports a wrong stride in ealier 
        // cuDNN versions for input projection weight tensors.
        if (cudnnGetVersion() < 7602 && wKind[i] != CUDNN_MH_ATTN_O_WEIGHTS) {
            strideA[2] = dimA[0] * dimA[1];
        }

        if (nbDims != 3) {
            fprintf(stderr, "ERROR: weight tensor descriptor should have 3 dimensions, not %d\n\n", nbDims);
            exit(-1);
        }

        sprintf(fileName, "%s%s", isGgrad ? "d" : "", baseName[i]);
        saveWeights<T_ELEM>(fileName, dimA, strideA, weightAddr);
    }

    cudnnDestroyTensorDescriptor(weightDesc);
}

template <typename T_ELEM>
void saveData(const char *fName, int batch, int beam, int nDims, int dimA[4], cudnnSeqDataAxis_t ordA[4], T_ELEM *dataBuf) {
    if (nDims != 4) {
        fprintf(stderr, "ERROR: unexpected number of dimensions %d!=4 in seqdata\n\n", nDims);
        exit(-1);
    }

    if (batch < 0 || batch >= dimA[CUDNN_SEQDATA_BATCH_DIM]) {
        fprintf(stderr, "ERROR: invalid batch=%d for file dump\n\n", batch);
        exit(-1);
    }

    if (beam < 0 || beam >= dimA[CUDNN_SEQDATA_BEAM_DIM]) {
        fprintf(stderr, "ERROR: invalid beam=%d for file dump\n\n", beam);
        exit(-1);
    }

    FILE *fp = fopen(fName, "w");
    if (fp == NULL) {
        const char *reason = (errno ? strerror(errno) : "unknown reason");
        fprintf(stderr, "ERROR: failed to open '%s' file (%s)\n\n", fName, reason);
        exit(-1);
    }

    // The length of embedding vector (same as CUDNN_SEQDATA_VECT_DIM).
    int veclen = dimA[nDims - 1];

    // Actual strides in memory (layout dependent).
    size_t strA[4] = {0};

    // Compute strides from dimensions (SeqData is a packed container).
    strA[nDims - 1] = 1;
    size_t stride   = veclen;
    for (int i = nDims - 2; i >= 0; i--) {
        if (ordA[i] < nDims - 1 && strA[ordA[i]] == 0) {
            strA[ordA[i]] = stride;
            stride *= dimA[ordA[i]];
        } else {
            fprintf(stderr, "ERROR: invalid re-order index ordA[i=%d]=%d\n\n", i, ordA[i]);
            exit(-1);
        }
    }

    // Number of decimal digits when saving as text.
    int decDigs = (sizeof(T_ELEM) == sizeof(double) ? 16 : 8);

    // Write one full sentence (all time-steps for the given batch/beam).
    size_t base = size_t(batch) * strA[CUDNN_SEQDATA_BATCH_DIM] + size_t(beam) * strA[CUDNN_SEQDATA_BEAM_DIM];
    for (int vect = 0; vect < veclen; vect++) {
        for (int time = 0; time < dimA[CUDNN_SEQDATA_TIME_DIM]; time++) {
            size_t idx = size_t(time) * strA[CUDNN_SEQDATA_TIME_DIM] + size_t(vect) * strA[CUDNN_SEQDATA_VECT_DIM];
            fprintf(fp, " % -.*e", decDigs, double(dataBuf[base + idx]));
        }
        fprintf(fp, "\n");
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
    mainCfg.sweep       = opts.attnSweep;
    mainCfg.randGeom    = opts.attnRandGeom != 0 ? 1 : 0;
    mainCfg.randSeed    = opts.attnRandSeed;
    mainCfg.dataType    = cudnnDataType_t(opts.attnDataType);
    mainCfg.compPrec    = cudnnDataType_t(opts.attnCompPrec);
    mainCfg.fileDump    = opts.attnFileDump != 0 ? 1 : 0;

    if (opts.attnQueryMap == 0) {
        mainCfg.attnMode = (mainCfg.attnMode | CUDNN_ATTN_QUERYMAP_ALL_TO_ONE);
    } else if (opts.attnQueryMap == 1) {
        mainCfg.attnMode = (mainCfg.attnMode | CUDNN_ATTN_QUERYMAP_ONE_TO_ONE);
    } else {
        fprintf(stderr, "ERROR: wrong -attnQueryMap value\n\n");
        exit(-1);
    }

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

    if (mainCfg.fileDump != 0) {
        if (mainCfg.batchSize > 1) {
            fprintf(stderr, "ERROR: -attnFileDump%d requires -attnBatchSize=1\n\n", opts.attnFileDump);
            exit(-1);
        }

        if (mainCfg.beamSize > 1) {
            fprintf(stderr, "ERROR: -attnFileDump%d requires -attnBeamSize=1\n\n", opts.attnFileDump);
            exit(-1);
        }

        if (mainCfg.randGeom != 0) {
            fprintf(stderr, "ERROR: -attnFileDump%d requires -attnRandGeom=0\n\n", opts.attnFileDump);
            exit(-1);
        }

        if (mainCfg.projBias != 0) {
            fprintf(stderr, "ERROR: -attnFileDump%d requires -attnProjBias=0\n\n", opts.attnFileDump);
            exit(-1);
        }

        if (IS_TRAINING && mainCfg.dropoutRate > 0.0) {
            fprintf(stderr, "ERROR: -attnFileDump%d requires -attnDropoutRate=0\n\n", opts.attnFileDump);
            exit(-1);
        }
    }

    int qProjLen = mainCfg.qLength();
    int kProjLen = mainCfg.kLength();
    int outLen   = mainCfg.oLength();

    mainCfg.dataLayout = opts.attnDataLayout;

    switch (mainCfg.dataLayout) {
        case 0:  // dataAxes = [T, N, B]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_TIME_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_BEAM_DIM;
            break;

        case 1:  // dataAxes = [T, B, N]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_TIME_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_BEAM_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_BATCH_DIM;
            break;

        case 2:  // dataAxes = [N, T, B]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_BATCH_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_TIME_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_BEAM_DIM;
            break;

        case 3:  // dataAxes = [N, B, T]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_BATCH_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_BEAM_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_TIME_DIM;
            break;

        case 4:  // dataAxes = [B, T, N]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_BEAM_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_TIME_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_BATCH_DIM;
            break;

        case 5:  // dataAxes = [B, N, T]
            mainCfg.dataAxes[0] = CUDNN_SEQDATA_BEAM_DIM;
            mainCfg.dataAxes[1] = CUDNN_SEQDATA_BATCH_DIM;
            mainCfg.dataAxes[2] = CUDNN_SEQDATA_TIME_DIM;
            break;

        default:
            fprintf(stderr, "ERROR: wrong -attnDataLayout%d option\n\n", opts.attnDataLayout);
            exit(-1);
    }

    mainCfg.dataAxes[3] = CUDNN_SEQDATA_VECT_DIM;

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

    // No problem size randomization when the RNG seed is zero.
    if (testCfg->randGeom != 0) {
        for (size_t i = 0; i < qBatches; ++i) {
            qSeqArray[i] = randRangeInt(1, testCfg->seqLenQ);
        }

        for (size_t i = 0; i < kBatches; ++i) {
            kSeqArray[i] = randRangeInt(1, testCfg->seqLenK);
        }

        // Set the random size of attention window in all time-steps.
        for (int i = 0; i < testCfg->seqLenQ; ++i) {
            loWinIdx[i] = randRangeInt(0, testCfg->seqLenK - 1);
            hiWinIdx[i] = randRangeInt(loWinIdx[i], testCfg->seqLenK);
        }
    } else {
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
    }

    const char standardAxes[CUDNN_SEQDATA_DIM_COUNT] = {'T', 'N', 'B', 'V'};
    char dataAxes[CUDNN_SEQDATA_DIM_COUNT];
    for (int ii = 0; ii < CUDNN_SEQDATA_DIM_COUNT; ++ii) {
        dataAxes[ii] = standardAxes[testCfg->dataAxes[ii]];
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
    printf("#### attnDataLayout  = %d (%c,%c,%c,%c)\n", testCfg->dataLayout, dataAxes[0], dataAxes[1], dataAxes[2], dataAxes[3]);
    printf("#### attnResLink     = %d\n", testCfg->resLink);
    printf("#### attnProjBias    = %d\n", testCfg->projBias);
    printf("#### attnSweep       = %d\n", testCfg->sweep);
    printf("#### attnRandGeom    = %d\n", testCfg->randGeom);
    printf("#### attnRandSeed    = %d\n", testCfg->randSeed);
    printf("#### attnFileDump    = %d\n\n", testCfg->fileDump);

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

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        q_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, testCfg.dataAxes, qSeqArraySize, qSeqArray, NULL));

    dimA[CUDNN_SEQDATA_BEAM_DIM]  = testCfg.beamSize;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenQ;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = o_len;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        o_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, testCfg.dataAxes, qSeqArraySize, qSeqArray, NULL));

    // seq-k
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = testCfg.queryMap == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE ? testCfg.beamSize : 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = testCfg.kSize;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        k_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, testCfg.dataAxes, kSeqArraySize, kSeqArray, NULL));

    // seq-v
    dimA[CUDNN_SEQDATA_BEAM_DIM]  = testCfg.queryMap == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE ? testCfg.beamSize : 1;
    dimA[CUDNN_SEQDATA_BATCH_DIM] = testCfg.batchSize;
    dimA[CUDNN_SEQDATA_TIME_DIM]  = testCfg.seqLenK;
    dimA[CUDNN_SEQDATA_VECT_DIM]  = testCfg.vSize;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        v_desc, testCfg.dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, testCfg.dataAxes, kSeqArraySize, kSeqArray, NULL));

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

    if (testCfg.fileDump) {
        cudnnSeqDataAxis_t ordA[CUDNN_SEQDATA_DIM_COUNT];
        int nbDims;

        saveMeta("meta.dat", IS_TRAINING, &testCfg);

        CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(q_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
        saveData<T_ELEM>("q.dat", 0, 0, nbDims, dimA, ordA, hostQ);
        CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(k_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
        saveData<T_ELEM>("k.dat", 0, 0, nbDims, dimA, ordA, hostK);
        CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(v_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
        saveData<T_ELEM>("v.dat", 0, 0, nbDims, dimA, ordA, hostV);

        if (IS_TRAINING) {
            CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(o_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
            saveData<T_ELEM>("dout.dat", 0, 0, nbDims, dimA, ordA, hostDO);
        }

        saveAllParams<T_ELEM>(handle, attn_desc, false, sizeWeights, hostW);
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
        if (testCfg.sweep == 0) {
            // Explicit looping through all time-steps in the inference mode.
            for (int currIdx = 0; currIdx < testCfg.seqLenQ; ++currIdx) {
                printf("Calling cudnnMultiHeadAttnForward(currIdx = %d)\n", currIdx);
                CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(handle,
                                                          attn_desc,
                                                          currIdx,
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
        } else {
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
    }

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    double stop = seconds();

    printf("Elapsed time = %g sec\n", stop - start);

    // Copy forward output to host.
    CHECK_CUDA_ERR(cudaMemcpy(hostO, devO, oNmbElem * sizeof(devO[0]), cudaMemcpyDeviceToHost));

    if (testCfg.fileDump) {
        cudnnSeqDataAxis_t ordA[CUDNN_SEQDATA_DIM_COUNT];
        int nbDims;

        CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(o_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
        saveData<T_ELEM>("out.dat", 0, 0, nbDims, dimA, ordA, hostO);

        if (IS_TRAINING) {
            CHECK_CUDA_ERR(cudaMemcpy(hostDQ, devDQ, sizeof(devDQ[0]) * qNmbElem, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(hostDK, devDK, sizeof(devDK[0]) * kNmbElem, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(hostDV, devDV, sizeof(devDV[0]) * vNmbElem, cudaMemcpyDeviceToHost));

            CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(q_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
            saveData<T_ELEM>("dq.dat", 0, 0, nbDims, dimA, ordA, hostDQ);

            CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(k_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
            saveData<T_ELEM>("dk.dat", 0, 0, nbDims, dimA, ordA, hostDK);

            CHECK_CUDNN_ERR(cudnnGetSeqDataDescriptor(v_desc, NULL, &nbDims, CUDNN_SEQDATA_DIM_COUNT, dimA, ordA, NULL, 0, NULL, NULL));
            saveData<T_ELEM>("dv.dat", 0, 0, nbDims, dimA, ordA, hostDV);

            // Copy wgrad results to host
            if (sizeWeights > 0) {
                CHECK_CUDA_ERR(cudaMemcpy(hostDW, devDW, sizeWeights, cudaMemcpyDeviceToHost));
                saveAllParams<T_ELEM>(handle, attn_desc, true, sizeWeights, hostDW);
            }
        }
    }
}

template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
void doTest(testOpts &opts) {
    MultiheadAttentionTest<IS_TRAINING, T_ELEM, T_MATH> attnTest;
    attnTest.setup(opts);
    attnTest.run();
    attnTest.teardown();
    printf("\nTest DONE\n\n");
    fflush(stdout);
}

static char * baseFile(char *fname) {
    char *base;
    for (base = fname; *fname != '\0'; fname++) {
        if (*fname == '/' || *fname == '\\') {
            base = fname + 1;
        }
    }
    return base;
}

static void parseAttnParameters(int argc, char **argv, testOpts *opts) {
    struct cmdParams {
        const char  *name;
        const char  *fmt;
        size_t      offs;
        const char  *desc;
    } param[] = {
        { "attnTrain",       "%d",  offsetof(testOpts, attnTrain),       "selects API mode (0-inference, 1-training)"     },
        { "attnDataType",    "%d",  offsetof(testOpts, attnDataType),    "selects data format (0-FP16, 1-FP32, 2-FP64)"   },
        { "attnCompPrec",    "%d",  offsetof(testOpts, attnCompPrec),    "selects math format (0-FP16, 1-FP32, 2-FP64)"   },
        { "attnNumHeads",    "%d",  offsetof(testOpts, attnNumHeads),    "number of attenton heads"                       },
        { "attnBatchSize",   "%d",  offsetof(testOpts, attnBatchSize),   "batch size for Q, R, K, V and O arguments"      },
        { "attnBeamSize",    "%d",  offsetof(testOpts, attnBeamSize),    "number of sentence candidates in Q, R inputs"   },
        { "attnSmScaler",    "%lg", offsetof(testOpts, attnSmScaler),    "softmax smoothing or sharpening coefficient"    },
        { "attnDropoutRate", "%g",  offsetof(testOpts, attnDropoutRate), "dropout rate settings applied during training"  },
        { "attnQsize",       "%d",  offsetof(testOpts, attnQsize),       "original vector length for 'queries'"           },
        { "attnKsize",       "%d",  offsetof(testOpts, attnKsize),       "original vector length for 'keys'"              },
        { "attnVsize",       "%d",  offsetof(testOpts, attnVsize),       "original vector length for 'values'"            },
        { "attnProjQsize",   "%d",  offsetof(testOpts, attnProjQsize),   "length of 'queries' vector after projection"    },
        { "attnProjKsize",   "%d",  offsetof(testOpts, attnProjKsize),   "length of 'keys' vector after projection"       },
        { "attnProjVsize",   "%d",  offsetof(testOpts, attnProjVsize),   "length of 'values' vector after projection"     },
        { "attnProjOsize",   "%d",  offsetof(testOpts, attnProjOsize),   "length of 'output' vector after projection"     },
        { "attnSeqLenQ",     "%d",  offsetof(testOpts, attnSeqLenQ),     "largest sequence length for Q, R, O arguments"  },
        { "attnSeqLenK",     "%d",  offsetof(testOpts, attnSeqLenK),     "largest sequence length for K, V arguments"     },
        { "attnDataLayout",  "%d",  offsetof(testOpts, attnDataLayout),  "data layout for Q, K, V, O inputs"              },
        { "attnResLink",     "%d",  offsetof(testOpts, attnResLink),     "enable/disable residual connections"            },
        { "attnProjBias",    "%d",  offsetof(testOpts, attnProjBias),    "enable/disable projection biases"               },
        { "attnSweep",       "%d",  offsetof(testOpts, attnSweep),       "sweep all time-steps in one inference API call" },
        { "attnRandGeom",    "%d",  offsetof(testOpts, attnRandGeom),    "randomize attention task dimensions"            },
        { "attnRandSeed",    "%d",  offsetof(testOpts, attnRandSeed),    "seed for the random number generator"           },
        { "attnFileDump",    "%d",  offsetof(testOpts, attnFileDump),    "dump weights/data to file for single sentence"  },
    };

    if (argc == 1) {
        printf("This is the cuDNN multi-head attention API test.\n\n");
        printf("Usage: ./%s [OPTIONS]\n\nProgram options:\n\n", baseFile(*argv));

        for (int i = 0; i < COUNTOF(param); i++) {
            char buf[64];
            sprintf(buf, "-%s<%s>", param[i].name, param[i].fmt);
            printf("%-20s - %s\n", buf, param[i].desc);
        }
        printf("\n");

        exit(-1);
    }

    while (argc > 1) {
        argc--;
        argv++;

        int i;

        for (i = 0; i < COUNTOF(param); i++) {
            const char *pname = param[i].name;
            size_t plen = strlen(pname);
            if (strncmp(*argv + 1, pname, plen) == 0) {
                int count = sscanf(*argv + plen + 1, param[i].fmt, (char*)opts + param[i].offs);
                if (count != 1) {
                    fprintf(stderr, "ERROR: missing numerical argument in option '%s'\n\n", *argv);
                    exit(-1);
                }
                break;
            }
        }

        if (i >= COUNTOF(param)) {
            fprintf(stderr, "ERROR: unknown switch '%s'\n\n", *argv);
            exit(-1);
        }
    }
}

typedef void (*do_test_fp) (testOpts &opts);

// int main(int argc, char **argv) {
//     testOpts opts;

//     printf("Executing: %s", baseFile(argv[0]));
//     for (int i = 1; i < argc; i++) {
//         printf(" %s", argv[i]);
//     }
//     printf("\n\n");

//     // Default test parameters to be overwritten by user cmd line options.
//     opts.attnTrain       = 0;
//     opts.attnDataType    = CUDNN_DATA_FLOAT;
//     opts.attnCompPrec    = CUDNN_DATA_FLOAT;
//     opts.attnQueryMap    = 0;
//     opts.attnNumHeads    = 2;
//     opts.attnBeamSize    = 3;
//     opts.attnSmScaler    = 1.0;
//     opts.attnDropoutRate = 0.0;
//     opts.attnQsize       = 32;
//     opts.attnKsize       = 32;
//     opts.attnVsize       = 32;
//     opts.attnProjQsize   = 8;
//     opts.attnProjKsize   = 8;
//     opts.attnProjVsize   = 8;
//     opts.attnProjOsize   = 8;
//     opts.attnSeqLenQ     = 24;
//     opts.attnSeqLenK     = 20;
//     opts.attnBatchSize   = 4;
//     opts.attnDataLayout  = 0;
//     opts.attnResLink     = 0;
//     opts.attnProjBias    = 0;
//     opts.attnSweep       = 0;
//     opts.attnRandGeom    = 0;
//     opts.attnRandSeed    = 1234;
//     opts.attnFileDump    = 0;

//     parseAttnParameters(argc, argv, &opts);

//     static do_test_fp func_arr[] = {
//         doTest<false, float,  float >,
//         doTest<true,  float,  float >,
//     };

//     int func_idx = -1;

//     if (opts.attnDataType == CUDNN_DATA_HALF) {
//         if (opts.attnCompPrec == CUDNN_DATA_HALF) {
//             func_idx = 0;
//         } else if (opts.attnCompPrec == CUDNN_DATA_FLOAT){
//             func_idx = 2;
//         }
//     } else if (opts.attnDataType == CUDNN_DATA_FLOAT && opts.attnCompPrec == CUDNN_DATA_FLOAT) {
//         func_idx = 4;
//     } else if (opts.attnDataType == CUDNN_DATA_DOUBLE && opts.attnCompPrec == CUDNN_DATA_DOUBLE) {
//         func_idx = 6;
//     }

//     if (func_idx < 0) {
//         fprintf(stderr, "ERROR: -attnDataType%d and -attnCompPrec%d are not supported\n\n", opts.attnDataType, opts.attnCompPrec);
//         exit(-1);
//     }

//     func_idx += (opts.attnTrain == 0 ? 0 : 1);

//     // Any error will cause the program to terminate with a non-zero exit code.
//     func_arr[func_idx](opts);

//     return 0;
// }

