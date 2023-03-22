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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda.h>
#include <cudnn.h>

inline void checkCudaError(cudaError_t code, const char *expr, const char *file, int line) {
    if (code) {
        fprintf(stderr, "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
                file, line, (int)code, cudaGetErrorString(code), expr);
        exit(1);
    }
}

inline void checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line) {
    if (code) {
        fprintf(stderr, "CUDNN error at %s:%d, code=%d (%s) in '%s'\n\n",
                file, line, (int)code, cudnnGetErrorString(code), expr);
        exit(1);
    }
}

#define CHECK_CUDA_ERR(...)                                             \
    do {                                                                \
        checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  \
    } while (0)

#define CHECK_CUDNN_ERR(...)                                            \
    do {                                                                \
        checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

struct testOpts {
    testOpts() { memset(this, 0, sizeof(*this)); }
    int attnTrain;
    int attnDataType;
    int attnCompPrec;
    int attnQueryMap;
    int attnNumHeads;
    int attnBatchSize;
    int attnBeamSize;
    double attnSmScaler;
    float attnDropoutRate;
    int attnQsize;
    int attnKsize;
    int attnVsize;
    int attnProjQsize;
    int attnProjKsize;
    int attnProjVsize;
    int attnProjOsize;
    int attnSeqLenQ;
    int attnSeqLenK;
    int attnDataLayout;
    int attnResLink;
    int attnProjBias;
    int attnSweep;
    int attnRandGeom;
    int attnRandSeed;
};

struct attnConfig {
    attnConfig() { memset(this, 0, sizeof(*this)); }  // sets queryMap=ALL_TO_ONE

    cudnnAttnQueryMap_t queryMap;  // queryMap mode

    int numHeads;       // number of attention heads
    int beamSize;       // number of candidates of the same sentence
    double smScaler;    // softmax smoothing or sharpening coefficient
    float dropoutRate;  // dropout probability
    int qSize;          // original vector length of "queries"
    int kSize;          // original vector length of "keys"
    int vSize;          // original vector length of "values"
    int qProjSize;      // "queries" after projection (0=no projection)
    int kProjSize;      // "keys" after projection (0=no projection)
    int vProjSize;      // "values" after projection (0=no projection)
    int oProjSize;      // "output" after projection (0=no projection)
    int seqLenQ;        // max seq length for Q, R, O buffers
    int seqLenK;        // max seq length for K, V buffers
    int batchSize;      // batch size for Q, R, K, V, O buffers
    bool resLink;       // enable/disable residual connections
    bool projBias;      // enable/disable residual connections
    int sweep;          // sweep all time-steps in inference mode
    int randGeom;       // randomize poblem dimensions
    int randSeed;       // random number generator seed

    unsigned attnMode;  // Attention Mode parameter

    // Attention window boundaries for every time-step.
    int *loWinIdx;
    int *hiWinIdx;

    // Query and key sequence lengths (for each batch/beam sentence).
    int *qSeqLen;
    int *kSeqLen;

    int dataLayout;                                        // data layout, map to one of 6 possible dataAxes
    cudnnSeqDataAxis_t dataAxes[CUDNN_SEQDATA_DIM_COUNT];  // data order for T, N, and B dim

    cudnnDataType_t dataType;  // data type for Q,K,V inputs, weights, output
    cudnnDataType_t compPrec;  // compute precision

    int qLength() {
        return this->qProjSize > 0 ? this->qProjSize : this->qSize;
    }

    int kLength() {
        return this->kProjSize > 0 ? this->kProjSize : this->kSize;
    }

    int vLength() {
        return this->vProjSize > 0 ? this->vProjSize : this->vSize;
    }

    int oLength() {
        return this->oProjSize > 0 ? this->oProjSize : this->vLength() * this->numHeads;
    }

    size_t qoTokens() {
        return size_t(this->seqLenQ) * this->batchSize * this->beamSize;
    }

    size_t kvTokens() {
        size_t t = size_t(this->seqLenK) * this->batchSize;
        if (this->queryMap == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE) {
            t *= this->beamSize;
        }
        return t;
    }

    size_t qAllData() {
        return this->qoTokens() * this->qSize;
    }

    size_t kAllData() {
        return this->kvTokens() * this->kSize;
    }

    size_t vAllData() {
        return this->kvTokens() * this->vSize;
    }

    size_t oAllData() {
        return this->qoTokens() * this->oLength();
    }

    size_t qAllWeights() {
        size_t q_weights = (this->qProjSize > 0 ? size_t(this->qSize) * this->qProjSize : 0);
        return q_weights * this->numHeads;
    }

    size_t kAllWeights() {
        size_t k_weights = (this->kProjSize > 0 ? size_t(this->kSize) * this->kProjSize : 0);
        return k_weights * this->numHeads;
    }

    size_t vAllWeights() {
        size_t v_weights = (this->vProjSize > 0 ? size_t(this->vSize) * this->vProjSize : 0);
        return v_weights * this->numHeads;
    }

    size_t oAllWeights() {
        size_t o_weights = (this->oProjSize > 0 ? size_t(this->vLength()) * this->oProjSize : 0);
        return o_weights * this->numHeads;
    }

    size_t qSeqLenCount() {
        return this->batchSize * this->beamSize;
    }

    size_t kSeqLenCount() {
        return this->batchSize * (this->queryMap == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE ? this->beamSize : 1);
    }
};

template <bool IS_TRAINING, typename T_ELEM, typename T_MATH>
class MultiheadAttentionTest {
   public:
    cudnnHandle_t handle;

    attnConfig mainCfg;

    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;

    // Attention in/out buffers on the GPU side.
    T_ELEM *devQ;
    T_ELEM *devK;
    T_ELEM *devV;
    T_ELEM *devO;
    T_ELEM *devDQ;
    T_ELEM *devDK;
    T_ELEM *devDV;
    T_ELEM *devDO;
    T_ELEM *devW;
    T_ELEM *devDW;

    // Buffers with in/out data and weights on the CPU side.
    T_ELEM *hostQ;
    T_ELEM *hostK;
    T_ELEM *hostV;
    T_ELEM *hostO;
    T_ELEM *hostDQ;
    T_ELEM *hostDK;
    T_ELEM *hostDV;
    T_ELEM *hostDO;
    T_ELEM *hostW;
    T_ELEM *hostDW;

    // Work-space and reserve-space GPU buffers required by API.
    T_MATH *devWkspace;
    T_MATH *devReserve;

    // Capacity of weight/wkspace/reserve buffers (in bytes).
    size_t maxWeights;
    size_t maxWkspace;
    size_t maxReserve;

    // Capacity of each "seq" data container (in elements).
    size_t maxElemQ;
    size_t maxElemK;
    size_t maxElemV;
    size_t maxElemO;
    size_t maxElemA;

    size_t maxElemQbar;
    size_t maxElemKbar;
    size_t maxElemVbar;
    size_t maxElemHbar;

    // Dropout descriptor settings.
    size_t dropoutBufSize;
    void *dropoutBuf;

    // Sequence length arrays for Q,R,O and K,V.
    int *qSeqArray;
    int *kSeqArray;

    int *devQSeqArray;
    int *devKSeqArray;

    // Attention window.
    int *loWinIdx;
    int *hiWinIdx;

    void setup(testOpts &opts);

    void run();

    void teardown(void);

    void testgen(attnConfig *testDesc);
};
