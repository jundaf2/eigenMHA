#pragma once
#include "model_base.h"
#include "util.h"
#include "fmha_kernel.h"
#include <gemm.h>
#include <utils.h>

template<int S, int D, int STEP, int WARPS_M, int WARPS_N, uint32_t FLAGS = 0x08u, typename elem_type_=__half> // S, D, STEP = 64, 64, 16
struct FMHA_kernel_traits {

    // this is defined (1) for both gmem and smem (2) for the 2 GEMM output P & O
    // The CTA description for the 1st GEMM.
    using Cta_tile_p = fmha::Cta_tile_extd<STEP, S, D, WARPS_M, WARPS_N, 1>; 
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = fmha::Cta_tile_extd<STEP, D, S, WARPS_M, 1, WARPS_N>;

    // Do we use one buffer for K and V.
    static constexpr bool SHARE_SMEM_FOR_K_AND_V = false;// (FLAGS & 0x08u) != 0u;
    // Do we keep K in registers.
    static constexpr bool K_IN_REGS = true;//(FLAGS & 0x10u) == 0u;
    // Do we keep V in registers.
    static constexpr bool V_IN_REGS = true;//(FLAGS & 0x100u) == 0u;

    // The global memory tile to load Q.
    using Gmem_tile_q = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>; // COL = D = 64, STEP is number of Rows of the tile, D is the Dim of head

    // The shared memory tile to swizzle Q.
    using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 1>;
    // using Smem_tile_q = fmha::Smem_tile_a<Cta_tile_p, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_B, S, D>; // S = 64 --> outer loop
    // The shared memory tile to swizzle K.
    using Smem_tile_k = fmha::Smem_tile_b<Cta_tile_p, fmha::Col>;

    // The global memory tile to load V.
    using Gmem_tile_v = fmha::Gmem_tile_qkv<Cta_tile_o, fmha::BITS_PER_ELEMENT_B, S, D>;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = fmha::Smem_tile_v<Cta_tile_o>;

    // The global memory tile to store O.
    using Gmem_tile_o = fmha::Gmem_tile_o<Cta_tile_o>;
    // The shared memory tile for O.
    using Smem_tile_o = fmha::Smem_tile_o<Cta_tile_o>;;

    // The global memory tile to load/store S.
    using Gmem_tile_s = fmha::Gmem_tile_mma_s<Cta_tile_p>;

    // The shared memory tile to transpose S.
    // using Smem_tile_st = fmha::Smem_tile_mma_transposed<Cta_tile_p>;

    // using Gmem_tile_do = fmha::Gmem_tile_qkv<Cta_tile_p, fmha::BITS_PER_ELEMENT_A, STEP, D>;

    // The global memory tile to store the softmax sum.
    using Gmem_softmax_sum = fmha::Gmem_summary_stats<Cta_tile_p>;

    // The shared memory tile to store dp sum.
    // using Smem_dp_sum = fmha::Smem_tile_dp_sum<Gmem_tile_q, 2>; //2

    using elem_type = elem_type_;

    // Make sure the number of threads match.
    static_assert((int)Gmem_tile_o::THREADS_PER_ROW == (int)Smem_tile_o::THREADS_PER_ROW, "");

    // The number of threads.
    static constexpr int THREADS = Cta_tile_p::THREADS_PER_CTA;
    // Make sure the number of threads matches both CTAs.
    static_assert(THREADS == Cta_tile_o::THREADS_PER_CTA, "");

    // The amount of shared memory needed to load Q and K.
    static constexpr int BYTES_PER_SMEM_QK = Smem_tile_q::BYTES_PER_TILE + Smem_tile_k::BYTES_PER_TILE;
    // The extra amount of shared memory needed to load V.
    static constexpr int BYTES_PER_SMEM_V = SHARE_SMEM_FOR_K_AND_V ? 0u : Smem_tile_v::BYTES_PER_TILE;
    // The amount of shared memory needed for Q, K and V..
    static constexpr int BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V;
    // The amount of shared memory needed to load Q and store O.
    static constexpr int BYTES_PER_SMEM_QO = Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE;

    // The amount of shared memory needed for Q, K, V and O.
    static constexpr int BYTES_PER_SMEM = fmha::MaxConstexpr(BYTES_PER_SMEM_QKV, BYTES_PER_SMEM_QO);
    // Make sure we have enough shared memory.
    static_assert(Smem_tile_q::BYTES_PER_TILE + Smem_tile_o::BYTES_PER_TILE <= BYTES_PER_SMEM, "");
};


////////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

struct Qkv_params {
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    // size_t qkv_stride_in_elts;
    // size_t qkv_stride_in_bytes;
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t q_row_stride_in_elts;
    uint32_t k_row_stride_in_elts;
    uint32_t v_row_stride_in_elts;
    uint32_t q_head_stride_in_elts;
    uint32_t k_head_stride_in_elts;
    uint32_t v_head_stride_in_elts;

    // The number of heads.
    int h;
};

struct FMHA_fprop_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    // size_t o_stride_in_elts;
    // size_t o_stride_in_bytes;
    uint32_t o_row_stride_in_elts;
    uint32_t o_head_stride_in_elts;

    // The pointer to the O_tmp matrix, which holds O intermediate value during
    // the loop;
    void *__restrict__ o_tmp_ptr;

    // The pointer to the S matrix.
    void * __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    // int64_t s_stride_in_bytes;
    uint32_t s_stride_in_bytes;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr; // l(x) --> 3D: batch size, head num , max seq len

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_bmm1f;
    uint32_t scale_bmm1;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int *__restrict__ blockmask;

};

template<typename Kernel_params>
struct Launch_params{
    Launch_params(/*cudaDeviceProp * props_,*/
                  cudaStream_t stream_)
        : elts_per_thread(0)
        // , props(props_)
        , stream(stream_){
    }

    size_t elts_per_thread;

    //cudaDeviceProp * props;

    cudaStream_t stream;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
};

namespace fmha {
template<typename Kernel_traits>
struct Gemm_Q_K_base {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx) 
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx) {

    }

    __device__ inline void load_q() { // first
        smem_q.load(frag_q[0], 0);
    }

    __device__ inline void reload_q() {
        smem_q.load(frag_q[0], 0);
    }

    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];//[2][1]
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
};

template<typename Kernel_traits, bool K_in_regs, typename elem_type_=__half>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits> {

    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    // If V is stored in shared memory, we can't load K using the same shared memory.
    static_assert(Kernel_traits::V_IN_REGS);

    static constexpr int SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);

    // Q | K / V
    //   | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE 
                                    + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
                                               Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char * smem_, const int tidx) 
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){ // all
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) { // 4
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) { // 4
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            // fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            // fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
            fmha::gemm(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
    }

    __device__ inline void reload_k(){
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N]; // [4][1], Mma_tile_p::MMAS_K is LDGs and STSs
};



template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size(){
    return Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

template<typename Kernel_traits, bool Is_first, bool Is_last, typename Params> // Is_first = True, Is_last= True
inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int begin, int steps, const int loop_step_idx) { //begin=0

    // using elem_type = __half;
    using elem_type = typename Kernel_traits::elem_type;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    // using Gmem_tile_o_tmp = fmha::Gmem_tile_o<Cta_tile_o, 4>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    // using Smem_softmax_sum = typename Kernel_traits::Smem_dp_sum;

    using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS, elem_type>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x; // 0 - 128

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    // if( binfo.stop_early() ) return;
    // if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return; // if actual_seqlen_k <= loop_step_idx * 64, false at least when loop_step_idx = 0

    // ---------------------- //
    // Q K V O are all of [batch_seq_len, hidden_dim]
    // Gmem_tile_o is fmha::Gmem_tile_o<Cta_tile_o, 2>
    // Gmem_tile_o_tmp is fmha::Gmem_tile_o<Cta_tile_o, 4>
    // ---------------------- //
    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params.q_ptr, params.q_row_stride_in_elts, params.q_head_stride_in_elts, binfo, tidx, true); // q_row_stride_in_elts = 
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts, binfo, tidx); // o_row_stride_in_elts
    // Gmem_tile_o_tmp gmem_o_tmp(params.o_tmp_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts, binfo, tidx);
    
    // Allocate the global memory tile loader for S.
    // Gmem_tile_s gmem_s(params, binfo, tidx); // the global params.s_ptr
    // Gmem_softmax_sum gmem_softmax_lse(params.softmax_lse_ptr, params, tidx);

    // Wind gmem tiles to the correct position.
    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    const int begin_og = begin; // always 0
    const bool Is_causal = false;
    begin = Is_causal ? std::max(begin, loop_step_idx * Cta_tile_p::N / Cta_tile_p::M) : begin; // non-causal -> 0 vs causal -> loop_step_idx * 64 / 16 = outer loop idx * 4
    const int steps_og = steps; // num of steps of the inner loop = 4
    steps -= begin - begin_og; // steps -= loop_step_idx*4 (every time comsumes 4 steps)

    // inner loop movement
    gmem_q.move(begin);
    gmem_o.move(begin);
    // gmem_o_tmp.move(begin);
    // gmem_s.move(begin);
    // gmem_softmax_lse.move(begin);


    fmha::Mask<Cta_tile_p, Is_causal> mask(binfo, tidx, loop_step_idx); // apply to softmax and matrix S

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params.k_ptr, params.k_row_stride_in_elts, params.k_head_stride_in_elts, binfo, tidx, false);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params.v_ptr, params.v_row_stride_in_elts, params.v_head_stride_in_elts, binfo, tidx, false);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];
    
    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    // if (!Is_first) { // outer loop movement
    //     gmem_k.move(loop_step_idx);
    //     gmem_v.move(loop_step_idx);
    //     // gmem_s.move(loop_step_idx * steps_og);
    // }

    // PRINT GLOBAL LOCATION
    if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
        // printf("BATCH %d, HEAD %d, THREAD %d\n", blockIdx.x, blockIdx.y, threadIdx.x);

        // printf("gmem_q::row_stride_in_bytes %d\n", gmem_q.row_stride_in_bytes);
        // printf("gmem_k::row_stride_in_bytes %d\n", gmem_k.row_stride_in_bytes);
        // printf("gmem_v::row_stride_in_bytes %d\n", gmem_v.row_stride_in_bytes);

        // printf("gmem_q::THREADS_PER_ROW %d\n", gmem_q.THREADS_PER_ROW);
        // printf("gmem_k::THREADS_PER_ROW %d\n", gmem_k.THREADS_PER_ROW);
        // printf("gmem_v::THREADS_PER_ROW %d\n", gmem_v.THREADS_PER_ROW);


        // printf("gmem_s::BYTES_PER_STG %d\n", gmem_s.BYTES_PER_STG);
        // printf("gmem_s::MMAS_M %d\n", gmem_s.MMAS_M);
        // printf("gmem_s::MMAS_N %d\n", gmem_s.MMAS_N);
        // printf("gmem_s::M_PER_MMA_PER_CTA %d\n", gmem_s.M_PER_MMA_PER_CTA);
        // printf("gmem_s::N_PER_MMA_PER_CTA %d\n", gmem_s.N_PER_MMA_PER_CTA);
        // printf("gmem_s::THREADS_PER_CTA %d\n", gmem_s.THREADS_PER_CTA);
        // printf("gmem_s::BYTES_PER_ROW %d\n", gmem_s.BYTES_PER_ROW);
        // printf("gmem_s::LOOP_STRIDE_BYTES %d\n", gmem_s.LOOP_STRIDE_BYTES);
        
    }

    // PRINT SLB LOCATION
    if (false/*(threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)*/)  {

        printf("Smem_tile_q:BYTES_PER_STS: %d\n", Kernel_traits::Smem_tile_q::BYTES_PER_STS);
        printf("Smem_tile_k::BYTES_PER_STS %d\n", Kernel_traits::Smem_tile_k::BYTES_PER_STS);
        printf("Smem_tile_V::BYTES_PER_STS %d\n", Kernel_traits::Smem_tile_v::BYTES_PER_STS);

        printf("Smem_tile_q:ELEMENTS_PER_STS: %d\n", Kernel_traits::Smem_tile_q::ELEMENTS_PER_STS);
        printf("Smem_tile_k::ELEMENTS_PER_STS %d\n", Kernel_traits::Smem_tile_k::ELEMENTS_PER_STS);
        printf("Smem_tile_V::ELEMENTS_PER_STS %d\n", Kernel_traits::Smem_tile_v::ELEMENTS_PER_STS);

        printf("Smem_tile_q:N_WITH_PADDING: %d\n", Kernel_traits::Smem_tile_q::N_WITH_PADDING);
        printf("Smem_tile_k::N_WITH_PADDING %d\n", Kernel_traits::Smem_tile_k::N_WITH_PADDING);
        printf("Smem_tile_V::N_WITH_PADDING %d\n", Kernel_traits::Smem_tile_v::N_WITH_PADDING);

        printf("Smem_tile_q:BYTES_PER_ROW_BEFORE_PACKING: %d\n", Kernel_traits::Smem_tile_q::BYTES_PER_ROW_BEFORE_PACKING);
        printf("Smem_tile_k::BYTES_PER_ROW_BEFORE_PACKING %d\n", Kernel_traits::Smem_tile_k::BYTES_PER_ROW_BEFORE_PACKING);
        printf("Smem_tile_V::BYTES_PER_ROW_BEFORE_PACKING %d\n", Kernel_traits::Smem_tile_v::BYTES_PER_ROW_BEFORE_PACKING);

        printf("Smem_tile_q:BYTES_PER_ROW: %d\n", Kernel_traits::Smem_tile_q::BYTES_PER_ROW);
        printf("Smem_tile_k::BYTES_PER_ROW %d\n", Kernel_traits::Smem_tile_k::BYTES_PER_ROW);
        printf("Smem_tile_V::BYTES_PER_ROW %d\n", Kernel_traits::Smem_tile_v::BYTES_PER_ROW);

        printf("Smem_tile_q:ROWS: %d\n", Kernel_traits::Smem_tile_q::ROWS);
        printf("Smem_tile_k::ROWS %d\n", Kernel_traits::Smem_tile_k::ROWS);
        printf("Smem_tile_V::ROWS %d\n", Kernel_traits::Smem_tile_v::ROWS);

        printf("Smem_tile_q:THREADS_PER_ROW: %d\n", Kernel_traits::Smem_tile_q::THREADS_PER_ROW);
        printf("Smem_tile_k::THREADS_PER_ROW %d\n", Kernel_traits::Smem_tile_k::THREADS_PER_ROW);
        printf("Smem_tile_V::THREADS_PER_ROW %d\n", Kernel_traits::Smem_tile_v::THREADS_PER_ROW);

        printf("Smem_tile_q:STS_PER_ROW: %d\n", Kernel_traits::Smem_tile_q::STS_PER_ROW);
        printf("Smem_tile_k::STS_PER_ROW %d\n", Kernel_traits::Smem_tile_k::STS_PER_ROW);
        printf("Smem_tile_V::STS_PER_ROW %d\n", Kernel_traits::Smem_tile_v::STS_PER_ROW);

        printf("Smem_tile_q:ROWS_PER_STS: %d\n", Kernel_traits::Smem_tile_q::ROWS_PER_STS);
        printf("Smem_tile_k::ROWS_PER_STS %d\n", Kernel_traits::Smem_tile_k::ROWS_PER_STS);
        printf("Smem_tile_V::ROWS_PER_STS %d\n", Kernel_traits::Smem_tile_v::ROWS_PER_STS);

        printf("Smem_tile_q:STS_PER_COL: %d\n", Kernel_traits::Smem_tile_q::STS_PER_COL);
        printf("Smem_tile_k::STS_PER_COL %d\n", Kernel_traits::Smem_tile_k::STS_PER_COL);
        printf("Smem_tile_V::STS_PER_COL %d\n", Kernel_traits::Smem_tile_v::STS_PER_COL);

        printf("Smem_tile_q:STS: %d\n", Kernel_traits::Smem_tile_q::STS);
        printf("Smem_tile_k::STS %d\n", Kernel_traits::Smem_tile_k::STS);
        printf("Smem_tile_V::STS %d\n", Kernel_traits::Smem_tile_v::STS);

        printf("Smem_tile_q:PARTIAL_STORE: %d\n", Kernel_traits::Smem_tile_q::PARTIAL_STORE);
        printf("Smem_tile_k::PARTIAL_STORE %d\n", Kernel_traits::Smem_tile_k::PARTIAL_STORE);
        printf("Smem_tile_V::PARTIAL_STORE %d\n", Kernel_traits::Smem_tile_v::PARTIAL_STORE);

        printf("Smem_tile_q:STORING_THREADS: %d\n", Kernel_traits::Smem_tile_q::STORING_THREADS);
        printf("Smem_tile_k::STORING_THREADS %d\n", Kernel_traits::Smem_tile_k::STORING_THREADS);
        printf("Smem_tile_V::STORING_THREADS %d\n", Kernel_traits::Smem_tile_v::STORING_THREADS);

        printf("Smem_tile_q:STS_PER_ROW: %d\n", Kernel_traits::Smem_tile_q::STS_PER_ROW);
        printf("Smem_tile_k::STS_PER_ROW %d\n", Kernel_traits::Smem_tile_k::STS_PER_ROW);
        printf("Smem_tile_V::STS_PER_ROW %d\n", Kernel_traits::Smem_tile_v::STS_PER_ROW);

        printf("Smem_tile_q:BYTES_PER_BUFFER: %d\n", Kernel_traits::Smem_tile_q::BYTES_PER_BUFFER);
        printf("Smem_tile_k::BYTES_PER_BUFFER %d\n", Kernel_traits::Smem_tile_k::BYTES_PER_BUFFER);
        printf("Smem_tile_V::BYTES_PER_BUFFER %d\n", Kernel_traits::Smem_tile_v::BYTES_PER_BUFFER);

        printf("Smem_tile_q:BUFFERS_PER_TILE: %d\n", Kernel_traits::Smem_tile_q::BUFFERS_PER_TILE);
        printf("Smem_tile_k::BUFFERS_PER_TILE %d\n", Kernel_traits::Smem_tile_k::BUFFERS_PER_TILE);
        printf("Smem_tile_V::BUFFERS_PER_TILE %d\n", Kernel_traits::Smem_tile_v::BUFFERS_PER_TILE);

        printf("Smem_tile_q:BYTES_PER_TILE: %d\n", Kernel_traits::Smem_tile_q::BYTES_PER_TILE);
        printf("Smem_tile_k::BYTES_PER_TILE %d\n", Kernel_traits::Smem_tile_k::BYTES_PER_TILE);
        printf("Smem_tile_V::BYTES_PER_TILE %d\n", Kernel_traits::Smem_tile_v::BYTES_PER_TILE);

        printf("Smem_tile_q:ROWS_PER_XOR_PATTERN: %d\n", Kernel_traits::Smem_tile_q::ROWS_PER_XOR_PATTERN);
        printf("Smem_tile_k::ROWS_PER_XOR_PATTERN %d\n", Kernel_traits::Smem_tile_k::ROWS_PER_XOR_PATTERN);
        printf("Smem_tile_V::ROWS_PER_XOR_PATTERN %d\n", Kernel_traits::Smem_tile_v::ROWS_PER_XOR_PATTERN);

        printf("Smem_tile_q:COLS_PER_XOR_PATTERN: %d\n", Kernel_traits::Smem_tile_q::COLS_PER_XOR_PATTERN);
        printf("Smem_tile_k::COLS_PER_XOR_PATTERN %d\n", Kernel_traits::Smem_tile_k::COLS_PER_XOR_PATTERN);
        printf("Smem_tile_V::COLS_PER_XOR_PATTERN %d\n", Kernel_traits::Smem_tile_v::COLS_PER_XOR_PATTERN);

        
    }

    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for V.
    gmem_v.load();

    // if (!Is_first) { __syncthreads(); }
     __syncthreads();

    // float p_prev_lse[Mma_tile_p::MMAS_M * 2]; //[2]
    // if (!Is_first) {
    //     gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse)); // from gmem to reg frag
    // }

    
    // gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse)); // from gmem to reg frag


    // Commit the data for Q and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);

    // const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    // #pragma unroll
    // for(int it=0;it < Gmem_tile_k::LDGS;it++){
    //     gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    // }

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    gemm_q_k.load_q();

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N]; //[1][4]
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) { // 1
        smem_v.load(frag_v[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    // if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
    //     // Make sure we are done loading the fragments for K.
    //     __syncthreads();

    //     // Commit the data to shared memory for V.
    //     gmem_k.commit(gemm_q_k.smem_k);

    //     // Make sure the data is in shared memory.
    //     __syncthreads();
    // }

    // Load the fragments for K. 
    gemm_q_k.load_k();

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_SOFTMAX], tidx); // include both sum and max

    // Smem_softmax_sum smem_softmax_lse(reinterpret_cast<float *>(&smem_[Gemm1::SMEM_BYTES]), tidx);

    // Load over the entire sequence length.
    for( int l = 0; l < steps; l++ ) {
        if((begin + l) * Cta_tile_p::M >= binfo.actual_seqlen_q) break; // if the seq len is not sufficiently enough

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N]; // [1][1]
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P = Q * K^T. 
        gemm_q_k(acc_p);

        // printf("gmem_q::row col row_offset %d %d %d\n", gmem_q.row_, gmem_q.col_, gmem_q.row_offset_);
        // printf("gmem_k::row col row_offset %d %d %d\n", gmem_k.row_, gmem_k.col_, gmem_k.row_offset_);
        // printf("gmem_v::row col row_offset %d %d %d\n", gmem_v.row_, gmem_v.col_, gmem_v.row_offset_);
        // if((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)){
        //     printf("l in steps: %d\n", l);
        // }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        // //     printf("acc_p.NUM_REGS %d\n",acc_p[0][0].NUM_REGS);
        // //     printf("acc_p.NUM_ELTS %d\n",acc_p[0][0].NUM_ELTS);
        // //     printf("gemm_q_k.frag_q[0][0].NUM_REGS %d\n",gemm_q_k.frag_q[0][0].NUM_REGS);
        // //     printf("gemm_q_k.frag_q[0][0].NUM_ELTS %d\n",gemm_q_k.frag_q[0][0].NUM_ELTS);
        // //     printf("gemm_q_k.frag_k[0][0].NUM_REGS %d\n",gemm_q_k.frag_k[0][0].NUM_REGS);
        // //     printf("gemm_q_k.frag_k[0][0].NUM_ELTS %d\n",gemm_q_k.frag_k[0][0].NUM_ELTS);
        //     // printf("Mma_tile_p::MMAS_M Mma_tile_p::MMAS_N Mma_tile_p::MMAS_K  %d %d %d\n",Mma_tile_p::MMAS_M,Mma_tile_p::MMAS_N,Mma_tile_p::MMAS_K);
        //     // printf("Mma_tile_o::MMAS_M Mma_tile_o::MMAS_N Mma_tile_o::MMAS_K  %d %d %d\n",Mma_tile_o::MMAS_M,Mma_tile_o::MMAS_N,Mma_tile_o::MMAS_K);
        //     printf("Flash-Attention first GEMM input Q tidx=0 gemm_q_k.frag_q[0][0].elt(0)=%.6f gemm_q_k.frag_q[0][0].elt(1)=%.6f\n", __half2float(__ushort_as_half(gemm_q_k.frag_q[0][0].elt(0))), __half2float(__ushort_as_half(gemm_q_k.frag_q[0][0].elt(1))));
        //     printf("Flash-Attention first GEMM input K tidx=0 gemm_q_k.frag_k[0][0].elt(0)=%.6f gemm_q_k.frag_k[0][0].elt(1)=%.6f\n", __half2float(__ushort_as_half(gemm_q_k.frag_k[0][0].elt(0))), __half2float(__ushort_as_half(gemm_q_k.frag_k[0][0].elt(1))));
        //     printf("Flash-Attention first GEMM output P tidx=0 acc_p[0][0].elt(0)=%.6f acc_p[0][0].elt(1)=%.6f acc_p[0][0].elt(2)=%.6f acc_p[0][0].elt(3)=%.6f acc_p[0][0].elt(4)=%.6f acc_p[0][0].elt(5)=%.6f acc_p[0][0].elt(6)=%.6f acc_p[0][0].elt(7)=%.6f\n", acc_p[0][0].elt(0), acc_p[0][0].elt(1), acc_p[0][0].elt(2), acc_p[0][0].elt(3), acc_p[0][0].elt(4), acc_p[0][0].elt(5), acc_p[0][0].elt(6), acc_p[0][0].elt(7));
        // }

        //  if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("Flash-Attention first GEMM output S[0][0] tidx=0 acc_p[0][0].elt(0)=%.6f \n", acc_p[0][0].elt(0)/8);
        //     printf("Flash-Attention first GEMM output S[0][1]  tidx=0 acc_p[0][0].elt(1)=%.6f \n", acc_p[0][0].elt(1)/8);
        // }
        // if ((threadIdx.x == 1) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("Flash-Attention first GEMM output S[0][2] tidx=1 acc_p[0][0].elt(0)=%.6f \n", acc_p[0][0].elt(0)/8);
        //     printf("Flash-Attention first GEMM output S[0][3] tidx=1 acc_p[0][0].elt(1)=%.6f \n", acc_p[0][0].elt(1)/8);
        // }


        /***********************************************************
         * The following implements:
         * (1) the row softmax
         * (2) couples the softmax result of this loop with previous loops and accumulates to the Output
         * ***********************************************************/

        uint4 out[Gmem_tile_o::STGS_PER_LOOP]; // [2], 16 bytes, 8 half
        // if (!Is_first) { gmem_o_tmp.load(out, 0); } // load from out[0*STGS_PER_LOOP]

        // Trigger the load for the next Q values.
        if( l < steps - 1) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load();
        }

        // Load the mask for that iteration.
        mask.load(begin + l); // calculate row offset

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p); // assign acc_p to softmax

        // Apply the mask.
        softmax.apply_mask(mask); // if is not valid, then apply -INF mask

        // if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
        //     // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
        //     __syncthreads();
        // }

        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2]; //[2], m_ij
        // if (!Is_first) {
            // smem_softmax_lse.store_pair(p_prev_lse, l % 2); // store p_prev_lse from regs to slb
            // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi]; }

            // // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi] / params.scale_bmm1f; }  // params.scale_bmm1f =  1 / sqrt(headdim)
        // }
        // smem_softmax_lse.store_pair(p_prev_lse, l % 2); // store p_prev_lse from regs to slb
        // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi]; }

        // Trigger the load for the next LSE values.
        // if( l < steps - 1) {
        //     if (!Is_first) {
        //         gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
        //     }
        // }
        // gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));

        softmax.template reduce_max</*zero_init=*/Is_first>(p_max); // get the max of the whole line of S, before scale and exp
        
        // Compute the exponential value.
        // softmax.apply_exp(p_max); // softmax exp
        softmax.scale_apply_exp(p_max, params.scale_bmm1f); // softmax -> scale and exp, f(x)


        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2]; // [2], l_ij
        // if (!Is_first) {
        //     int warp = tidx / Cta_tile_p::THREADS_PER_WARP;
        //     int lane = tidx % Cta_tile_p::THREADS_PER_WARP;
        //     for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) {
        //         p_sum[mi] = ((warp == 0) && (lane % 4 == 0)) ? expf(p_prev_lse[mi] - p_max[mi]) : 0;
        //     }
        // }
        // softmax.reduce_sum(p_sum);
        softmax.reduce_sum_before_sync_(p_sum); // reduce [2][4] softmax to p_sum [2] and store to SLB (partial sum)
        // softmax.template reduce_sum_before_sync_</*zero_init=*/Is_first>(p_sum);

        // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum));
        // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum_log));
        // gmem_softmax_lse.move();

        // Finalize softmax on the accumulators of P^T.
        // softmax.scale(p_sum);
        

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M]; // [1][1]
        static_assert(Mma_tile_o::MMAS_M == Mma_tile_p::MMAS_M); // 1
        static_assert(Mma_tile_o::MMAS_K == Mma_tile_p::MMAS_N); // 1
        softmax.template pack<elem_type>(frag_p); // softmax uses float, reduce precisions and pack to the half2 (uint32) in frag_p [from reg to reg], the one in frag_p is f(x)

        // gmem_s.store(frag_p, mask);
        // gmem_s.move();

        // Commit the values for Q into shared memory.
        if(l < steps - 1) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];// [1][4]
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        
        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) { // 1
            // fmha::gemm_cl<elem_type>(acc_o, frag_p[ki], frag_v[ki]);
            fmha::gemm(acc_o, frag_p[ki], frag_v[ki]); // ([1][4],[1],[4])
        }

        // The mapping from tidx to rows changes between the softmax and the
        // O-reduction. So we recalculate the max.
        // float p_max_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M]; // [2][1]
        int rows[Gmem_tile_o::STGS_PER_LOOP]; // [2]
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            rows[jj] = tidx / Gmem_tile_o::THREADS_PER_ROW + jj * Gmem_tile_o::ROWS_PER_STG; // indices of SMEM row, tidx / 16 + jj * 8 -> [0,7]+0/8 a total of 16
        }


        // When d = 16, O only has 16 x 16 = 256 elements, and each of the 128 threads wants
        // to write 4 elements, so only half of the thread should deal with O.
        bool o_rows_are_valid = true;
            // (Kernel_traits::THREADS <= Gmem_tile_o::THREADS_PER_ROW * Gmem_tile_o::ROWS)
            // || (tidx / Gmem_tile_o::THREADS_PER_ROW < Gmem_tile_o::ROWS); 

        // if((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))
        // {
        //     printf("o_rows_are_valid: ");
        //     printf((o_rows_are_valid ? "true" : "false"));
        //     printf("\n");
        // }

        // if (o_rows_are_valid) {
        //     softmax.reduce_max_after_sync_(p_max_o, rows); // the data in the SLB_max are the non-scaled version
        // }
        // static_assert(Mma_tile_o::MMAS_M == 1);
        // for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //     p_max_o[jj][0] *= params.scale_bmm1f;
        // }

        // float p_prev_scale_o[Gmem_tile_o::STGS_PER_LOOP]; // 2
        
        // if ((!Is_first) && o_rows_are_valid) {
        //     smem_softmax_lse.load(p_prev_scale_o, rows, l % 2);
        // }

        static_assert(Gmem_tile_o::LOOPS == 1);

        // Swizzle the elements and do the final reduction.
        smem_o.store(acc_o, 0);

        // Make sure the data is in shared memory.
        __syncthreads();
        

        static_assert(Mma_tile_o::MMAS_M == 1);
        float p_sum_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M]; // [2][1], l(x)
        if (o_rows_are_valid) {
            softmax.reduce_sum_after_sync_(p_sum_o, rows); // get the sum of row of S, after scale and exp, l(x)
        }
        // if (!Is_first) {
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         p_prev_scale_o[jj] = expf(p_prev_scale_o[jj] - p_max_o[jj][0]);
        //         p_sum_o[jj][0] += p_prev_scale_o[jj];
        //     }
        // }

        // float p_sum_log[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M]; // [2][1]
        // #pragma unroll
        // for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) { // 2
        //     float sum = p_sum_o[jj][0];
        //     p_sum_log[jj][0] = (sum == 0.f || sum != sum) ? -INFINITY : p_max_o[jj][0] + __logf(sum);

        //     if ((tidx % Gmem_tile_o::THREADS_PER_ROW == 0) && o_rows_are_valid) {
        //         gmem_softmax_lse.store_row(
        //             reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M]>(p_sum_log[jj]), rows[jj]);
        //     }
        // }
        // gmem_softmax_lse.move();

        // Load from shared memory.
        // if (!Is_first) {
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         out[jj] = fmha::fmul4(out[jj], p_prev_scale_o[jj]); // uint4 * float
        //     }
        // }

        smem_o.template load</*zero_init=*/true>(out); // Is_first, load & sum

        // const bool is_final_write = true;
            // Is_last
            // || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
            // || ((Is_causal) && ((begin + l) * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
        
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) { // 2
            float sum = p_sum_o[jj][0]; // sum
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            out[jj] = fmha::fmul4(out[jj], inv_sum);
        }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("Flash-Attention second GEMM output O[0][0] tidx=0 out[0].x=%.6f \n", reinterpret_cast<const float&>(out[0].x));
        //     printf("Flash-Attention second GEMM output O[0][1]  tidx=0 out[0].y=%.6f \n", reinterpret_cast<const float&>(out[0].y));
        //     printf("Flash-Attention second GEMM output O[0][2] tidx=0 out[0].z=%.6f \n", reinterpret_cast<const float&>(out[0].z));
        //     printf("Flash-Attention second GEMM output O[0][3] tidx=0 out[0].w=%.6f \n", reinterpret_cast<const float&>(out[0].w));
        // }

        // Output the values.
        // if (is_final_write) {
        //     gmem_o.template store<elem_type>(out, 0);
        //     gmem_o.move();
        // } else {
        //     gmem_o_tmp.store(out, 0);
        // }
        gmem_o.template store<elem_type>(out, 0); // convert to half
        gmem_o.move();

        // Move to the next part of the output.
        // if (!(Is_first && Is_last)) { gmem_o_tmp.move(); }

        // Make sure we are reading from the correct buffer.
        gemm_q_k.smem_q.move_to_next_read_buffer();
        // Trigger the load from shared memory for the next series of Q values.
        if(l < steps - 1) {
            gemm_q_k.reload_q();
        }

    }  // Outer loop over the sequence length.
}

template<typename Kernel_traits>
__global__ void fmha_fprop_fp16_sm80_loop_kernel(FMHA_fprop_params params) {
    // The block index for the batch.
    const int bidb = blockIdx.x; // [0] for test, [0-31] for benchmark
    // The block index for the head.
    const int bidh = blockIdx.y; // [0-7]
    // The thread index.
    const int tidx = threadIdx.x; // [0-127]

    const int tidx_global = (bidb * params.h + bidh) * blockDim.x * 2 + tidx;  // for random number generation, not used in inference

    constexpr int M = Kernel_traits::Cta_tile_p::M;  // M is STEP, fixed to 16
    const int STEPS = (params.seqlen_q + M - 1) / M; // STEPS = 64/16 = 4

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N; // 64, Cta_tile_p::N is S (seq len) S is 128 in this case, or,  Cta_tile_o::N is D (head dim)  D is 16, 32, 64, 128

    // if or not to use several loops to process seqlen_k
    if (params.seqlen_k == blocksize_c) { // this
        fmha::device_1xN_<Kernel_traits, true, true>(params, bidb, bidh, 0, STEPS, 0);
    } else {
        const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
        fmha::device_1xN_<Kernel_traits, true, false>(params, bidb, bidh, 0, STEPS, 0);
        for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
            fmha::device_1xN_<Kernel_traits, false, false>(params, bidb, bidh, 0, STEPS, loop_step_idx);
        }
        fmha::device_1xN_<Kernel_traits, false, true>(params, bidb, bidh, 0, STEPS, max_loop_steps - 1);
    }
}


template<typename Kernel_traits>
void run_fmha_fp16_sm80_loop_(Launch_params<FMHA_fprop_params> &launch_params) {
  constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
  const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;

//   std::cout << "blocksize_c: " << blocksize_c << std::endl;
//   std::cout << "launch_params.params.seqlen_q: " << launch_params.params.seqlen_q << std::endl;
//   std::cout << "launch_params.params.seqlen_k: " << launch_params.params.seqlen_k << std::endl;  


    // using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // std::cout << "Kernel_traits::Cta_tile_p::M: " << Kernel_traits::Cta_tile_p::M << std::endl; 
    // std::cout << "Kernel_traits::Cta_tile_p::N: " << Kernel_traits::Cta_tile_p::N << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_p::K: " << Kernel_traits::Cta_tile_p::M << std::endl; 
    // std::cout << "Kernel_traits::Cta_tile_p::WARPS_M: " << Kernel_traits::Cta_tile_p::WARPS_M << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_p::WARPS_N: " << Kernel_traits::Cta_tile_p::WARPS_N << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_p::WARPS_K: " << Kernel_traits::Cta_tile_p::WARPS_K << std::endl; 
    // std::cout << "Kernel_traits::Mma_tile_p::MMAS_M: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>::MMAS_M << std::endl;
    // std::cout << "Kernel_traits::Mma_tile_p::MMAS_N: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>::MMAS_N << std::endl;
    // std::cout << "Kernel_traits::Mma_tile_p::MMAS_K: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>::MMAS_K << std::endl; 


    // std::cout << "Kernel_traits::Cta_tile_o::M: " << Kernel_traits::Cta_tile_o::M << std::endl; 
    // std::cout << "Kernel_traits::Cta_tile_o::N: " << Kernel_traits::Cta_tile_o::N << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_o::K: " << Kernel_traits::Cta_tile_o::M << std::endl; 
    // std::cout << "Kernel_traits::Cta_tile_o::WARPS_M: " << Kernel_traits::Cta_tile_o::WARPS_M << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_o::WARPS_N: " << Kernel_traits::Cta_tile_o::WARPS_N << std::endl;
    // std::cout << "Kernel_traits::Cta_tile_o::WARPS_K: " << Kernel_traits::Cta_tile_o::WARPS_K << std::endl; 
    // std::cout << "Kernel_traits::Mma_tile_o::MMAS_M: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_o>::MMAS_M << std::endl;
    // std::cout << "Kernel_traits::Mma_tile_o::MMAS_N: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_o>::MMAS_N << std::endl;
    // std::cout << "Kernel_traits::Mma_tile_o::MMAS_K: " << fmha::Hmma_tile<typename Kernel_traits::Cta_tile_o>::MMAS_K << std::endl; 

    // std::cout << "Kernel_traits::Gmem_tile_o::BYTES_PER_STG: " << Kernel_traits::Gmem_tile_o::BYTES_PER_STG << std::endl; 
    // std::cout << "Kernel_traits::Gmem_tile_o::COLS: " << Kernel_traits::Gmem_tile_o::COLS << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::BYTES_PER_ROW: " << Kernel_traits::Gmem_tile_o::BYTES_PER_ROW << std::endl; 
    // std::cout << "Kernel_traits::Gmem_tile_o::THREADS_PER_ROW: " << Kernel_traits::Gmem_tile_o::THREADS_PER_ROW << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::ROWS: " << Kernel_traits::Gmem_tile_o::ROWS << std::endl; 
    // std::cout << "Kernel_traits::Gmem_tile_o::ROWS_PER_LOOP: " << Kernel_traits::Gmem_tile_o::ROWS_PER_LOOP << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::LOOPS: " << Kernel_traits::Gmem_tile_o::LOOPS << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::ROWS_PER_STG: " << Kernel_traits::Gmem_tile_o::ROWS_PER_STG << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::STGS_PER_LOOP: " << Kernel_traits::Gmem_tile_o::STGS_PER_LOOP << std::endl;
    // std::cout << "Kernel_traits::Gmem_tile_o::STGS: " << Kernel_traits::Gmem_tile_o::STGS << std::endl;

    // // 
    // std::cout << "Kernel_traits::Smem_tile_o::BYTES_PER_ELEMENT: " << Kernel_traits::Smem_tile_o::BYTES_PER_ELEMENT << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::BYTES_PER_STS: " << Kernel_traits::Smem_tile_o::BYTES_PER_STS << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::BYTES_PER_ROW: " << Kernel_traits::Smem_tile_o::BYTES_PER_ROW << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::BYTES_PER_LDS: " << Kernel_traits::Smem_tile_o::BYTES_PER_LDS << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::THREADS_PER_ROW: " << Kernel_traits::Smem_tile_o::THREADS_PER_ROW << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::ROWS: " << Kernel_traits::Smem_tile_o::ROWS << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::ROWS_PER_LOOP: " << Kernel_traits::Smem_tile_o::ROWS_PER_LOOP << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::LOOPS: " << Kernel_traits::Smem_tile_o::LOOPS << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::ROWS_PER_LDS: " << Kernel_traits::Smem_tile_o::ROWS_PER_LDS << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::LDS_PER_LOOP: " << Kernel_traits::Smem_tile_o::LDS_PER_LOOP << std::endl;
    // std::cout << "Kernel_traits::Smem_tile_o::BYTES_PER_TILE: " << Kernel_traits::Smem_tile_o::BYTES_PER_TILE << std::endl;

    // std::cout << "Kernel_traits::Gmem_softmax_sum::BYTES_PER_MMA: " << Kernel_traits::Gmem_softmax_sum::BYTES_PER_MMA << std::endl; 
    // std::cout << "Kernel_traits::Gmem_softmax_sum::ROWS: " << Kernel_traits::Gmem_softmax_sum::ROWS << std::endl;

    // std::cout << "Kernel_traits::Smem_dp_sum::ROWS: " << Kernel_traits::Smem_dp_sum::ROWS << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::THREADS_PER_ROW: " << Kernel_traits::Smem_dp_sum::THREADS_PER_ROW << std::endl;
    // std::cout << "Kernel_traits::Smem_dp_sum::MMAS_M: " << Kernel_traits::Smem_dp_sum::MMAS_M << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::ROWS_PER_LDG: " << Kernel_traits::Smem_dp_sum::ROWS_PER_LDG << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::LDGS: " << Kernel_traits::Smem_dp_sum::LDGS << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::ROWS_PER_MMA: " << Kernel_traits::Smem_dp_sum::ROWS_PER_MMA << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::BYTES_PER_BUFFER: " << Kernel_traits::Smem_dp_sum::BYTES_PER_BUFFER << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::BUFFERS_PER_TILE: " << Kernel_traits::Smem_dp_sum::BUFFERS_PER_TILE << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::BYTES_PER_TILE: " << Kernel_traits::Smem_dp_sum::BYTES_PER_TILE << std::endl; 
    // std::cout << "Kernel_traits::Smem_dp_sum::ROWS_PER_TILE_INC_BOUNDARY: " << Kernel_traits::Smem_dp_sum::ROWS_PER_TILE_INC_BOUNDARY << std::endl;
    
    // std::cout << "Kernel_traits::Softmax::WARPS_M: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::WARPS_M << std::endl; 
    // std::cout << "Kernel_traits::Softmax::WARPS_N: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::WARPS_N << std::endl; 
    // std::cout << "Kernel_traits::Softmax::MMAS_M: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::MMAS_M << std::endl; 
    // std::cout << "Kernel_traits::Softmax::MMAS_N: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::MMAS_N << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::ROWS: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::ROWS << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::COLS: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::COLS << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::ROWS_PER_XOR_PATTERN: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::ROWS_PER_XOR_PATTERN << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::BYTES_PER_TILE: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::BYTES_PER_TILE << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::ELTS_PER_TILE: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::ELTS_PER_TILE << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::THREADS_PER_GROUP: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::THREADS_PER_GROUP << std::endl; 
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::ROWS_PER_WARP: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::ROWS_PER_WARP << std::endl;
    // std::cout << "Kernel_traits::Softmax::Smem_tile_red::LOOPS: " << fmha::Softmax<typename Kernel_traits::Cta_tile_p, Kernel_traits>::Smem_tile_red::LOOPS << std::endl; 
    

    // configure
    using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
    constexpr int M = Kernel_traits::Cta_tile_p::M;
    size_t STEPS = (launch_params.params.seqlen_q + M - 1) / M; // 128/16=8
    constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
    constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;
    size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8 * loop_steps;
    launch_params.elts_per_thread = elts_per_head;


//   constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
  // Don't need smem_size_softmax_lse if we're not looping
  const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>() + 0;
    //   + (loop_steps > 1 ? smem_size_softmax_lse : 0); // + 0

  // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
  // https://github.com/kokkos/kokkos-kernels/issues/349
  // https://github.com/HazyResearch/flash-attention/issues/21
  auto kernel = &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits>;

  if( smem_size >= 48 * 1024 ) {
      FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

//   std::cout << "smem kernel launch (fmha_fprop_fp16_sm80_loop_kernel): " << smem_size*Kernel_traits::THREADS << std::endl;

  dim3 grid(launch_params.params.b, launch_params.params.h);   // gridDim.x = batch size, gridDim.y = head number
  kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(launch_params.params); // 4 warps and 128 threads
  FMHA_CHECK_CUDA(cudaStreamSynchronize(launch_params.stream));
  FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void set_params_fprop(FMHA_fprop_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      void *q_ptr, 
                      void *k_ptr, 
                      void *v_ptr,
                      const uint32_t q_stride_0,
                      const uint32_t k_stride_0,
                      const uint32_t v_stride_0,
                      const uint32_t q_stride_1,
                      const uint32_t k_stride_1,
                      const uint32_t v_stride_1,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *o_packed_d,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float softmax_scale) {

    Data_type acc_type = DATA_TYPE_FP16;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.q_row_stride_in_elts = q_stride_0;
    params.k_row_stride_in_elts = k_stride_0;
    params.v_row_stride_in_elts = v_stride_0;
    params.q_head_stride_in_elts = q_stride_1;
    params.k_head_stride_in_elts = k_stride_1;
    params.v_head_stride_in_elts = v_stride_1;
    params.o_ptr = o_packed_d;
    params.o_row_stride_in_elts = h * d;
    params.o_head_stride_in_elts = d;
    params.o_tmp_ptr = o_tmp_d;

    assert(params.o_row_stride_in_elts==params.q_row_stride_in_elts);
    assert(params.o_row_stride_in_elts==params.k_row_stride_in_elts);
    assert(params.o_row_stride_in_elts==params.v_row_stride_in_elts);

    assert(params.o_head_stride_in_elts==params.q_head_stride_in_elts);
    assert(params.o_head_stride_in_elts==params.k_head_stride_in_elts);
    assert(params.o_head_stride_in_elts==params.v_head_stride_in_elts);
    

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
}
}