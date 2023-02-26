/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "utils.h"
#include <gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////
// M = 16, N = 64, K = 64
// M_ N_ => M K, N K, K N
template<
    // The description of the tile computed by this CTA.
    typename Cta_tile,
    // The number of rows in the 2D shared memory buffer.
    int M_,
    // The number of cols.
    int N_,
    // The size in bits of each element.
    int BITS_PER_ELEMENT_,
    // The number of bytes per STS.
    int BYTES_PER_STS_ = 16,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1,
    // Do we enable the fast path for LDS.128 and friends.
    int ENABLE_LDS_FAST_PATH_ = 0,
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    int ROWS_PER_XOR_PATTERN_ = 8,
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    int COLS_PER_XOR_PATTERN_ = 1,
    // Use or not predicates
    bool USE_PREDICATES_ = true
>
struct Smem_tile_without_skews { // enum, runtime defined

    // The size in bits of each element.
    enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ }; // 2*8=16
    // The size in bytes of a single STS.
    enum { BYTES_PER_STS = BYTES_PER_STS_ }; // 16
    // The number of elements per STS.
    enum { ELEMENTS_PER_STS = BYTES_PER_STS * 8 / BITS_PER_ELEMENT }; // 16*8/(2*8) = 8
    // To support arbitrary N, we pad some values to a power-of-2.
    enum { N_WITH_PADDING = Next_power_of_two<N_>::VALUE }; // 64
    // The number of bytes per row without packing of rows.
    enum { BYTES_PER_ROW_BEFORE_PACKING = N_WITH_PADDING * BITS_PER_ELEMENT / 8 }; // 128 -> 8 uint4
    // The number of bytes per row -- we want at least 128B per row.
    enum { BYTES_PER_ROW = Max<BYTES_PER_ROW_BEFORE_PACKING, 128>::VALUE }; // 128 -> 8 uint4
    // The number of rows in shared memory (two rows may be packed into a single one).
    enum { ROWS = M_ * BYTES_PER_ROW_BEFORE_PACKING / BYTES_PER_ROW }; // 16 * 1 or 64 * 1 => 16/64

    // The number of threads per row.
    enum { THREADS_PER_ROW_UNBOUNDED = BYTES_PER_ROW / BYTES_PER_STS }; // 8
    // The number of threads per row.
    enum { THREADS_PER_ROW = Min<Cta_tile::THREADS_PER_CTA, THREADS_PER_ROW_UNBOUNDED>::VALUE }; // 8

    // The number of STS per row.
    enum { STS_PER_ROW = BYTES_PER_ROW / THREADS_PER_ROW / BYTES_PER_STS }; // 1 
    // It must be at least one.
    static_assert(STS_PER_ROW >= 1, "");
    // The number of rows written with a single STS.
    enum { ROWS_PER_STS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW }; // 16
    // Make sure we write to at least one row per STS. Thanks Dr. Obvious ;)
    static_assert(ROWS_PER_STS >= 1, "");
    // The number of STS needed to store all rows.
    enum { STS_PER_COL = Div_up<ROWS, ROWS_PER_STS>::VALUE }; // 16/16=1, 128/16=8, 64/8=8
    // The number of STS in total.
    enum { STS = STS_PER_COL * STS_PER_ROW }; // 1, 8, 8

    // TD [2022-06-02] In the case of Q (16 x 64) in the backward pass with 256 threads,
    // we only need to store 16 * 64 * 2 = 2KB instead of 4KB.
    static constexpr bool PARTIAL_STORE = ROWS_PER_STS > ROWS;
    static constexpr int STORING_THREADS = PARTIAL_STORE ? ROWS * THREADS_PER_ROW : Cta_tile::THREADS_PER_CTA; // assume 128

    // The size of one buffer in bytes in shared memory.
    // enum { BYTES_PER_BUFFER = STS * BYTES_PER_STS * Cta_tile::THREADS_PER_CTA };
    enum { BYTES_PER_BUFFER = STS * BYTES_PER_STS * STORING_THREADS }; // (1,4,4)*16*128 = (2048,8192,8192)
    // The number of buffers.
    enum { BUFFERS_PER_TILE = BUFFERS_PER_TILE_ }; // 1 or 2
    // The size in bytes of total buffers.
    enum { BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE }; // (2048,8192,8192)*(1,2)
    // The boundary for smem_read_offset and smem_write_offset increment.
    enum { BYTES_PER_TILE_INC_BOUNDARY = BYTES_PER_TILE - BYTES_PER_BUFFER }; // (2048,8192,8192)*(0,1)

    // Do we enable the LDS.128 fast path?
    enum { ENABLE_LDS_FAST_PATH = ENABLE_LDS_FAST_PATH_ };
    static_assert(ENABLE_LDS_FAST_PATH == 0);
    // The number of rows that are used for the XOR swizzling to allow fast STS/LDS.
    enum { ROWS_PER_XOR_PATTERN = ROWS_PER_XOR_PATTERN_ }; // 8
    // The number of cols that are used for the XOR swizzling to allow fast STS/LDS.
    enum { COLS_PER_XOR_PATTERN = COLS_PER_XOR_PATTERN_ * 16 / BYTES_PER_STS }; // 1
    // Use or not predicates
    enum { USE_PREDICATES = USE_PREDICATES_ };

    // The type of elements that are stored in shared memory by each thread.
    using Store_type = typename Uint_from_size_in_bytes<BYTES_PER_STS>::Type; // 16 bytes, 128 bits --> uint4

    // Ctor.
    inline __device__ Smem_tile_without_skews(void *smem, int tidx)
        : smem_(__nvvm_get_smem_pointer(smem)), tidx_(tidx) {

        // The row written by a thread. See doc/mma_smem_layout.xlsx.
        int smem_write_row = tidx / THREADS_PER_ROW; // [0,127]/8=[0,15]

        // The XOR pattern.
        int smem_write_xor = smem_write_row % ROWS_PER_XOR_PATTERN * COLS_PER_XOR_PATTERN; // smem_write_row%8*1=[0,7]
        // Compute the column and apply the XOR pattern.
        int smem_write_col = (tidx % THREADS_PER_ROW) ^ smem_write_xor; // [0,7] ^ [0,7]

        // The offset.
        this->smem_write_offset_ = smem_write_row*BYTES_PER_ROW + smem_write_col*BYTES_PER_STS; // where to write, each a write to shared is a STS

        // TODO: Why not merge it with the read offset?
        // this->smem_read_buffer_ = __shfl_sync(0xffffffff, 0, 0);
        // this->smem_write_buffer_ = __shfl_sync(0xffffffff, 0, 0);
    }

    // Compute the store pointers.
    template< int N >
    inline __device__ void compute_store_pointers(uint32_t (&ptrs)[N]) {
        #pragma unroll
        for( int ii = 0; ii < N; ++ii ) { // 1
            // Decompose the STS into row/col.
            int row = ii / STS_PER_ROW; // ii/1
            int col = ii % STS_PER_ROW; // ii%1

            // Assemble the offset.
            int offset = smem_write_offset_ + row*ROWS_PER_STS*BYTES_PER_ROW; //  row*16*128 = row*2048

            // Take the column into account.
            if( STS_PER_ROW > 1 ) { // 1 can't be > 1
                offset += col*THREADS_PER_ROW*BYTES_PER_STS;
            }

            // Apply the XOR pattern if needed.
            if( ROWS_PER_STS < ROWS_PER_XOR_PATTERN ) { // 16 can't be < 8
                const int m = row * ROWS_PER_STS % ROWS_PER_XOR_PATTERN;
                offset ^= m * COLS_PER_XOR_PATTERN * BYTES_PER_STS;
            }

            // Assemble the final pointer :)
            // ptrs[ii] = smem_ + offset + smem_write_buffer_;
            // smem_write_buffer_ is already merged with smem_write_offset_
            ptrs[ii] = smem_ + offset;
        }
    }

    inline __device__ void debug_reset() {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
        for( int row = 0; row < ROWS; ++row ) {
            for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                if( threadIdx.x == 0 ) {
                    uint32_t val = 0x0;
                    sts(val, smem_ + row*BYTES_PER_ROW + col + buffer);
                }
            }
        }
        }
    }

    // Print the content of the tile (only for debug ;)).
    inline __device__ void debug_print() const {
        for( int buffer = 0; buffer < BYTES_PER_TILE; buffer += BYTES_PER_BUFFER) {
            for( int row = 0; row < ROWS; ++row ) {
                for( int col = 0; col < BYTES_PER_ROW; col += 4 ) {
                    if( threadIdx.x == 0 ) {
                        uint32_t val;
                        lds(val, smem_ + row*BYTES_PER_ROW + col + buffer);
                        printf("block=(x=%2d, y=%2d, z=%2d) (smem_=%2d, buffer=%2d, row=%2d, byte=%4d)=0x%08x\n",
                            blockIdx.x,
                            blockIdx.y,
                            blockIdx.z,
                            smem_,
                            buffer,
                            row,
                            col,
                            val);
                    }
                }
            }
        }
    }

    // circular R&W buffer, for those more than 1 buffer (double/triple buffer)
    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_read_offset_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_offset_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_offset_ += BYTES_PER_BUFFER;
        }
    }


    // Move the write offset to next buffer.
    inline __device__ void move_to_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 && smem_write_offset_ >= BYTES_PER_TILE_INC_BOUNDARY ) {
            this->smem_write_offset_ -= BYTES_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_offset_ += BYTES_PER_BUFFER;
        }
    }


    // Store to the tile in shared memory.
    template< int N >
    inline __device__ void store(const Store_type (&data)[N], uint64_t = 0) { // N = 1 for fetch q and N=4 for fetch k, Store_type is uint4
        uint32_t smem_ptrs[N];
        this->compute_store_pointers(smem_ptrs);
        // Trying to reduce the shared mem for Q from 4KB per buffer to 2KB per buffer.
        if (!PARTIAL_STORE || (tidx_ / THREADS_PER_ROW < ROWS)) {
            sts(smem_ptrs, data);
        }
    }


    // The shared memory pointer.
    const uint32_t smem_;
    // The read offset. Reserve 4 offsets if needed.
    int smem_read_offset_;
    // The write offset.
    int smem_write_offset_;
    const int tidx_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true
>
struct Smem_tile_a {}; // Forward reference declarations, either {} or no {} 

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
struct Rows_per_xor_pattern_a {
    // The size in bits.
    enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_A }; // N*16=K*16=64*16=1024
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) }; // 8
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
struct Rows_per_xor_pattern_row_a : public Rows_per_xor_pattern_a<N> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS, // 16
    // The number of buffers per tile.
    int BUFFERS_PER_TILE, // 2
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_row_a<Cta_tile::K>::VALUE //8
>
struct Smem_tile_row_a : public Smem_tile_without_skews<Cta_tile,
                                                               Cta_tile::M,
                                                               Cta_tile::K,
                                                               fmha::BITS_PER_ELEMENT_A,
                                                               BYTES_PER_STS,
                                                               BUFFERS_PER_TILE,
                                                               0,
                                                               ROWS_PER_XOR_PATTERN_,
                                                               1> {
    // The MMA tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::M,
                                         Cta_tile::K,
                                         fmha::BITS_PER_ELEMENT_A,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_a<Row>; // uint32_t regs_[Base_::NUM_REGS];  <uint16_t, 8>  <float, 4>

    // when Cta_tile::K is a power of 2, there is no diff between padding and non-padding
    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding<Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = fmha::Hmma_tile<Cta_tile_with_padding>; 

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_row_a(void *smem, int tidx) : Base(smem, tidx) {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;

        static_assert(WARPS_M == 1);
        static_assert(WARPS_N == 4 || WARPS_N == 8); // 4
        static_assert(WARPS_K == 1);
        static_assert(Base::ROWS_PER_XOR_PATTERN == 2 || Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 8); // 8

        // The row and column read by the thread.
        int smem_read_row  = (tidx & 0x0f); // 0-15, %16
        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING; // 1
        int smem_read_col = ((smem_read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN; // (0-16)%8*1
        smem_read_col ^= (tidx & 0x10) / 16; // 0 or 1. ??? [0-7]^([0,16]/16)= (0, 16, 32, 48, 64, ..., 127) / 16 = 0,1,...,7

        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW_BEFORE_PACKING + smem_read_col*BYTES_PER_LDS; // row*128 + 0/1*16
    }


    // Load from shared memory.
    inline __device__ void load(Fragment (&a)[Mma_tile::MMAS_M], int ki) { // ki=0,1,2,3
        #pragma unroll
        for( int mi = 0; mi < Mma_tile::MMAS_M; ++mi ) { // 1
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = mi * Mma_tile::M_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING; //  0, 0*16*128

            // Load using LDSM.M88.4.
            uint4 tmp;
            // ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + offset);

            // Store the value into the fragment.
            a[mi].reg(0) = tmp.x;
            a[mi].reg(1) = tmp.y;
            a[mi].reg(2) = tmp.z;
            a[mi].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if(        Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2; // ^=3*16*2=96
        } else if( Mma_tile_with_padding::MMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS, // 16
    // The number of buffers per tile.
    int BUFFERS_PER_TILE // 2
>
struct Smem_tile_a<Cta_tile, Row, BYTES_PER_STS, BUFFERS_PER_TILE> // for q
    : public Smem_tile_row_a<Cta_tile,
                                    BYTES_PER_STS,
                                    BUFFERS_PER_TILE> {
    // The base class.
    using Base = Smem_tile_row_a<Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_a(void *smem, int tidx) : Base(smem, tidx) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The layout of the tile.
    typename Layout,
    // The size of the STS.
    int BYTES_PER_STS = 16,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE = 1,
    // Use or not predicates
    bool USE_PREDICATES = true
>
struct Smem_tile_b {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
struct Rows_per_xor_pattern_b {
    // The size in bits.
    enum { N_IN_BITS = N * fmha::BITS_PER_ELEMENT_B }; // Cta_tile::K*16=64*16=1024
    // The number of rows.
    enum { VALUE = N_IN_BITS <= 256 ? 2 : (N_IN_BITS <= 512 ? 4 : 8) }; // 8
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
struct Rows_per_xor_pattern_col_b : public Rows_per_xor_pattern_b<N> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE,
    // How many rows to use for the XOR pattern to avoid bank conflicts?
    int ROWS_PER_XOR_PATTERN_ = Rows_per_xor_pattern_col_b<Cta_tile::K>::VALUE
>
struct Smem_tile_col_b : public Smem_tile_without_skews<Cta_tile,
                                                           Cta_tile::N,
                                                           Cta_tile::K,
                                                           fmha::BITS_PER_ELEMENT_B,
                                                           BYTES_PER_STS,
                                                           BUFFERS_PER_TILE,
                                                           0,
                                                           ROWS_PER_XOR_PATTERN_,
                                                           1> {
    // The MMA tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;
    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile,
                                         Cta_tile::N,
                                         Cta_tile::K,
                                         fmha::BITS_PER_ELEMENT_B,
                                         BYTES_PER_STS,
                                         BUFFERS_PER_TILE,
                                         0,
                                         ROWS_PER_XOR_PATTERN_,
                                         1>;
    // The fragment.
    using Fragment = Fragment_b< Col>;

    // When we use padding to reach a power of two, special care has to be taken.
    using Cta_tile_with_padding = Cta_tile_with_k_with_padding< Cta_tile>;
    // The number of MMAs.
    using Mma_tile_with_padding = fmha::Hmma_tile<Cta_tile_with_padding>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // The number of STS per thread
    enum { STS_PER_THREAD_ = Base::ROWS * Base::THREADS_PER_ROW / Cta_tile::THREADS_PER_CTA }; // (16,128,64)* (8,16) / 128
    // The number of STS per thread must be at least 1.
    enum { STS_PER_THREAD = Max<1, STS_PER_THREAD_>::VALUE };

    // Ctor.
    inline __device__ Smem_tile_col_b(void *smem, int tidx) : Base(smem, tidx) {

        // For documentation on the layout, see doc/mma_smem_layout.xlsx.

        // The number of warps.
        const int WARPS_M = Cta_tile::WARPS_M;
        const int WARPS_N = Cta_tile::WARPS_N;
        const int WARPS_K = Cta_tile::WARPS_K;
        static_assert(Base::ROWS_PER_XOR_PATTERN == 2 || Base::ROWS_PER_XOR_PATTERN == 4 || Base::ROWS_PER_XOR_PATTERN == 8); // 8
        static_assert(WARPS_M == 1);
        static_assert(WARPS_N == 4 || WARPS_N == 8);
        static_assert(WARPS_K == 1);

        // The masks to select the warps.
        const int WARP_MASK_N = Warp_masks<WARPS_M, WARPS_N, WARPS_K>::N;  // 0110 0000, <1,4,1>::M = 0x00, N = 0x60, K = 0x00
        // The divisor for the warps.
        const int WARP_DIV_N = WARPS_M *       1 * Cta_tile::THREADS_PER_WARP; // 32

        // The row and column read by the thread. -> 64 rows in total
        int smem_read_row  = (tidx & WARP_MASK_N) / WARP_DIV_N * Mma_tile::N_PER_MMA + // (00000000-11111111)&01100000/32*16 -> warp_id*16
                             (tidx & 0x07) + // (00000000-11111111)&00000111 -> 0-7
                             (tidx & 0x10) / 2; // (00000000-11111111)&00010000/2 -> 0 or 8
        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING; // 128/128=1    128 B = 64 half = 8 uint4
        int smem_read_col = ((smem_read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN; // (0-64)%8 * 1
        smem_read_col ^= (tidx & 0x08) / 8; // (0-7) ^ ((00000000-11111111) & 00001000) / 8) = (0000-0111)^(0000 or 0001), unchange or xor 1
        // The shared memory offset.
        this->smem_read_offset_ = smem_read_row*Base::BYTES_PER_ROW_BEFORE_PACKING + smem_read_col*BYTES_PER_LDS;
    }


    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
        #pragma unroll
        for( int ni = 0; ni < Mma_tile::MMAS_N; ++ni ) { // 1, for each thread
            // Jump by as many matrix rows as needed (a row in smem may pack multiple matrix rows).
            int offset = ni * Mma_tile::N_PER_MMA_PER_CTA * Base::BYTES_PER_ROW_BEFORE_PACKING; // ni*64*128

            // Load using LDSM.M88.4.
            uint4 tmp;
            // ldsm(tmp, this->smem_ + this->smem_read_offset_ + this->smem_read_buffer_ + offset);
            ldsm(tmp, this->smem_ + this->smem_read_offset_ + offset);

            // Store the value into the fragment.
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;
        }

        // Move the offset to the next possition. See doc/mma_smem_layout.xlsx.
        static_assert(Mma_tile_with_padding::MMAS_K < 64, "Not implemented");
        if(        Mma_tile_with_padding::MMAS_K >= 32 && ki % 16 == 15 ) {
            this->smem_read_offset_ ^= 31 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >= 16 && ki %  8 ==  7 ) {
            this->smem_read_offset_ ^= 15 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  8 && ki %  4 ==  3 ) {
            this->smem_read_offset_ ^=  7 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  4 && ki %  2 ==  1 ) {
            this->smem_read_offset_ ^=  3 * BYTES_PER_LDS * 2;
        } else if( Mma_tile_with_padding::MMAS_K >=  2 ) {
            this->smem_read_offset_ ^=  1 * BYTES_PER_LDS * 2;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The size of the STS.
    int BYTES_PER_STS,
    // The number of buffers per tile.
    int BUFFERS_PER_TILE
>
struct Smem_tile_b< Cta_tile, Col, BYTES_PER_STS, BUFFERS_PER_TILE >
    : public Smem_tile_col_b<Cta_tile,
                             BYTES_PER_STS,
                             BUFFERS_PER_TILE> {

    // The base class.
    using Base = Smem_tile_col_b< Cta_tile, BYTES_PER_STS, BUFFERS_PER_TILE>;

    // Ctor.
    inline __device__ Smem_tile_b(void *smem, int tidx) : Base(smem, tidx) {
    }
};



////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile>
struct Smem_tile_v : public fmha::Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, 1, 0, Rows_per_xor_pattern_col_b<Cta_tile::N>::VALUE, 1> {

    // The base class.
    using Base = Smem_tile_without_skews<Cta_tile, Cta_tile::K, Cta_tile::N, 16, 16, 1, 0, Rows_per_xor_pattern_col_b<Cta_tile::N>::VALUE, 1>; // Cta_tile::N=64, VALUE=8
    // The MMA tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;
    // The fragment.
    using Fragment = Fragment_b< fmha::Col>;

    // The size of a single LDS in bytes.
    enum { BYTES_PER_LDS = 16 };

    // Ctor.
    inline __device__ Smem_tile_v(void *smem, int tidx) : Base(smem, tidx) {

        // The row/col read by the thread.
        int read_row, read_col;

        static_assert(Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1 && (Cta_tile::WARPS_K == 4 || Cta_tile::WARPS_K == 8));

        read_row = (tidx & 0xe0) / 2 + (tidx & 0x0f);
        constexpr int ROWS_PER_PACKING = Base::BYTES_PER_ROW / Base::BYTES_PER_ROW_BEFORE_PACKING; // 1
        read_col = ((read_row / ROWS_PER_PACKING) % Base::ROWS_PER_XOR_PATTERN) * Base::COLS_PER_XOR_PATTERN;
        read_col ^= (tidx & 0x10) / 16;

        // The shared memory offset.
        this->smem_read_offset_ = read_row * Base::BYTES_PER_ROW_BEFORE_PACKING + read_col * BYTES_PER_LDS;
    }

    // Load from shared memory.
    inline __device__ void load(Fragment (&b)[Mma_tile::MMAS_N], int ki) {
#pragma unroll
        for( int ni = 0; ni < Mma_tile::MMAS_N; ++ni ) { // 4
            // Jump by 16 * #warps row.
            int row = ki * 16 * Cta_tile::WARPS_K;

            // Load the data using LDSM.MT88.2.
            uint4 tmp;
            fmha::ldsmt(tmp, this->smem_ + this->smem_read_offset_ + row * Base::BYTES_PER_ROW_BEFORE_PACKING);
            b[ni].reg(0) = tmp.x;
            b[ni].reg(1) = tmp.y;
            b[ni].reg(2) = tmp.z;
            b[ni].reg(3) = tmp.w;

            // Move the pointer for the next ni. I expect the compiler to not recompute those.
            if( Mma_tile::MMAS_N == 1 ) {
                // noop
            } else if( Mma_tile::MMAS_N == 2 ) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * 2;
            } else if( Mma_tile::MMAS_N == 4 ) { // this
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 2 == 0 ? 2 : 6);
            } else if (Mma_tile::MMAS_N == 8) {
                this->smem_read_offset_ ^= BYTES_PER_LDS * (ni % 4 == 3 ? 14 : (ni % 2 == 1 ? 6 : 2));
            } else {
                assert(false);  // Not implemented!
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile>
struct Smem_tile_o {

    // The MMA tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;
    // The accumulators.
    using Accumulator = fmha::Fragment_accumulator;
    // The accumulators.
    using Data_type = typename Accumulator::Data_type;

    // The size of each element.
    static constexpr int BYTES_PER_ELEMENT = sizeof(Data_type);
    // The size of each STS.
    static constexpr int BYTES_PER_STS = 8; // 4 half
    // The size of each row in shared memory.
    static constexpr int BYTES_PER_ROW = Cta_tile::N * Cta_tile::WARPS_K * BYTES_PER_ELEMENT; // 64*4*4=1024

    // The size of each LDS.
    static constexpr int BYTES_PER_LDS = 16; // 16, 4 float -> 8 4 half
    static constexpr int THREADS_PER_ROW = Cta_tile::N * BYTES_PER_ELEMENT / BYTES_PER_LDS; // 16

    // The number of rows.
    static constexpr int ROWS = Cta_tile::M;
    // The number of "rows" to process per loop iteration (in the "epilogue").
    static constexpr int ROWS_PER_LOOP = ROWS <= 64 ? ROWS : (int)Mma_tile::M_PER_MMA_PER_CTA;
    // The number of outer loops.
    static constexpr int LOOPS = ROWS / ROWS_PER_LOOP;
    // Make sure it matches our expectations.
    static_assert(LOOPS == 1 || LOOPS == (int)Mma_tile::MMAS_M, "");

    // The number of rows loaded per LDS.
    static constexpr int ROWS_PER_LDS = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW;
    // Do we have to guard against partial writes/reads.
    static constexpr bool HAS_INCOMPLETE_LDS = ROWS_PER_LOOP % ROWS_PER_LDS != 0;
    // The total number of LDS per loop.
    static constexpr int LDS_PER_LOOP = fmha::DivUpConstexpr(ROWS_PER_LOOP, ROWS_PER_LDS);

    // The amount of shared memory.
    static constexpr int BYTES_PER_TILE = ROWS_PER_LOOP * BYTES_PER_ROW;

    // The write pointer.
    uint32_t smem_write_, smem_read_;
    // Is the thread active for the last LDS of the series?
    int is_active_for_last_lds_;

    // static_assert(BYTES_PER_ROW == 64 * 4 * Cta_tile::WARPS_K);
    static_assert(LOOPS == 1 || LOOPS == (int)Mma_tile::MMAS_M, "");

    // Ctor.
    inline __device__ Smem_tile_o(void *smem, int tidx) {

        // Get a 32-bit value for the shared memory address.
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        static_assert(Cta_tile::WARPS_M == 1 && Cta_tile::WARPS_N == 1 && (Cta_tile::WARPS_K == 4 || Cta_tile::WARPS_K == 8));
        static_assert(Cta_tile::N == 16 || Cta_tile::N == 32 || Cta_tile::N == 64 || Cta_tile::N == 128);

        int write_row = (tidx & 0x1c) / 4;

        const int lane = tidx % 32;
        const int warp = tidx / 32;

        constexpr int ELEMENTS_PER_STS = BYTES_PER_STS / BYTES_PER_ELEMENT;
        constexpr int STS_PER_WARP = 16 * Mma_tile::MMAS_N / ELEMENTS_PER_STS;
        int write_col = warp * STS_PER_WARP + lane % STS_PER_WARP;

        // if ((threadIdx.x == 16) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("write_row = %d, write_col = %d\n", write_row, write_col);
        // }

        // if ((blockIdx.x == 0) && (blockIdx.y == 0) && (write_row == 0) && (write_col == 0)) {
        //     printf("threadIdx.x = %d\n", threadIdx.x);
        // }

        // Assemble the write pointer.
        smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;

        // The element read by each thread.
        int read_row = tidx / THREADS_PER_ROW;
        int read_col = tidx % THREADS_PER_ROW;

        // Take the XOR pattern into account for the column.
        read_col ^= 2 * (read_row % (Cta_tile::N == 16 ? 2 : (Cta_tile::N == 32 ? 4 : 8)));
        // read_col ^= 2 * (read_row % (Cta_tile::N == 16 ? 2 : (Cta_tile::N == 32 ? 4 : (Cta_tile::N == 128 ? 16 : 8))));

        // if ((threadIdx.x == 8) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("read_row = %d, read_col = %d\n", read_row, read_col);
        // }
        // if ((blockIdx.x == 0) && (blockIdx.y == 0) && (read_row == 0) && (read_col == 0)) {
        //     printf("threadIdx.x = %d\n", threadIdx.x);
        // }
        // Assemble the read pointer.
        this->smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;

        // Is that thread active on the last LDS?
        if( HAS_INCOMPLETE_LDS ) {
            this->is_active_for_last_lds_ = read_row + (LDS_PER_LOOP - 1) * ROWS_PER_LDS < Cta_tile::M;
        }
    }

    // Load the output fragments.
    template <bool zero_init=true>
    inline __device__ void load(uint4 (&out)[LDS_PER_LOOP]) const { // irrelvant to which warp currently running
        #pragma unroll
        for( int ii = 0; ii < LDS_PER_LOOP; ++ii ) { // 2

            // Load the elements before the reduction (split-K).
            uint4 tmp[Cta_tile::WARPS_K]; // 4
            #pragma unroll
            for( int jj = 0; jj < Cta_tile::WARPS_K; ++jj ) { // 4
                int imm = ii * ROWS_PER_LDS * BYTES_PER_ROW + jj * Cta_tile::N * BYTES_PER_ELEMENT; // ii* 8192 + jj*256 (64*4)
                uint32_t smem_read = this->smem_read_ + imm;
                // TD [2022-06-05] Ugly fix for d=128 in the forward pass, maybe there's a better way.
                if ((Cta_tile::N == 128) && (ROWS_PER_LDS == 4) && (ii % 2 == 1)) {
                    smem_read ^= 8 * BYTES_PER_LDS;
                }
                // if ((Cta_tile::N == 64) && (ROWS_PER_LDS == 8) && (ii % 2 == 1)) {
                //     smem_read ^= 4 * BYTES_PER_LDS;
                // }
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
                //     printf("Cta_tile::N = %d\n", Cta_tile::N);
                //     printf("imm diff = %d\n", smem_read - this->smem_read_);
                // }
                if( !HAS_INCOMPLETE_LDS || (ii < LDS_PER_LOOP - 1 || this->is_active_for_last_lds_) ) {
                    // fmha::lds(tmp[jj], this->smem_read_ + imm);
                    fmha::lds(tmp[jj], smem_read);
                }
            }

            // Perform the reduction.
            out[ii] = zero_init ? tmp[0] : fmha::fadd4(out[ii], tmp[0]);
            // if ((threadIdx.x == 8) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
            //     printf("out reduction: out = %.6f\n", reinterpret_cast<float (&)[4]>(out[ii])[0]);
            // }
            // #pragma unroll
            for( int jj = 1; jj < Cta_tile::WARPS_K; ++jj ) {
                out[ii] = fmha::fadd4(out[ii], tmp[jj]);
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
                //     printf("out reduction tmp = %.6f, out = %.6f\n", reinterpret_cast<float (&)[4]>(tmp[jj])[0], reinterpret_cast<float (&)[4]>(out[ii])[0]);
                // }
            }
        }
    }

    // Store the accumulators.
    template <int M, int N>
    inline __device__ void store(const Accumulator (&acc)[M][N], int mi) { // M=1,N=4
        // uint32_t smem_write_og = this->smem_write_;
        static constexpr int M_PER_MMA = Mma_tile::M_PER_MMA_PER_CTA;
        #pragma unroll
        for( int ni = 0; ni < Mma_tile::MMAS_N; ++ni ) { // 4

            // The number of MMAs that are stored per loop iteration.
            static constexpr int MMAS_M_PER_LOOP = Mma_tile::MMAS_M / LOOPS; // 1/1=1

            // Store 1st column of the different MMAs.
            #pragma unroll
            for( int mj = 0; mj < MMAS_M_PER_LOOP; ++mj ) { // 1
                // Precompute the immediates to jump between rows.
                int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;
                uint2 tmp0, tmp1;
                tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(0); // acc[0] [0]-[3]
                tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(1); 

                tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(2);
                tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(3);

                // Store.
                fmha::sts(this->smem_write_ + row_0, tmp0); // 2 float
                fmha::sts(this->smem_write_ + row_1, tmp1); // 2 float
            }
            // if ((threadIdx.x == 16) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
            //     printf("smem_write diff = %d\n", this->smem_write_ - smem_write_og);
            // }

            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
            //     uint4 read_tmp;
            //     fmha::lds(read_tmp, this->smem_read_);
            //     printf("smem_o = %.6f\n", reinterpret_cast<float (&)[4]>(read_tmp)[0]);
            // }
            // Swizzle the write pointer using a XOR of 16B.
            this->smem_write_ ^= 32;

            // Store 2nd column of the different MMAs.
            #pragma unroll
            for( int mj = 0; mj < MMAS_M_PER_LOOP; ++mj ) {
                // Precompute the immediates to jump between rows.
                int row_0 = (mj * M_PER_MMA + 0) * BYTES_PER_ROW;
                int row_1 = (mj * M_PER_MMA + 8) * BYTES_PER_ROW;

                uint2 tmp0, tmp1;
                tmp0.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(4);
                tmp0.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(5);

                tmp1.x = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(6);
                tmp1.y = acc[mi * MMAS_M_PER_LOOP + mj][ni].reg(7);
                // Store.
                fmha::sts(this->smem_write_ + row_0, tmp0); // 2 float
                fmha::sts(this->smem_write_ + row_1, tmp1); // 2 float
            }

            // if ((threadIdx.x == 16) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
            //     printf("smem_write diff = %d\n", this->smem_write_ - smem_write_og);
            // }

            // Cancel the previous XOR of 1 + swizzle the write pointer using a XOR of 32B or 64B.
            static_assert(Mma_tile::MMAS_N <= 8, "Not implemented");
            if(        Mma_tile::MMAS_N >= 8 && ni % 4 == 3 ) {
                this->smem_write_ ^= 15 * 32;
            } else if( Mma_tile::MMAS_N >= 4 && ni % 2 == 1 ) {
                this->smem_write_ ^= 7 * 32;
            } else if( Mma_tile::MMAS_N >= 2 ) {
                this->smem_write_ ^= 3 * 32;
            } else {
                this->smem_write_ ^= 3 * 32;
            }
            // this->smem_write_ ^= (ni & 1) ? 7 * 32 : 3 * 32;
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
            //     uint4 read_tmp;
            //     fmha::lds(read_tmp, this->smem_read_);
            //     printf("smem_o = %.6f\n", reinterpret_cast<float (&)[4]>(read_tmp)[0]);
            // }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile>
struct Smem_tile_mma {

    using Mma_tile = fmha::Hmma_tile<Cta_tile>;
    using Fragment = fmha::Fragment_a<fmha::Col>;

    enum { COLS = Cta_tile::N };
    enum { BYTES_PER_ELT = 2 };
    enum { BYTES_PER_STS = 4 };
    enum { BYTES_PER_ROW = COLS * BYTES_PER_ELT };  // TODO
    enum { BYTES_PER_TILE = Cta_tile::M * BYTES_PER_ROW };

    enum { WARPS_M = Cta_tile::WARPS_M };
    enum { WARPS_N = Cta_tile::WARPS_N };
    enum { WARPS_K = Cta_tile::WARPS_K };

    static_assert(WARPS_K == 1);
    inline __device__ Smem_tile_mma(char *smem, int tidx) {
        uint32_t smem_ = __nvvm_get_smem_pointer(smem);

        int write_col, write_row;
        static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8) || (WARPS_M == 4 || WARPS_M == 8) || WARPS_N == 1);
        if( WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8) ) {
            write_row = (tidx & 0x1c) / 4;
            write_col = (tidx & 0xe0) / 4 + (tidx & 0x03);
            write_col ^= (write_row & 0x07) * 4;
        } else {
            write_row = (tidx & 0xe0) / 2 + (tidx & 0x1c) / 4;
            write_col = (tidx & 0x03);
            // write_col ^= (write_row & (BYTES_PER_ROW == 32 ? 0x01 : (BYTES_PER_ROW == 64 ? 0x03 : (BYTES_PER_ROW == 128 ? 0x07 : 0x0f)))) * 4;
            write_col ^= (write_row & (BYTES_PER_ROW == 32 ? 0x01 : (BYTES_PER_ROW == 64 ? 0x03 : (BYTES_PER_ROW == 128 ? 0x07 : 0x07)))) * 4;
        }

        // write_offset_ = write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;
        smem_write_ = smem_ + write_row * BYTES_PER_ROW + write_col * BYTES_PER_STS;
    }

    template<int M, int N>
    inline __device__ void store(const uint4 (&regs)[M][N]) {
        static_assert(COLS == Cta_tile::N);
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                // size_t offset = write_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                // fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
                // fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
                // offset ^= 4 * BYTES_PER_STS;
                // fmha::sts(smem_ + offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
                // fmha::sts(smem_ + offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
                // size_t offset = smem_write_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                uint32_t offset = smem_write_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                fmha::sts(offset + 0 * BYTES_PER_ROW, regs[mi][ni].x);
                fmha::sts(offset + 8 * BYTES_PER_ROW, regs[mi][ni].z);
                offset ^= 4 * BYTES_PER_STS;
                fmha::sts(offset + 0 * BYTES_PER_ROW, regs[mi][ni].y);
                fmha::sts(offset + 8 * BYTES_PER_ROW, regs[mi][ni].w);
            }
        }
    }

    template<typename Fragment, int M, int N>
    inline __device__ void store(const Fragment (&frag)[N][M]) {
        static_assert(COLS == Cta_tile::N);
        uint4 regs[M][N];
        #pragma unroll
        for( int mi = 0; mi < M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < N; ni++ ) {
                // Need to transpose ref(1) and reg(2) here since when we load it we transpose again.
                regs[mi][ni] = make_uint4(frag[ni][mi].reg(0), frag[ni][mi].reg(2),
                                          frag[ni][mi].reg(1), frag[ni][mi].reg(3));
            }
        }
        this->store(regs);
    }

    // uint32_t smem_;
    // uint32_t write_offset_;
    uint32_t smem_write_;
};

template< typename Cta_tile, typename Base = Smem_tile_mma< Cta_tile>>
struct Smem_tile_mma_transposed : public Base {
    enum { BYTES_PER_LDS = 16 };
    enum { BYTES_PER_ROW = Base::BYTES_PER_ROW };
    enum { BYTES_PER_ELT = Base::BYTES_PER_ELT };
    enum { WARPS_M = Base::WARPS_M };
    enum { WARPS_N = Base::WARPS_N };
    static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
    using Fragment = typename Base::Fragment;
    inline __device__ Smem_tile_mma_transposed(char *smem, int tidx) : Base(smem, tidx) {

        uint32_t smem_ = __nvvm_get_smem_pointer(smem);
        static_assert(WARPS_M == 1 && (WARPS_N == 4 || WARPS_N == 8));
        int read_row, read_col;
        read_row = (tidx & 0x0f);
        read_col = (tidx & 0xe0) / 16 + (tidx & 0x1c) / 16;

        // read_col ^= (read_row & (Base::BYTES_PER_ROW == 32 ? 0x01 : (Base::BYTES_PER_ROW == 64 ? 0x03 : (Base::BYTES_PER_ROW == 128 ? 0x07 : 0x0f))));
        read_col ^= (read_row & 0x07);
        // read_offset_ = read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
        smem_read_ = smem_ + read_row * BYTES_PER_ROW + read_col * BYTES_PER_LDS;
    }

    template<int M, int N>
    inline __device__ void load(Fragment (&frag)[M][N]) {
        static_assert(Base::COLS == Cta_tile::N);
        for( int mi = 0; mi < M; mi++ ) {
            for( int ni = 0; ni < N; ni++ ) {
                // size_t offset = read_offset_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                uint4 dst;
                // fmha::ldsmt(dst, this->smem_ + offset);
                // size_t offset = smem_read_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                uint32_t offset = smem_read_ + mi * WARPS_M * 16 * BYTES_PER_ROW + ni * WARPS_N * 16 * BYTES_PER_ELT;
                fmha::ldsmt(dst, offset);
                frag[mi][ni].reg(0) = dst.x;
                frag[mi][ni].reg(1) = dst.z;  // Fragment A regs col major!
                frag[mi][ni].reg(2) = dst.y;
                frag[mi][ni].reg(3) = dst.w;
            }
        }
    }

    // uint32_t read_offset_;
    uint32_t smem_read_;
};


template<
    typename Gmem_tile,
    // The number of buffers. (Used in multistage and double buffer cases.)
    int BUFFERS_PER_TILE_ = 1
>
struct Smem_tile_dp_sum {

    using Cta_tile = typename Gmem_tile::Cta_tile;
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The size of each element.
    static constexpr int BYTES_PER_ELEMENT = 4;
    static constexpr int ROWS = Gmem_tile::ROWS;
    static constexpr int THREADS_PER_ROW = Gmem_tile::THREADS_PER_ROW;
    static constexpr int MMAS_M = Mma_tile::MMAS_M;

    static constexpr int ROWS_PER_LDG = Gmem_tile::ROWS_PER_LDG;
    static constexpr int LDGS = Gmem_tile::LDGS;

    static constexpr int ROWS_PER_MMA = Mma_tile::M_PER_MMA;

    // The size of one buffer in bytes in shared memory.
    static constexpr int BYTES_PER_BUFFER = ROWS * BYTES_PER_ELEMENT;
    // The number of buffers.
    static constexpr int BUFFERS_PER_TILE = BUFFERS_PER_TILE_;
    // The size in bytes of total buffers.
    static constexpr int BYTES_PER_TILE = BYTES_PER_BUFFER * BUFFERS_PER_TILE;
    // The boundary for smem_read_offset and smem_write_offset increment.
    static constexpr int ROWS_PER_TILE_INC_BOUNDARY = ROWS * BUFFERS_PER_TILE - ROWS;

    inline __device__ Smem_tile_dp_sum(float *smem, const int tidx)
        : smem_(smem), smem_read_buffer_(smem), smem_write_buffer_(smem), tidx_(tidx) {
    }

    // Move the read offset to next buffer.
    inline __device__ void move_to_next_read_buffer() {
        if( BUFFERS_PER_TILE > 1 && (smem_read_buffer_ - smem_) >= ROWS_PER_TILE_INC_BOUNDARY ) {
            this->smem_read_buffer_ -= ROWS_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_read_buffer_ += ROWS;
        }
    }

    // Move the write offset to next buffer.
    inline __device__ void move_to_next_write_buffer() {
        if( BUFFERS_PER_TILE > 1 && (smem_write_buffer_ - smem_) >= ROWS_PER_TILE_INC_BOUNDARY ) {
            this->smem_write_buffer_ -= ROWS_PER_TILE_INC_BOUNDARY;
        } else if( BUFFERS_PER_TILE > 1 ) {
            this->smem_write_buffer_ += ROWS;
        }
    }

    inline __device__ void store(const float (&sum)[LDGS]) {
        if (tidx_ % THREADS_PER_ROW == 0) {
            int row = tidx_ / THREADS_PER_ROW;
            #pragma unroll
            for (int i = 0; i < LDGS; ++i) {
                if (row + i * ROWS_PER_LDG < ROWS) {
                    smem_write_buffer_[row + i * ROWS_PER_LDG] = sum[i];
                }
            }
        }
    }

    inline __device__ void store(const float sum, const int buffer_idx) {
        float *smem_write = smem_ + buffer_idx * ROWS;
        int row = tidx_ / THREADS_PER_ROW;
        if ((row < ROWS) && (tidx_ % THREADS_PER_ROW == 0)) {
            smem_write[row] = sum;
        }
    }

    inline __device__ void store(const float (&sum)[LDGS], const int buffer_idx) {
        float *smem_write = smem_ + buffer_idx * ROWS;
        if (tidx_ % THREADS_PER_ROW == 0) {
            int row = tidx_ / THREADS_PER_ROW;
            #pragma unroll
            for (int i = 0; i < LDGS; ++i) {
                if (row + i * ROWS_PER_LDG < ROWS) {
                    smem_write[row + i * ROWS_PER_LDG] = sum[i];
                }
            }
        }
    }

    inline __device__ void store_pair(const float (&sum)[MMAS_M * 2], const int buffer_idx) {
        float *smem_write = smem_ + buffer_idx * ROWS;
        // Extract the position in the warp.
        int warp = tidx_ / Cta_tile::THREADS_PER_WARP;
        int lane = tidx_ % Cta_tile::THREADS_PER_WARP;
        int row = lane / 4;
        #pragma unroll
        for (int mi = 0; mi < MMAS_M; ++mi) {
            smem_write[mi * ROWS_PER_MMA + row + 0] = sum[mi * 2 + 0];
            smem_write[mi * ROWS_PER_MMA + row + 8] = sum[mi * 2 + 1];
        }
    }

    template<int N>
    inline __device__ void load(float (&sum)[N], const int (&row)[N]) {
        #pragma unroll
        for( int ni = 0; ni < N; ni++ ) {
            sum[ni] = smem_read_buffer_[row[ni]];
        }
    }

    template<int N>
    inline __device__ void load(float (&sum)[N], const int (&row)[N], const int buffer_idx) {
        float *smem_read = smem_ + buffer_idx * ROWS;
        #pragma unroll
        for( int ni = 0; ni < N; ni++ ) {
            sum[ni] = smem_read[row[ni]];
        }
    }

    static inline __device__ float reduce_warp(float sum) {
        fmha::SumOp<float> sum_op;
        return fmha::Allreduce<THREADS_PER_ROW>::run(sum, sum_op);
    }

    const int tidx_;
    float * const smem_;
    float *smem_read_buffer_;
    float *smem_write_buffer_;
};

}  // namespace fmha
