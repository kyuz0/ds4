/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ROCM_WMMA_LDS_SCOPE_HPP
#define ROCM_WMMA_LDS_SCOPE_HPP

#include "common.hpp"
#include "fragment.hpp"
#include "load.hpp"
#include "mapping.hpp"
#include "wmma.hpp"
#include "../half_epilogue.hpp"

namespace output_a_k64_core
{
using namespace rocm_wmma_gemm;
// Inputs are immutable; inter-wave communication occurs only in LDS.
// Preserve release/acquire ordering for local memory and the execution barrier.
__device__ __forceinline__ void local_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}


/**
 * @brief Base configuration struct for wave (warp) settings.
 */
template<int warps_m, int warps_n>
struct wave_config_base
{
    static constexpr int total_warps = warps_m * warps_n;
};

/**
 * @brief Configuration struct for tuning AMD wave bounds.
 *
 * Provides minimum and maximum waves per execution unit to optimize register usage
 * and occupancy for the given matrix layouts.
 */
template<m_layout LAYOUT_A, m_layout LAYOUT_B, int warps_m, int warps_n>
struct wave_config : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 8;
};

/**
 * @brief Wave configuration specialization for row-major A and col-major B.
 */
template<int warps_m, int warps_n>
struct wave_config<m_layout::row_major, m_layout::col_major, warps_m, warps_n>
    : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 4;
};

/**
 * @brief Core device function implementing the WMMA-based GEMM operation.
 *
 * This function computes a block of the output matrix C using the following optimizations:
 *
 * 1. Double Buffering (Software Pipelining): Overlaps global-to-register memory loads with
 *    register-to-LDS and compute stages using two shared memory (LDS) buffers.
 * 2. Multi-Stage Register Prefetching: Stages global loads in register fragments via
 *    `prefetch_fragment` before committing to LDS, hiding instruction and global latency.
 * 3. Hierarchical Warp Tiling: Subdivides the thread block into warps (mapped via warps_m x warps_n)
 *    where each warp registers and computes a specific sub-tile of the block.
 * 4. Dual K-Slices (Layout-Specific): For row-major A and col-major B layouts, traverses K-steps
 *    in nested slices (snake pattern) to interleave LDS read latency directly behind WMMA math execution.
 * 5. Chunked LDS-Buffered Epilogue: Buffer warp register C fragments back to LDS in structured
 *    chunks for coalesced, conflict-free global writes.
 * 6. Memory Load/Commit Interleaving: Interleaves global prefetching and LDS commits across
 *    computation phases to optimize instruction-level parallelism (ILP).
 *
 * @tparam T Data type of output matrix C.
 * @tparam U Data type of input matrices A and B.
 * @tparam LAYOUT_C Memory layout of C.
 * @tparam LAYOUT_A Memory layout of A.
 * @tparam LAYOUT_B Memory layout of B.
 * @tparam warps_m Number of warps mapped to the M dimension.
 * @tparam warps_n Number of warps mapped to the N dimension.
 * @tparam warp_tile_m Number of WMMA tiles per warp in the M dimension.
 * @tparam warp_tile_n Number of WMMA tiles per warp in the N dimension.
 * @tparam k_slices Number of wmma_tile-sized K-slices packed into one block.
 * @tparam single_buffer 1 = single LDS buffer (half LDS, read-sync-write, direct C store)
 *                       for higher occupancy; 0 = double buffer (software-pipelined).
 * @tparam swizzle Swizzle size for the block mapping.
 * @tparam bits Vectorization bit width for memory loads.
 * @tparam is_aligned True if the global matrix dimensions are strictly aligned to the load width.
 */
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      k_slices,
         int      single_buffer,
         int      swizzle,
         int      bits,
         int      is_aligned>
__device__ __forceinline__ void gemm_impl(
    U* __restrict__ C, const U* __restrict__ A, const U* __restrict__ B, int M, int N, int K, unsigned stagger_mask)
{
    static_assert(warps_m != 0 && warps_n != 0);
    static_assert((warp_tile_m * warp_tile_n) > 1);
    static_assert(k_slices > 0);

    constexpr int padding_a = (LAYOUT_A == m_layout::row_major) ? 8 : 0;
    constexpr int padding_b = (LAYOUT_B == m_layout::col_major) ? 8 : 0;

    constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    constexpr int block_n = warps_n * warp_tile_n * wmma_tile;

    // Number of wmma_tile-sized K-slices packed into one block along the contraction
    // dimension. Packing multiple slices lets their LDS reads interleave behind WMMA
    // math and widens each global K-burst; block_k is derived from it.
    constexpr int block_k  = k_slices * wmma_tile;
    constexpr int stride_a = block_k + padding_a;
    constexpr int stride_b = block_k + padding_b;
    constexpr int lds_size = (block_m * stride_a) + (stride_b * block_n);

    // Single buffering halves the LDS tile (one buffer instead of the pipelined pair),
    // holding the next tile in registers (already done by prefetch_fragment) and committing
    // after a read-sync-write barrier — trading one extra __syncthreads per K-iteration and
    // a direct (non-staged) C store for higher occupancy. Whether it wins is config- and
    // size-dependent, so it is a tuned parameter rather than a compile-time heuristic.
    // single_buffer == 0 is byte-identical to the double-buffered software pipeline.
    constexpr bool use_single_buffer = (single_buffer != 0);
    constexpr int  lds_buffers       = use_single_buffer ? 1 : 2;

    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    using mapper = tile_mapper<block_m, block_n, LAYOUT_A, LAYOUT_B, swizzle>;

    int block_row, block_col;
    mapper().map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    __shared__ U lds_mem[lds_buffers * lds_size];

    U* a_tiles_0 = lds_mem;
    U* b_tiles_0 = lds_mem + (block_m * stride_a);

    constexpr int full_block = warp_size * warps_m * warps_n;
    constexpr int half_block = full_block / 2;
    const int     tid        = threadIdx.x;
    const int     cid        = tid % half_block;

    A += blockIdx.y * M * K;
    B += blockIdx.y * K * N;
    C += blockIdx.y * 1024; // [token][group][rank], no intermediate

    const U* A_base = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);
    const U* B_base = B + block_col * ((LAYOUT_B == m_layout::col_major) ? K : 1);

    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = tid % warp_size;
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    const int warp_m_base = warp_row * warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * warp_tile_n * wmma_tile;

    constexpr int a_frag_size = k_slices * warp_tile_m;
    constexpr int b_frag_size = k_slices * warp_tile_n;

    fragment<T, wmma_tile> c_frags[warp_tile_m][warp_tile_n];
    fragment<U, wmma_tile> a_frag[a_frag_size];
    fragment<U, wmma_tile> b_frag[b_frag_size];

    const int lead_a = (LAYOUT_A == m_layout::col_major) ? M : K;
    const int lead_b = (LAYOUT_B == m_layout::col_major) ? K : N;

    // Constructor builds the SRD once from the matrix base + allocation size.
    // All prefetch/partial_prefetch calls reuse it with no per-call overhead.
    prefetch_fragment<LAYOUT_A, bits, full_block, block_m, block_k, padding_a, U> regs_a(
        A,
        static_cast<unsigned>(M) * static_cast<unsigned>(K));
    prefetch_fragment<LAYOUT_B, bits, full_block, block_k, block_n, padding_b, U> regs_b(
        B,
        static_cast<unsigned>(K) * static_cast<unsigned>(N));

    // Native BLAS rank tile is physical donor column tile. The native starts
    // ((rank_tile & OrigStaggerUIter)<<2) K32 steps =128 elements per tile.
    const int k_start = ((block_col / 128) & stagger_mask) * 128;
    const U* A_tile_ptr = A_base + k_start;
    const U* B_tile_ptr = B_base + k_start;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);
    const int global_mult_B = block_k * ((LAYOUT_B == m_layout::col_major) ? 1 : N);

    constexpr int frag_mult_A   = (LAYOUT_A == m_layout::col_major) ? 1 : stride_a;
    constexpr int frag_mult_B   = (LAYOUT_B == m_layout::col_major) ? stride_b : 1;
    constexpr int frag_offset_A = wmma_tile * frag_mult_A;
    constexpr int frag_offset_B = wmma_tile * frag_mult_B;

    const int warp_offset_A = (warp_m_base + half_lane) * frag_mult_A;
    const int warp_offset_B = (warp_n_base + half_lane) * frag_mult_B;

    // LDS stride between consecutive K-slices (from prefetch_fragment::commit layout):
    //   row-major A: m*stride_a + k  => 1
    //   col-major A: k*block_m  + m  => block_m
    //   col-major B: n*stride_b + k  => 1
    //   row-major B: k*block_n  + n  => block_n
    constexpr int slice_stride_A
        = (LAYOUT_A == m_layout::row_major) ? wmma_tile : wmma_tile * block_m;
    constexpr int slice_stride_B
        = (LAYOUT_B == m_layout::col_major) ? wmma_tile : wmma_tile * block_n;

    regs_a.prefetch(A_tile_ptr, lead_a, tid);
    regs_b.prefetch(B_tile_ptr, lead_b, tid);
    regs_a.commit(a_tiles_0, tid);
    regs_b.commit(b_tiles_0, tid);
    local_barrier();

    // In double-buffer mode current/next alternate between the two LDS halves; in
    // single-buffer mode there is one buffer and next_* alias current_* (unused).
    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = use_single_buffer ? a_tiles_0 : (lds_mem + lds_size);
    U* next_b    = use_single_buffer ? b_tiles_0 : (lds_mem + lds_size + (block_m * stride_a));

    constexpr bool   warp_m_is_major = warp_tile_m >= warp_tile_n;
    constexpr size_t warp_inner_max  = warp_m_is_major ? warp_tile_n : warp_tile_m;
    // Slice-packing layouts scale the outer dimension by k_slices so every slice is
    // traversed as one contiguous snake. Single-slice keeps the original outer extent.
    constexpr size_t warp_outer_max
        = k_slices * static_cast<size_t>(warp_m_is_major ? warp_tile_m : warp_tile_n);
    constexpr size_t total_combos = warp_outer_max * warp_inner_max;
    // The prefetch of the next tile is split across the fold, alternating A and B every
    // other iteration, so each matrix is staged in total_combos / 2 chunks.
    constexpr size_t prefetch_steps = total_combos / 2;

    // get_wm/get_wn produce indices spanning all k_slices in one contiguous snake:
    // the outer index runs 0..k_slices*wt-1, the inner 0..wt-1 (or vice versa).
    constexpr auto get_wm = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_o : w_i_s;
    };

    constexpr auto get_wn = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_i_s : w_o;
    };

    auto run_fold
        = [&]<bool do_prefetch>(const U* ca0, const U* cb0, const U* next_A, const U* next_B)
    {
        // Returns the LDS pointer for fragment index WM/WN in the flat array. The high
        // part of the index selects the K-slice (WM / warp_tile_m) and the low part the
        // fragment within that slice. The index is a template parameter so the slice and
        // fragment offsets are guaranteed to constant-fold to a single immediate.
        auto a_ptr = [&]<size_t WM>() constexpr -> const U*
        { return ca0 + (WM / warp_tile_m) * slice_stride_A + (WM % warp_tile_m) * frag_offset_A; };
        auto b_ptr = [&]<size_t WN>() constexpr -> const U*
        { return cb0 + (WN / warp_tile_n) * slice_stride_B + (WN % warp_tile_n) * frag_offset_B; };

        if constexpr(warp_m_is_major)
        {
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0],
                                                     a_ptr.template operator()<0>(),
                                                     block_m,
                                                     stride_a);
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0],
                                                     b_ptr.template operator()<0>(),
                                                     stride_b,
                                                     block_n);
        }
        else
        {
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0],
                                                     b_ptr.template operator()<0>(),
                                                     stride_b,
                                                     block_n);
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0],
                                                     a_ptr.template operator()<0>(),
                                                     block_m,
                                                     stride_a);
        }

        [&]<size_t... i>(std::index_sequence<i...>)
        {
            (
                [&]()
                {
                    if constexpr(do_prefetch)
                    {
                        constexpr size_t step = i / 2;
                        if constexpr((i % 2) == 0)
                        {
                            regs_a.template partial_prefetch<step, prefetch_steps>(next_A,
                                                                                   lead_a,
                                                                                   tid);
                        }
                        else
                        {
                            regs_b.template partial_prefetch<step, prefetch_steps>(next_B,
                                                                                   lead_b,
                                                                                   tid);
                        }
                    }

                    constexpr size_t wm = get_wm(i);
                    constexpr size_t wn = get_wn(i);

                    constexpr size_t next_i = i + 1;
                    if constexpr(next_i < total_combos)
                    {
                        constexpr size_t next_wm = get_wm(next_i);
                        constexpr size_t next_wn = get_wn(next_i);
                        if constexpr(warp_m_is_major)
                        {
                            if constexpr(next_wm != wm)
                            {
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[next_wm],
                                    a_ptr.template operator()<next_wm>(),
                                    block_m,
                                    stride_a);
                            }
                            // B changes every time wm wraps to a new outer row (including slice boundary)
                            if constexpr(next_wm % warp_tile_m == 0)
                            {
                                constexpr size_t b_idx
                                    = next_wn + (next_wm / warp_tile_m) * warp_tile_n;
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[b_idx],
                                    b_ptr.template operator()<b_idx>(),
                                    stride_b,
                                    block_n);
                            }
                        }
                        else
                        {
                            if constexpr(next_wn != wn)
                            {
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[next_wn],
                                    b_ptr.template operator()<next_wn>(),
                                    stride_b,
                                    block_n);
                            }
                            if constexpr(next_wn % warp_tile_n == 0)
                            {
                                constexpr size_t a_idx
                                    = next_wm + (next_wn / warp_tile_n) * warp_tile_m;
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[a_idx],
                                    a_ptr.template operator()<a_idx>(),
                                    block_m,
                                    stride_a);
                            }
                        }
                    }

                    // Map local iteration to flat fragment arrays
                    constexpr size_t a_slice = warp_m_is_major ? 0 : (wn / warp_tile_n);
                    constexpr size_t b_slice = warp_m_is_major ? (wm / warp_tile_m) : 0;
                    constexpr size_t a_wmma  = wm + a_slice * warp_tile_m;
                    constexpr size_t b_wmma  = wn + b_slice * warp_tile_n;

                    __builtin_amdgcn_s_setprio(1);
                    wmma(a_frag[a_wmma],
                         b_frag[b_wmma],
                         c_frags[wm % warp_tile_m][wn % warp_tile_n]);
                    __builtin_amdgcn_s_setprio(0);
                }(),
                ...);
        }(std::make_index_sequence<total_combos>{});
    };

    for(int k_tile = 0; k_tile < K - block_k; k_tile += block_k)
    {
        const U* ca0    = current_a + warp_offset_A;
        const U* cb0    = current_b + warp_offset_B;
        const int next_k = (k_start + k_tile + block_k) & (4096-1);
        const U* next_A = A_base + next_k;
        const U* next_B = B_base + next_k;

        // Stages the next tile global->regs (partial_prefetch) while computing the current
        // tile from LDS (load_matrix).
        run_fold.template operator()<true>(ca0, cb0, next_A, next_B);

        A_tile_ptr = next_A;
        B_tile_ptr = next_B;

        if constexpr(use_single_buffer)
        {
            // Read-sync-write: the next commit overwrites the same buffer run_fold just
            // read, so every load_matrix read must retire before the commit stores begin.
            local_barrier();
            regs_a.commit(current_a, tid);
            regs_b.commit(current_b, tid);
            local_barrier();
        }
        else
        {
            // Double buffer: commit the next tile into the free half while the current
            // half is still being read, then swap. One barrier suffices.
            regs_a.commit(next_a, tid);
            regs_b.commit(next_b, tid);

            U* temp_a = current_a;
            U* temp_b = current_b;
            current_a = next_a;
            current_b = next_b;
            next_a    = temp_a;
            next_b    = temp_b;
            local_barrier();
        }
    }

    const U* ca0 = current_a + warp_offset_A;
    const U* cb0 = current_b + warp_offset_B;

    run_fold.template operator()<false>(ca0, cb0, nullptr, nullptr);

    local_barrier();

    static_assert(std::is_same<T,float>::value && std::is_same<U,half>::value);
    static_assert(warps_m==4 && warps_n==2 && warp_tile_m==4 && warp_tile_n==4);
    static_assert(k_slices==4 && single_buffer==1 && swizzle==8 && bits==256);
    output_a_c::store_half(C,c_frags,lds_mem,block_col,block_row,tid);
}
} // namespace output_a_k64_core
#endif
