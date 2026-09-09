#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

// -----------------------------------------------------------------------------
// Q3 G64 exact-BF16 small-T MMA.
//
// Q3 representation:
//   * 64 signed 3-bit weights / group
//   * 24 bytes / group
//   * one FP16 scale / group
//
// The signed Q3 integer {-4..3} is converted exactly to BF16.
// MMA therefore computes:
//
//     group_partial = sum(x_bf16 * q_bf16)
//
// followed by:
//
//     acc += group_partial * FP16_group_scale
//
// This preserves the Q3 model formulation. Differences versus the scalar
// kernel are limited to normal floating-point reduction/rounding order.
// -----------------------------------------------------------------------------

struct Q3SmallTMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp; // 512
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kBytesPerQ3Group   = 24;
    static constexpr auto kCodeCache        = Cache::cg;
};

__device__ __forceinline__
int q3_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q3SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};


// Decode one consecutive 6-bit payload = two signed Q3 values.
//
// pair_index 0..31 addresses the 32 pairs in one 24-byte G64 group.
__device__ __forceinline__
unsigned q3_small_t_bf16_pair(
    const std::uint8_t* __restrict__ group_codes,
    int pair_index) {

    const int bit        = pair_index * 6;
    const int word_index = bit >> 5;
    const int shift      = bit & 31;

    const auto* words =
        reinterpret_cast<const std::uint32_t*>(group_codes);

    const std::uint32_t lo =
        words[word_index];

    const std::uint32_t hi =
        word_index < 5
            ? words[word_index + 1]
            : 0u;

    const unsigned packed =
        __funnelshift_r(
            lo,
            hi,
            static_cast<unsigned>(shift))
        & 0x3fu;

    const int raw0 =
        static_cast<int>(packed & 0x7u);

    const int raw1 =
        static_cast<int>((packed >> 3) & 0x7u);

    // Signed 3-bit two's complement:
    //   0,1,2,3,4,5,6,7 -> 0,1,2,3,-4,-3,-2,-1
    const int q0 =
        (raw0 ^ 0x4) - 0x4;

    const int q1 =
        (raw1 ^ 0x4) - 0x4;

    Q3SmallTBf16PairBits result;

    result.pair =
        __floats2bfloat162_rn(
            static_cast<float>(q0),
            static_cast<float>(q1));

    return result.bits;
}


template <
    class Geometry,
    int TileCols,
    int ActiveCols,
    class Epilogue,
    class RowPolicy>
__launch_bounds__(256, 6)
__global__ void q3_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    Epilogue epilogue,
    RowPolicy row_policy) {

    using Schedule = Q3SmallTMmaSchedule;

    constexpr int kHidden =
        Geometry::kInputRows;

    constexpr int kTileK =
        Schedule::kTileKPerWarp;

    constexpr int kWarps =
        Schedule::kKWarps;

    constexpr int kRowsPerCta =
        Schedule::kRowsPerCta;

    constexpr int kGroupK =
        Schedule::kGroupK;

    constexpr int kOuterGroups =
        kHidden / kGroupK;

    constexpr int kGroupsPerRow =
        Geometry::kGroupsPerRow;

    constexpr int kTileCols =
        TileCols;

    constexpr int kNt =
        kTileCols / 8;

    constexpr int kCodeBytesPerOuter =
        kWarps * Schedule::kBytesPerQ3Group; // 8 * 24 = 192

    static_assert(
        kTileCols >= 8
        && kTileCols <= 32
        && (kTileCols % 8) == 0);

    static_assert(
        ActiveCols >= 2
        && ActiveCols <= kTileCols
        && ActiveCols > kTileCols - 8);

    static_assert(
        (kHidden % kGroupK) == 0);

    static_assert(
        RowPolicy::kOutputRowsPerCta
        <= kRowsPerCta);


    union SharedStorage {

        struct {

            // Eight consecutive Q3 G64 groups for each logical row.
            std::uint8_t
                codes[kRowsPerCta]
                     [kCodeBytesPerOuter];

            __nv_bfloat16
                activations[kWarps]
                           [kTileCols * kTileK];

            std::uint16_t
                scales[kRowsPerCta]
                      [kWarps];

        } staging;

        float partial[
            kWarps
            * kNt
            * 32
            * 4];
    };


    __shared__ __align__(16)
        SharedStorage shared;

    auto& code_shared =
        shared.staging.codes;

    auto& x_shared =
        shared.staging.activations;

    auto& scale_shared =
        shared.staging.scales;


    const int tid =
        static_cast<int>(threadIdx.x);

    const int warp =
        tid >> 5;

    const int lane =
        tid & 31;

    const int gid =
        lane >> 2;

    const int lid =
        lane & 3;

    const int k_split =
        warp;

    const int row0 =
        static_cast<int>(blockIdx.x)
        * RowPolicy::kOutputRowsPerCta;


    // -------------------------------------------------------------------------
    // Activation staging.
    //
    // Important: only ActiveCols are fetched from global memory. For T=4 this
    // means four real token rows are loaded even though MMA has N=8.
    // -------------------------------------------------------------------------
    const auto stage_x =
        [&](int outer_k0) {

            constexpr int kItemsPerSplit =
                ActiveCols * (kTileK / 8);

            for (int item = lane;
                 item < kItemsPerSplit;
                 item += 32) {

                const int col =
                    item / (kTileK / 8);

                const int k8 =
                    item
                    - col * (kTileK / 8);

                auto* dst =
                    &x_shared[warp][
                        col * kTileK
                        + q3_small_t_swizzle_64(
                            col,
                            k8 * 8)];

                cp_async<16>(
                    dst,
                    &x[
                        static_cast<std::int64_t>(col)
                            * kHidden
                        + outer_k0
                        + warp * kTileK
                        + k8 * 8]);
            }
        };


    // -------------------------------------------------------------------------
    // Q3 weight + scale staging.
    //
    // One outer K=512 region is exactly eight Q3 G64 groups:
    //
    //     8 * 24 = 192 bytes / logical row.
    // -------------------------------------------------------------------------
    const auto stage_weight =
        [&](int outer_k0) {

            const int base_group =
                outer_k0 / 64;

#pragma unroll
            for (int row_item = 0;
                 row_item < Schedule::kRowsPerLoaderWarp;
                 ++row_item) {

                const int row =
                    warp
                        * Schedule::kRowsPerLoaderWarp
                    + row_item;

                const int weight_row =
                    row_policy.weight_row(
                        row0,
                        row);

                // 192 bytes = twelve 16-byte transactions.
                for (int chunk = lane;
                     chunk < kCodeBytesPerOuter / 16;
                     chunk += 32) {

                    cp_async<16, Schedule::kCodeCache>(
                        &code_shared[row][chunk * 16],
                        codes
                            + static_cast<std::int64_t>(
                                weight_row)
                                * kGroupsPerRow
                                * Schedule::kBytesPerQ3Group
                            + static_cast<std::int64_t>(
                                base_group)
                                * Schedule::kBytesPerQ3Group
                            + chunk * 16);
                }
            }


            // Eight FP16 scales = exactly 16 bytes per row.
            for (int row = tid;
                 row < kRowsPerCta;
                 row += kWarps * 32) {

                const int weight_row =
                    row_policy.weight_row(
                        row0,
                        row);

                cp_async<16>(
                    &scale_shared[row][0],
                    scales
                        + (
                            static_cast<std::int64_t>(
                                weight_row)
                                * kGroupsPerRow
                            + base_group)
                            * 2);
            }
        };


    const int b_rin =
        lane & 7;

    const int b_koff =
        ((lane >> 3) & 1) << 3;

    float acc[kNt][4] = {};


    stage_weight(0);
    stage_x(0);

    cp_commit();
    cp_wait<0>();

    __syncthreads();


#pragma unroll
    for (int outer = 0;
         outer < kOuterGroups;
         ++outer) {

        const int outer_k0 =
            outer * kGroupK;

        float group_acc[kNt][4] = {};


        // Each warp owns one complete Q3 G64 group.
        const auto* top_group =
            &code_shared[gid][
                k_split
                    * Schedule::kBytesPerQ3Group];

        const auto* bot_group =
            &code_shared[gid + 8][
                k_split
                    * Schedule::kBytesPerQ3Group];


#pragma unroll
        for (int ks = 0;
             ks < 4;
             ++ks) {

            // Same m16n8k16 A-fragment mapping used by the exact Q4 kernel.
            const int pair0 =
                ks * 8 + lid;

            const int pair1 =
                pair0 + 4;


            const unsigned af0 =
                q3_small_t_bf16_pair(
                    top_group,
                    pair0);

            const unsigned af1 =
                q3_small_t_bf16_pair(
                    bot_group,
                    pair0);

            const unsigned af2 =
                q3_small_t_bf16_pair(
                    top_group,
                    pair1);

            const unsigned af3 =
                q3_small_t_bf16_pair(
                    bot_group,
                    pair1);


#pragma unroll
            for (int nt = 0;
                 nt < kNt;
                 ++nt) {

                unsigned bf0;
                unsigned bf1;

                const int br =
                    nt * 8
                    + b_rin;

                ldmatrix_x2(
                    bf0,
                    bf1,
                    smem_addr(
                        &x_shared[k_split][
                            br * kTileK
                            + q3_small_t_swizzle_64(
                                br,
                                ks * 16
                                    + b_koff)]));


                mma_bf16(
                    group_acc[nt][0],
                    group_acc[nt][1],
                    group_acc[nt][2],
                    group_acc[nt][3],
                    af0,
                    af1,
                    af2,
                    af3,
                    bf0,
                    bf1);
            }
        }


        const float top_scale =
            __half2float(
                __ushort_as_half(
                    scale_shared[gid][k_split]));

        const float bot_scale =
            __half2float(
                __ushort_as_half(
                    scale_shared[gid + 8][k_split]));


#pragma unroll
        for (int nt = 0;
             nt < kNt;
             ++nt) {

            acc[nt][0] =
                fmaf(
                    group_acc[nt][0],
                    top_scale,
                    acc[nt][0]);

            acc[nt][1] =
                fmaf(
                    group_acc[nt][1],
                    top_scale,
                    acc[nt][1]);

            acc[nt][2] =
                fmaf(
                    group_acc[nt][2],
                    bot_scale,
                    acc[nt][2]);

            acc[nt][3] =
                fmaf(
                    group_acc[nt][3],
                    bot_scale,
                    acc[nt][3]);
        }


        if (outer + 1 < kOuterGroups) {

            __syncthreads();

            stage_weight(
                outer_k0
                    + kGroupK);

            stage_x(
                outer_k0
                    + kGroupK);

            cp_commit();
            cp_wait<0>();

            __syncthreads();
        }
    }


    // -------------------------------------------------------------------------
    // Reduce eight independent K-split warp results.
    // Same proven reduction shape as Q4 exact-small-T.
    // -------------------------------------------------------------------------
    __syncthreads();

    auto* partial =
        shared.partial;


    if ((k_split & 1) != 0) {

#pragma unroll
        for (int nt = 0;
             nt < kNt;
             ++nt) {

            store_vec(
                partial
                    + (
                        (k_split * kNt + nt)
                            * 32
                        + lane)
                        * 4,
                make_float4(
                    acc[nt][0],
                    acc[nt][1],
                    acc[nt][2],
                    acc[nt][3]));
        }
    }


    __syncthreads();


    if ((k_split & 1) == 0) {

#pragma unroll
        for (int nt = 0;
             nt < kNt;
             ++nt) {

            const float4 partner =
                load_vec<float4>(
                    partial
                        + (
                            ((k_split + 1)
                                 * kNt
                             + nt)
                                * 32
                            + lane)
                            * 4);

            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;


            if (k_split != 0) {

                store_vec(
                    partial
                        + (
                            (k_split * kNt + nt)
                                * 32
                            + lane)
                            * 4,
                    make_float4(
                        acc[nt][0],
                        acc[nt][1],
                        acc[nt][2],
                        acc[nt][3]));
            }
        }
    }


    __syncthreads();


    if (k_split == 0) {

#pragma unroll
        for (int nt = 0;
             nt < kNt;
             ++nt) {

            float4 sum =
                make_float4(
                    acc[nt][0],
                    acc[nt][1],
                    acc[nt][2],
                    acc[nt][3]);


#pragma unroll
            for (int split = 2;
                 split < kWarps;
                 split += 2) {

                const float4 value =
                    load_vec<float4>(
                        partial
                            + (
                                (split * kNt + nt)
                                    * 32
                                + lane)
                                * 4);

                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }


            const int col0 =
                nt * 8
                + 2 * lid;

            epilogue.template store<ActiveCols>(
                row0 + gid,
                col0,
                sum);
        }
    }
}

} // namespace ninfer::ops::detail
