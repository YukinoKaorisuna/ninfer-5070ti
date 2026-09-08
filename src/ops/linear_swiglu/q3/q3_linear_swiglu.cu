#include "ops/linear_swiglu/q3/q3_linear_swiglu.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/act_quant_g64.h"
#include "ops/linear_swiglu/q3/q3_linear_swiglu_int8_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

constexpr int kGateUpRows   = 34816;
constexpr int kIntermediate = 17408;
constexpr int kK            = 5120;
constexpr int kGroupK       = 64;
constexpr int kGroups       = kK / kGroupK;       // 80
constexpr int kBytesPerGroup = 24;
constexpr int kThreads      = 256;

constexpr int kQ3Int8MaxTokenTile = 4096;

bool q3_e101_enabled() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_Q3_E101");
        return v == nullptr || std::string_view(v) != "0";
    }();
    return enabled;
}


struct Q3LinearSwiGluInt8Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

constexpr int q3_int8_token_tile(int tokens) noexcept {
    return tokens < kQ3Int8MaxTokenTile
        ? tokens
        : kQ3Int8MaxTokenTile;
}

constexpr std::size_t q3_int8_workspace_bytes(int tokens) noexcept {
    const std::size_t tile =
        static_cast<std::size_t>(q3_int8_token_tile(tokens));

    // I8 [5120,tile] + FP32 [80,tile].
    //
    // 5120 is itself a 256-byte multiple, so the second default-aligned
    // WorkspaceArena allocation begins without any extra padding.
    return static_cast<std::size_t>(kK) * tile
        + static_cast<std::size_t>(kGroups) * tile * sizeof(float);
}


// Host-side profiling only. This counts dispatcher calls by token count,
// without touching the CUDA kernels.
std::atomic<std::uint64_t> q3_calls_t1{0};
std::atomic<std::uint64_t> q3_calls_t2{0};
std::atomic<std::uint64_t> q3_calls_t3{0};
std::atomic<std::uint64_t> q3_calls_t4{0};
std::atomic<std::uint64_t> q3_calls_t5_8{0};
std::atomic<std::uint64_t> q3_calls_t9_16{0};
std::atomic<std::uint64_t> q3_calls_t17_32{0};
std::atomic<std::uint64_t> q3_calls_t33_64{0};
std::atomic<std::uint64_t> q3_calls_t65_128{0};
std::atomic<std::uint64_t> q3_calls_t129_256{0};
std::atomic<std::uint64_t> q3_calls_t257_plus{0};

void print_q3_call_histogram() {
    std::fprintf(stderr, "\nQ3 SwiGLU call histogram:\n");
    std::fprintf(stderr, "  T=1       %llu\n",
        (unsigned long long)q3_calls_t1.load());
    std::fprintf(stderr, "  T=2       %llu\n",
        (unsigned long long)q3_calls_t2.load());
    std::fprintf(stderr, "  T=3       %llu\n",
        (unsigned long long)q3_calls_t3.load());
    std::fprintf(stderr, "  T=4       %llu\n",
        (unsigned long long)q3_calls_t4.load());
    std::fprintf(stderr, "  T=5..8    %llu\n",
        (unsigned long long)q3_calls_t5_8.load());
    std::fprintf(stderr, "  T=9..16   %llu\n",
        (unsigned long long)q3_calls_t9_16.load());
    std::fprintf(stderr, "  T=17..32  %llu\n",
        (unsigned long long)q3_calls_t17_32.load());
    std::fprintf(stderr, "  T=33..64  %llu\n",
        (unsigned long long)q3_calls_t33_64.load());
    std::fprintf(stderr, "  T=65..128 %llu\n",
        (unsigned long long)q3_calls_t65_128.load());
    std::fprintf(stderr, "  T=129..256 %llu\n",
        (unsigned long long)q3_calls_t129_256.load());
    std::fprintf(stderr, "  T=257+    %llu\n",
        (unsigned long long)q3_calls_t257_plus.load());
}

struct Q3ProfileAtExit {
    Q3ProfileAtExit() {
        std::atexit(print_q3_call_histogram);
    }
};

Q3ProfileAtExit q3_profile_at_exit;

// Fast T=1 GEMV schedule modelled on NInfer's native Q4 paired-row path.
constexpr int kVecBytesFast          = 16;
constexpr int kGroupsPerWarpTileFast = 16;
constexpr int kVecsPerWarpTileFast =
    kGroupsPerWarpTileFast * kBytesPerGroup / kVecBytesFast; // 24
constexpr int kWarpsPerBlockFast = 4;
constexpr int kBlockThreadsFast  = kWarpsPerBlockFast * 32;
constexpr int kPairsPerBlockFast = kWarpsPerBlockFast;

static_assert(kVecsPerWarpTileFast == 24);

__device__ __forceinline__
int q3_code(const std::uint8_t* row, int k_index) {
    const int group      = k_index >> 6;
    const int in_group   = k_index & 63;
    const int block8     = in_group >> 3;
    const int within8    = in_group & 7;

    const std::uint8_t* p =
        row + group * kBytesPerGroup + block8 * 3;

    const std::uint32_t word =
        static_cast<std::uint32_t>(p[0]) |
        (static_cast<std::uint32_t>(p[1]) << 8) |
        (static_cast<std::uint32_t>(p[2]) << 16);

    const int q = static_cast<int>((word >> (within8 * 3)) & 0x7u);

    // Signed 3-bit two's complement:
    // 0..3 => 0..3, 4..7 => -4..-1
    return (q & 4) ? q - 8 : q;
}

// Decode the two packed Q3 values assigned to one warp lane.
// Each 64-weight group is exactly 192 bits = 24 bytes.
// 32 lanes * 2 values/lane = 64 values.
__device__ __forceinline__
void q3_decode_lane_pair(
    const std::uint8_t* __restrict__ group_codes,
    int lane,
    int& q0,
    int& q1) {

    // A Q3 group is 24 bytes = six uint32 words.
    // Each lane consumes one consecutive 6-bit payload from the 192-bit group.
    const int bit        = lane * 6;
    const int word_index = bit >> 5;
    const int shift      = bit & 31;

    const auto* words =
        reinterpret_cast<const std::uint32_t*>(group_codes);

    const std::uint32_t lo = words[word_index];

    // The final word never needs bits from beyond the 24-byte group.
    const std::uint32_t hi =
        (word_index < 5) ? words[word_index + 1] : 0u;

    const unsigned pair =
        __funnelshift_r(lo, hi, static_cast<unsigned>(shift)) & 0x3fu;

    q0 = sign_extend<3>(static_cast<int>(pair & 0x7u));
    q1 = sign_extend<3>(static_cast<int>((pair >> 3) & 0x7u));
}


__device__ __forceinline__
void q3_issue_pair_tile(
    uint4 (*__restrict__ s_code)[kVecsPerWarpTileFast],
    uint4 (*__restrict__ s_scale)[2],
    const std::uint8_t* __restrict__ gate_code_row,
    const std::uint8_t* __restrict__ gate_scale_row,
    const std::uint8_t* __restrict__ up_code_row,
    const std::uint8_t* __restrict__ up_scale_row,
    int tile,
    int lane) {

    const int g0 = tile * kGroupsPerWarpTileFast;

    // 16 groups * 24 bytes = 384 bytes = 24 x uint4.
    if (lane < kVecsPerWarpTileFast) {
        pipe_copy<16>(
            &s_code[0][lane],
            reinterpret_cast<const uint4*>(
                gate_code_row + g0 * kBytesPerGroup) + lane);

        pipe_copy<16>(
            &s_code[1][lane],
            reinterpret_cast<const uint4*>(
                up_code_row + g0 * kBytesPerGroup) + lane);
    }

    // 16 FP16 scales = 32 bytes = two uint4.
    if (lane < 2) {
        pipe_copy<16>(
            &s_scale[0][lane],
            reinterpret_cast<const uint4*>(
                gate_scale_row + g0 * 2) + lane);

        pipe_copy<16>(
            &s_scale[1][lane],
            reinterpret_cast<const uint4*>(
                up_scale_row + g0 * 2) + lane);
    }

    pipe_commit();
}


__global__ void q3_linear_swiglu_gemv_pair_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out) {

    constexpr int kTiles = kGroups / kGroupsPerWarpTileFast; // 80 / 16 = 5
    static_assert(kGroups % kGroupsPerWarpTileFast == 0);
    static_assert(kIntermediate % kPairsPerBlockFast == 0);

    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    constexpr int kXVecs    = kK / 8; // uint4 contains eight BF16s

    // Like Q4: all four row-pair warps reuse one activation vector.
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];

    __shared__ uint4
        code_tile[kWarpsPerBlockFast]
                 [kStages]
                 [2]
                 [kVecsPerWarpTileFast];

    __shared__ uint4
        scale_tile[kWarpsPerBlockFast]
                  [kStages]
                  [2]
                  [2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);

    for (int i = static_cast<int>(threadIdx.x);
         i < kXVecs;
         i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }

    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    const int out_row =
        static_cast<int>(blockIdx.x) * kPairsPerBlockFast + warp;

    const std::int64_t gate_row = out_row;
    const std::int64_t up_row   = out_row + kIntermediate;

    const std::uint8_t* gate_code_row =
        codes + gate_row * kGroups * kBytesPerGroup;

    const std::uint8_t* up_code_row =
        codes + up_row * kGroups * kBytesPerGroup;

    const std::uint8_t* gate_scale_row =
        scales + gate_row * kGroups * 2;

    const std::uint8_t* up_scale_row =
        scales + up_row * kGroups * 2;

    const auto* x2 =
        reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q3_issue_pair_tile(
                code_tile[warp][p],
                scale_tile[warp][p],
                gate_code_row,
                gate_scale_row,
                up_code_row,
                up_scale_row,
                p,
                lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;

        if (fetch < kTiles) {
            const int buf = fetch % kStages;

            q3_issue_pair_tile(
                code_tile[warp][buf],
                scale_tile[warp][buf],
                gate_code_row,
                gate_scale_row,
                up_code_row,
                up_scale_row,
                fetch,
                lane);
        } else {
            pipe_commit();
        }

        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;

        const auto* gate_codes =
            reinterpret_cast<const std::uint8_t*>(
                code_tile[warp][buf][0]);

        const auto* up_codes =
            reinterpret_cast<const std::uint8_t*>(
                code_tile[warp][buf][1]);

        const auto* gate_scales =
            reinterpret_cast<const std::uint16_t*>(
                scale_tile[warp][buf][0]);

        const auto* up_scales =
            reinterpret_cast<const std::uint16_t*>(
                scale_tile[warp][buf][1]);

#pragma unroll
        for (int tile_group = 0;
             tile_group < kGroupsPerWarpTileFast;
             ++tile_group) {

            const float gate_scale =
                __half2float(
                    __ushort_as_half(gate_scales[tile_group]));

            const float up_scale =
                __half2float(
                    __ushort_as_half(up_scales[tile_group]));

            int gq0, gq1;
            int uq0, uq1;

            q3_decode_lane_pair(
                gate_codes + tile_group * kBytesPerGroup,
                lane,
                gq0,
                gq1);

            q3_decode_lane_pair(
                up_codes + tile_group * kBytesPerGroup,
                lane,
                uq0,
                uq1);

            const int k0 =
                (tile * kGroupsPerWarpTileFast + tile_group)
                    * kGroupK
                + lane * 2;

            const float2 xv =
                __bfloat1622float2(x2[k0 >> 1]);

            gate_acc = fmaf(
                static_cast<float>(gq0) * gate_scale,
                xv.x,
                gate_acc);

            gate_acc = fmaf(
                static_cast<float>(gq1) * gate_scale,
                xv.y,
                gate_acc);

            up_acc = fmaf(
                static_cast<float>(uq0) * up_scale,
                xv.x,
                up_acc);

            up_acc = fmaf(
                static_cast<float>(uq1) * up_scale,
                xv.y,
                up_acc);
        }

        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);

    if (lane == 0) {
        out[out_row] =
            __float2bfloat16_rn(silu(gate_acc) * up_acc);
    }
}


void q3_linear_swiglu_gemv_pair_launch(
    const Tensor& x,
    const Weight& w,
    Tensor& out,
    cudaStream_t stream) {

    const int grid =
        kIntermediate / kPairsPerBlockFast;

    q3_linear_swiglu_gemv_pair_kernel
        <<<grid, kBlockThreadsFast, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data));

    CUDA_CHECK(cudaGetLastError());
}



// Fast T=2..4 decode/verification path.
//
// Each warp owns one gate/up row pair. The packed Q3 weight pair is decoded
// once and reused across all active tokens, which is especially useful for
// MTP verification batches.
template <int ActiveTokens>
__global__ void q3_linear_swiglu_small_t_pair_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out) {

    static_assert(ActiveTokens >= 2 && ActiveTokens <= 32);

    constexpr int kTiles = kGroups / kGroupsPerWarpTileFast;
    static_assert(kGroups % kGroupsPerWarpTileFast == 0);
    static_assert(kIntermediate % kPairsPerBlockFast == 0);

    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;

    // Q3_SMALL_T_DIRECT_X
    //
    // Do NOT materialize all ActiveTokens x 5120 BF16 activations in
    // shared memory. At T=4 that consumed ~40 KiB/block and limited
    // occupancy to one CTA per SM. The activation footprint is tiny
    // relative to the model weights, so read it through L1/L2 instead.
    __shared__ uint4
        code_tile[kWarpsPerBlockFast]
                 [kStages]
                 [2]
                 [kVecsPerWarpTileFast];

    __shared__ uint4
        scale_tile[kWarpsPerBlockFast]
                  [kStages]
                  [2]
                  [2];

    const int lane =
        static_cast<int>(threadIdx.x) & 31;

    const int warp =
        static_cast<int>(threadIdx.x) >> 5;

    const int out_row =
        static_cast<int>(blockIdx.x)
            * kPairsPerBlockFast
        + warp;

    const std::int64_t gate_row = out_row;
    const std::int64_t up_row =
        out_row + kIntermediate;

    const std::uint8_t* gate_code_row =
        codes
        + gate_row * kGroups * kBytesPerGroup;

    const std::uint8_t* up_code_row =
        codes
        + up_row * kGroups * kBytesPerGroup;

    const std::uint8_t* gate_scale_row =
        scales
        + gate_row * kGroups * 2;

    const std::uint8_t* up_scale_row =
        scales
        + up_row * kGroups * 2;

    float gate_acc[ActiveTokens] = {};
    float up_acc[ActiveTokens]   = {};

#pragma unroll
    for (int pfetch = 0;
         pfetch < kPrefetch;
         ++pfetch) {

        if (pfetch < kTiles) {
            q3_issue_pair_tile(
                code_tile[warp][pfetch],
                scale_tile[warp][pfetch],
                gate_code_row,
                gate_scale_row,
                up_code_row,
                up_scale_row,
                pfetch,
                lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0;
         tile < kTiles;
         ++tile) {

        const int fetch =
            tile + kPrefetch;

        if (fetch < kTiles) {
            const int buf =
                fetch % kStages;

            q3_issue_pair_tile(
                code_tile[warp][buf],
                scale_tile[warp][buf],
                gate_code_row,
                gate_scale_row,
                up_code_row,
                up_scale_row,
                fetch,
                lane);
        } else {
            pipe_commit();
        }

        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf =
            tile % kStages;

        const auto* gate_codes =
            reinterpret_cast<const std::uint8_t*>(
                code_tile[warp][buf][0]);

        const auto* up_codes =
            reinterpret_cast<const std::uint8_t*>(
                code_tile[warp][buf][1]);

        const auto* gate_scales =
            reinterpret_cast<const std::uint16_t*>(
                scale_tile[warp][buf][0]);

        const auto* up_scales =
            reinterpret_cast<const std::uint16_t*>(
                scale_tile[warp][buf][1]);

#pragma unroll
        for (int tile_group = 0;
             tile_group < kGroupsPerWarpTileFast;
             ++tile_group) {

            const float gate_scale =
                __half2float(
                    __ushort_as_half(
                        gate_scales[tile_group]));

            const float up_scale =
                __half2float(
                    __ushort_as_half(
                        up_scales[tile_group]));

            int gq0, gq1;
            int uq0, uq1;

            q3_decode_lane_pair(
                gate_codes
                    + tile_group * kBytesPerGroup,
                lane,
                gq0,
                gq1);

            q3_decode_lane_pair(
                up_codes
                    + tile_group * kBytesPerGroup,
                lane,
                uq0,
                uq1);

            const int k0 =
                (tile * kGroupsPerWarpTileFast
                 + tile_group)
                    * kGroupK
                + lane * 2;

            // Q3_SMALL_T_XLOAD_BATCH
            //
            // Issue all independent activation loads before consuming any
            // of them. This increases the LDG -> use distance and gives the
            // scheduler useful independent memory operations while earlier
            // loads are waiting on L1TEX.
            float2 xv[ActiveTokens];

#pragma unroll
            for (int token = 0;
                 token < ActiveTokens;
                 ++token) {

                const auto* x2 =
                    reinterpret_cast<
                        const __nv_bfloat162*>(
                        x + static_cast<std::int64_t>(token) * kK);

                xv[token] =
                    __bfloat1622float2(
                        x2[k0 >> 1]);
            }

#pragma unroll
            for (int token = 0;
                 token < ActiveTokens;
                 ++token) {

                gate_acc[token] =
                    fmaf(
                        static_cast<float>(gq0)
                            * gate_scale,
                        xv[token].x,
                        gate_acc[token]);

                gate_acc[token] =
                    fmaf(
                        static_cast<float>(gq1)
                            * gate_scale,
                        xv[token].y,
                        gate_acc[token]);

                up_acc[token] =
                    fmaf(
                        static_cast<float>(uq0)
                            * up_scale,
                        xv[token].x,
                        up_acc[token]);

                up_acc[token] =
                    fmaf(
                        static_cast<float>(uq1)
                            * up_scale,
                        xv[token].y,
                        up_acc[token]);
            }
        }

        __syncwarp();
    }

#pragma unroll
    for (int token = 0;
         token < ActiveTokens;
         ++token) {

        gate_acc[token] =
            warp_reduce_sum(gate_acc[token]);

        up_acc[token] =
            warp_reduce_sum(up_acc[token]);
    }

    if (lane == 0) {
#pragma unroll
        for (int token = 0;
             token < ActiveTokens;
             ++token) {

            out[
                static_cast<std::int64_t>(token)
                    * kIntermediate
                + out_row] =
                __float2bfloat16_rn(
                    silu(gate_acc[token])
                    * up_acc[token]);
        }
    }
}


template <int ActiveTokens>
void q3_linear_swiglu_small_t_pair_launch(
    const Tensor& x,
    const Weight& w,
    Tensor& out,
    cudaStream_t stream) {

    const int grid =
        kIntermediate / kPairsPerBlockFast;

    q3_linear_swiglu_small_t_pair_kernel<ActiveTokens>
        <<<grid, kBlockThreadsFast, 0, stream>>>(
            static_cast<
                const __nv_bfloat16*>(x.data),
            static_cast<
                const std::uint8_t*>(w.qdata),
            static_cast<
                const std::uint8_t*>(w.scales),
            static_cast<
                __nv_bfloat16*>(out.data));

    CUDA_CHECK(cudaGetLastError());
}



// -----------------------------------------------------------------------------
// Q3 T32 Tensor-Core SwiGLU prototype.
//
// Adapted from NInfer's exact-small-T Q4 MMA decomposition:
//   * 8 K-split warps
//   * 64 K values per warp
//   * 512 K values per outer group
//   * 16 logical weight rows per CTA = 8 gate + 8 matching up rows
//
// Q3 differs only in packed weight representation:
// one G64 group = 24 bytes = 32 consecutive 6-bit signed pairs.
//
// Quantized integer values are fed to BF16 MMA without scale. Each warp computes
// one G64 partial and applies that group's FP16 scale to the resulting FP32
// accumulator before adding it to the running output accumulator.
// -----------------------------------------------------------------------------

constexpr int kQ3MmaWarps       = 8;
constexpr int kQ3MmaThreads     = kQ3MmaWarps * 32;
constexpr int kQ3MmaTileK       = 64;
constexpr int kQ3MmaGroupK      = kQ3MmaWarps * kQ3MmaTileK; // 512
constexpr int kQ3MmaRowsPerCta  = 16;
constexpr int kQ3MmaOutputRows  = 8;
constexpr int kQ3MmaTileCols    = 32;
constexpr int kQ3MmaNt          = kQ3MmaTileCols / 8;
constexpr int kQ3MmaCodeBytesPerSplit =
    kQ3MmaWarps * kBytesPerGroup; // 8 * 24 = 192

__device__ __forceinline__
int q3_mma_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q3MmaBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

// Decode one of the 32 consecutive 6-bit pairs in a 24-byte Q3 G64 group.
// No scale is applied here; scale is applied to the FP32 MMA result for the
// complete G64 partial.
__device__ __forceinline__
unsigned q3_mma_bf16_pair(
    const std::uint8_t* __restrict__ group_codes,
    int pair_index) {

    const int bit =
        pair_index * 6;

    const int word_index =
        bit >> 5;

    const int shift =
        bit & 31;

    const auto* words =
        reinterpret_cast<const std::uint32_t*>(
            group_codes);

    const std::uint32_t lo =
        words[word_index];

    const std::uint32_t hi =
        (word_index < 5)
            ? words[word_index + 1]
            : 0u;

    const unsigned packed =
        __funnelshift_r(
            lo,
            hi,
            static_cast<unsigned>(shift))
        & 0x3fu;

    const int q0 =
        sign_extend<3>(
            static_cast<int>(
                packed & 0x7u));

    const int q1 =
        sign_extend<3>(
            static_cast<int>(
                (packed >> 3) & 0x7u));

    Q3MmaBf16PairBits result;

    result.pair =
        __floats2bfloat162_rn(
            static_cast<float>(q0),
            static_cast<float>(q1));

    return result.bits;
}

__launch_bounds__(kQ3MmaThreads, 6)
__global__ void q3_linear_swiglu_t32_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out) {

    constexpr int kOuterGroups =
        kK / kQ3MmaGroupK; // 5120 / 512 = 10

    constexpr int kGroupsPerRow =
        kK / kGroupK; // 80

    union SharedStorage {
        struct {
            // 8 contiguous Q3 G64 groups for every logical weight row.
            std::uint8_t
                codes[kQ3MmaRowsPerCta]
                     [kQ3MmaCodeBytesPerSplit];

            __nv_bfloat16
                activations[kQ3MmaWarps]
                           [kQ3MmaTileCols * kQ3MmaTileK];

            std::uint16_t
                scales[kQ3MmaRowsPerCta]
                      [kQ3MmaWarps];
        } staging;

        // Same reduction workspace layout used by q4_small_t_mma_kernel.
        float partial[
            kQ3MmaWarps
            * kQ3MmaNt
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
        * kQ3MmaOutputRows;

    // Logical local rows:
    //   0..7   = gate rows
    //   8..15  = matching up rows.
    const auto weight_row =
        [&](int local_row) {

            return row0
                + (local_row & 7)
                + (local_row >= 8
                    ? kIntermediate
                    : 0);
        };

    const auto stage_x =
        [&](int outer_k0) {

            constexpr int kItemsPerSplit =
                kQ3MmaTileCols
                * (kQ3MmaTileK / 8);

            for (int item = lane;
                 item < kItemsPerSplit;
                 item += 32) {

                const int col =
                    item / (kQ3MmaTileK / 8);

                const int k8 =
                    item
                    - col * (kQ3MmaTileK / 8);

                auto* dst =
                    &x_shared[warp][
                        col * kQ3MmaTileK
                        + q3_mma_swizzle_64(
                            col,
                            k8 * 8)];

                cp_async<16>(
                    dst,
                    &x[
                        static_cast<std::int64_t>(col)
                            * kK
                        + outer_k0
                        + warp * kQ3MmaTileK
                        + k8 * 8]);
            }
        };

    const auto stage_weight =
        [&](int outer_k0) {

            const int base_group =
                outer_k0 / kGroupK;

#pragma unroll
            for (int row_item = 0;
                 row_item < 2;
                 ++row_item) {

                const int local_row =
                    warp * 2
                    + row_item;

                const int grow =
                    weight_row(local_row);

                // Eight adjacent G64 groups are exactly 192 bytes.
                // Load them as twelve 16-byte cp.async transactions.
                for (int chunk = lane;
                     chunk < kQ3MmaCodeBytesPerSplit / 16;
                     chunk += 32) {

                    cp_async<16, Cache::cg>(
                        &code_shared[local_row][chunk * 16],
                        codes
                            + static_cast<std::int64_t>(grow)
                                * kGroups
                                * kBytesPerGroup
                            + static_cast<std::int64_t>(base_group)
                                * kBytesPerGroup
                            + chunk * 16);
                }
            }

            // Eight FP16 scales = exactly 16 bytes.
            for (int local_row = tid;
                 local_row < kQ3MmaRowsPerCta;
                 local_row += kQ3MmaThreads) {

                const int grow =
                    weight_row(local_row);

                cp_async<16>(
                    &scale_shared[local_row][0],
                    scales
                        + (
                            static_cast<std::int64_t>(grow)
                                * kGroupsPerRow
                            + base_group)
                            * 2);
            }
        };

    const int b_rin =
        lane & 7;

    const int b_koff =
        ((lane >> 3) & 1) << 3;

    float acc[kQ3MmaNt][4] = {};

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
            outer * kQ3MmaGroupK;

        float group_acc[kQ3MmaNt][4] = {};

        // Every warp owns one 64-value Q3 quant group.
        const auto* gate_group0 =
            &code_shared[gid][
                k_split * kBytesPerGroup];

        const auto* gate_group1 =
            &code_shared[gid + 8][
                k_split * kBytesPerGroup];

#pragma unroll
        for (int ks = 0;
             ks < 4;
             ++ks) {

            // Same MMA A-fragment mapping as Q4 exact-small-T.
            const int pair0 =
                ks * 8 + lid;

            const int pair1 =
                pair0 + 4;

            const unsigned af0 =
                q3_mma_bf16_pair(
                    gate_group0,
                    pair0);

            const unsigned af1 =
                q3_mma_bf16_pair(
                    gate_group1,
                    pair0);

            const unsigned af2 =
                q3_mma_bf16_pair(
                    gate_group0,
                    pair1);

            const unsigned af3 =
                q3_mma_bf16_pair(
                    gate_group1,
                    pair1);

#pragma unroll
            for (int nt = 0;
                 nt < kQ3MmaNt;
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
                            br * kQ3MmaTileK
                            + q3_mma_swizzle_64(
                                br,
                                ks * 16 + b_koff)]));

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
             nt < kQ3MmaNt;
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
                + kQ3MmaGroupK);

            stage_x(
                outer_k0
                + kQ3MmaGroupK);

            cp_commit();
            cp_wait<0>();

            __syncthreads();
        }
    }

    // Reduce the 8 independent K-split warp accumulators.
    __syncthreads();

    auto* partial =
        shared.partial;

    if ((k_split & 1) != 0) {

#pragma unroll
        for (int nt = 0;
             nt < kQ3MmaNt;
             ++nt) {

            store_vec(
                partial
                    + (
                        (k_split * kQ3MmaNt + nt)
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
             nt < kQ3MmaNt;
             ++nt) {

            const float4 partner =
                load_vec<float4>(
                    partial
                        + (
                            ((k_split + 1)
                                 * kQ3MmaNt
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
                            (k_split * kQ3MmaNt + nt)
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
             nt < kQ3MmaNt;
             ++nt) {

            float4 sum =
                make_float4(
                    acc[nt][0],
                    acc[nt][1],
                    acc[nt][2],
                    acc[nt][3]);

#pragma unroll
            for (int split = 2;
                 split < kQ3MmaWarps;
                 split += 2) {

                const float4 value =
                    load_vec<float4>(
                        partial
                            + (
                                (split
                                     * kQ3MmaNt
                                 + nt)
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

            // gid 0..7 = gate output row.
            // sum.xy = gate for two token columns.
            // sum.zw = matching up row.
            const int out_row =
                row0 + gid;

            out[
                static_cast<std::int64_t>(col0)
                    * kIntermediate
                + out_row] =
                __float2bfloat16_rn(
                    silu(sum.x)
                    * sum.z);

            out[
                static_cast<std::int64_t>(col0 + 1)
                    * kIntermediate
                + out_row] =
                __float2bfloat16_rn(
                    silu(sum.y)
                    * sum.w);
        }
    }
}

void q3_linear_swiglu_t32_mma_launch(
    const Tensor& x,
    const Weight& w,
    Tensor& out,
    cudaStream_t stream) {

    if (x.ne[1] != 32) {
        throw std::invalid_argument(
            "Q3 T32 MMA prototype requires exactly 32 tokens");
    }

    constexpr int kBlocks =
        kIntermediate
        / kQ3MmaOutputRows;

    q3_linear_swiglu_t32_mma_kernel
        <<<kBlocks,
           kQ3MmaThreads,
           0,
           stream>>>(
            static_cast<
                const __nv_bfloat16*>(x.data),
            static_cast<
                const std::uint8_t*>(w.qdata),
            static_cast<
                const std::uint8_t*>(w.scales),
            static_cast<
                __nv_bfloat16*>(out.data));

    CUDA_CHECK(cudaGetLastError());
}


__global__ void q3_linear_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int tokens) {

    const int row = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);

    if (row >= kIntermediate || col >= tokens) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);

    const std::int64_t gate_row = row;
    const std::int64_t up_row   = row + kIntermediate;

    const std::uint8_t* gate_codes =
        codes + gate_row * kGroups * kBytesPerGroup;
    const std::uint8_t* up_codes =
        codes + up_row * kGroups * kBytesPerGroup;

    const std::uint16_t* gate_scales =
        scales + gate_row * kGroups;
    const std::uint16_t* up_scales =
        scales + up_row * kGroups;

    const __nv_bfloat16* x_col =
        x + static_cast<std::int64_t>(col) * kK;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (int kk = tid; kk < kK; kk += kThreads) {
        const int group = kk >> 6;

        const float xv = __bfloat162float(x_col[kk]);

        const float gate_scale =
            __half2float(__ushort_as_half(gate_scales[group]));
        const float up_scale =
            __half2float(__ushort_as_half(up_scales[group]));

        const int gq = q3_code(gate_codes, kk);
        const int uq = q3_code(up_codes, kk);

        gate_acc = fmaf(static_cast<float>(gq) * gate_scale, xv, gate_acc);
        up_acc   = fmaf(static_cast<float>(uq) * up_scale, xv, up_acc);
    }

    __shared__ float gate_shared[kThreads];
    __shared__ float up_shared[kThreads];

    gate_shared[tid] = gate_acc;
    up_shared[tid]   = up_acc;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            gate_shared[tid] += gate_shared[tid + stride];
            up_shared[tid]   += up_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float value = silu(gate_shared[0]) * up_shared[0];
        out[static_cast<std::int64_t>(col) * kIntermediate + row] =
            __float2bfloat16_rn(value);
    }
}


// -----------------------------------------------------------------------------
// E101: Q3 weight / group-64 A8 large-prefill fused SwiGLU.
//
// Decode and small-T kernels are intentionally untouched.
// Initial schedule is inherited from the measured RTX 4090 INT8 route;
// Blackwell-specific retuning comes only after correctness/performance gating.
// -----------------------------------------------------------------------------
using Q3Int8FoldedCfg =
    Q3Int8SwiGluSchedule<64, 256, 16, 128, 3, 1>;

template <class Cfg, bool Full>
void q3_launch_folded_int8(
    const std::int8_t* xq,
    const float* xs,
    const Weight& weight,
    Tensor& out,
    std::int32_t tokens,
    cudaStream_t stream) {

    static const bool configured = [] {
        CUDA_CHECK(
            cudaFuncSetAttribute(
                q3_linear_swiglu_int8_gemm_kernel<Cfg, Full>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                Cfg::kSharedBytes));
        return true;
    }();

    (void)configured;

    const dim3 grid(
        static_cast<unsigned>(
            div_up(out.ne[0], Cfg::kPairRows)),
        static_cast<unsigned>(
            div_up(tokens, Cfg::kBlockCols)));

    q3_linear_swiglu_int8_gemm_kernel<Cfg, Full>
        <<<grid,
           Cfg::kThreads,
           Cfg::kSharedBytes,
           stream>>>(
            xq,
            xs,
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data),
            out.ne[0],
            tokens,
            kK);

    CUDA_CHECK(cudaGetLastError());
}

void q3_linear_swiglu_int8_launch(
    const Tensor& x,
    const Weight& weight,
    Tensor& out,
    const Q3LinearSwiGluInt8Workspace& scratch,
    cudaStream_t stream) {

    using Cfg = Q3Int8FoldedCfg;

    const std::int32_t tile =
        q3_int8_token_tile(x.ne[1]);

    for (std::int32_t offset = 0;
         offset < x.ne[1];
         offset += tile) {

        const std::int32_t count =
            std::min(tile, x.ne[1] - offset);

        const Tensor x_slice =
            x.slice(1, offset, count);

        Tensor out_slice =
            out.slice(1, offset, count);

        act_quant_g64_launch(
            static_cast<const __nv_bfloat16*>(x_slice.data),
            scratch.codes,
            scratch.scales,
            kK,
            count,
            kK,
            stream);

        const bool full =
            (count % Cfg::kBlockCols) == 0
            && (out.ne[0] % Cfg::kPairRows) == 0;

        if (full) {
            q3_launch_folded_int8<Cfg, true>(
                scratch.codes,
                scratch.scales,
                weight,
                out_slice,
                count,
                stream);
        } else {
            q3_launch_folded_int8<Cfg, false>(
                scratch.codes,
                scratch.scales,
                weight,
                out_slice,
                count,
                stream);
        }
    }
}

} // namespace

std::size_t q3_linear_swiglu_workspace_capacity_bytes(
    std::int32_t gate_up_rows,
    std::int32_t input_rows,
    LinearPolicy policy,
    std::int32_t min_tokens,
    std::int32_t max_tokens) {

    if (gate_up_rows != kGateUpRows ||
        input_rows != kK ||
        policy != LinearPolicy::A16Only ||
        min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument(
            "q3 linear_swiglu: unsupported profile");
    }

    // Existing decode/small-T routes remain fully fused.
    if (max_tokens < 257 || !q3_e101_enabled()) {
        return 0;
    }

    // E101 A8 staging is bounded to at most 4096 tokens.
    return q3_int8_workspace_bytes(max_tokens);
}

void q3_linear_swiglu_dispatch(
    const Tensor& x,
    const Weight& w,
    Tensor& out,
    LinearPolicy policy,
    WorkspaceArena& workspace,
    cudaStream_t stream) {

    if (policy != LinearPolicy::A16Only ||
        w.qtype != QType::Q3G64_F16S ||
        w.n != kGateUpRows ||
        w.k != kK ||
        w.padded_shape[1] != kK ||
        w.group_size != kGroupK ||
        w.group != kGroupK ||
        w.qdata == nullptr ||
        w.scales == nullptr ||
        w.qhigh != nullptr ||
        w.high_plane_bytes != 0 ||
        x.ne[0] != kK ||
        out.ne[0] != kIntermediate ||
        x.ne[1] != out.ne[1]) {
        throw std::invalid_argument(
            "q3 linear_swiglu: invalid Q3 [34816,5120] problem");
    }

    const int tokens = x.ne[1];

    if (tokens == 1) {
        ++q3_calls_t1;
    } else if (tokens == 2) {
        ++q3_calls_t2;
    } else if (tokens == 3) {
        ++q3_calls_t3;
    } else if (tokens == 4) {
        ++q3_calls_t4;
    } else if (tokens <= 8) {
        ++q3_calls_t5_8;
    } else if (tokens <= 16) {
        ++q3_calls_t9_16;
    } else if (tokens <= 32) {
        ++q3_calls_t17_32;
    } else if (tokens <= 64) {
        ++q3_calls_t33_64;
    } else if (tokens <= 128) {
        ++q3_calls_t65_128;
    } else if (tokens <= 256) {
        ++q3_calls_t129_256;
    } else {
        ++q3_calls_t257_plus;
    }

    // Fast decode path. Keep original known-correct implementation
    // unchanged for T >= 2.
    if (tokens == 1) {
        q3_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    }

    if (tokens == 2) {
        q3_linear_swiglu_small_t_pair_launch<2>(
            x, w, out, stream);
        return;
    }

    if (tokens == 3) {
        q3_linear_swiglu_small_t_pair_launch<3>(
            x, w, out, stream);
        return;
    }

    if (tokens == 4) {
        q3_linear_swiglu_small_t_pair_launch<4>(
            x, w, out, stream);
        return;
    }

    // E101: only large prefill changes numerical path.
    // T=1 through T=256 remain on the existing A16 kernels.
    if (tokens >= 257 && q3_e101_enabled()) {
        auto scratch_scope = workspace.scope();

        const int tile =
            q3_int8_token_tile(tokens);

        Tensor code_storage =
            workspace.alloc(
                DType::I8,
                {kK, tile});

        Tensor scale_storage =
            workspace.alloc(
                DType::FP32,
                {kGroups, tile});

        const Q3LinearSwiGluInt8Workspace scratch{
            static_cast<std::int8_t*>(code_storage.data),
            static_cast<float*>(scale_storage.data),
        };

        q3_linear_swiglu_int8_launch(
            x,
            w,
            out,
            scratch,
            stream);

        return;
    }

    // Large-T prefill path.
    //
    // Reuse the proven T=4 pair kernel across token tiles instead of the
    // generic [row, token] kernel.  The T=4 kernel decodes each Q3 weight
    // tile once and reuses it across four activation rows, avoiding the
    // catastrophic per-token weight rereads of q3_linear_swiglu_kernel.
    const auto* x_base =
        static_cast<const __nv_bfloat16*>(x.data);

    auto* out_base =
        static_cast<__nv_bfloat16*>(out.data);

    int token = 0;

    // Main prefill tile: decode each Q3 weight tile once and reuse it
    // across eight activation rows.
    for (; token + 32 <= tokens; token += 32) {
        Tensor x_tile = x;
        Tensor out_tile = out;

        x_tile.data = const_cast<__nv_bfloat16*>(
            x_base
            + static_cast<std::int64_t>(token) * kK);

        out_tile.data =
            out_base
            + static_cast<std::int64_t>(token) * kIntermediate;

        x_tile.ne[1] = 32;
        out_tile.ne[1] = 32;

        q3_linear_swiglu_t32_mma_launch(
            x_tile, w, out_tile, stream);
    }

    if (token + 16 <= tokens) {
        Tensor x_tile = x;
        Tensor out_tile = out;

        x_tile.data = const_cast<__nv_bfloat16*>(
            x_base
            + static_cast<std::int64_t>(token) * kK);

        out_tile.data =
            out_base
            + static_cast<std::int64_t>(token) * kIntermediate;

        x_tile.ne[1] = 16;
        out_tile.ne[1] = 16;

        q3_linear_swiglu_small_t_pair_launch<16>(
            x_tile, w, out_tile, stream);

        token += 16;
    }

    if (token + 8 <= tokens) {
        Tensor x_tile = x;
        Tensor out_tile = out;

        x_tile.data = const_cast<__nv_bfloat16*>(
            x_base
            + static_cast<std::int64_t>(token) * kK);

        out_tile.data =
            out_base
            + static_cast<std::int64_t>(token) * kIntermediate;

        x_tile.ne[1] = 8;
        out_tile.ne[1] = 8;

        q3_linear_swiglu_small_t_pair_launch<8>(
            x_tile, w, out_tile, stream);

        token += 8;
    }

    // Handle a possible four-token tail with the existing proven T4 path.
    if (token + 4 <= tokens) {
        Tensor x_tile = x;
        Tensor out_tile = out;

        x_tile.data = const_cast<__nv_bfloat16*>(
            x_base
            + static_cast<std::int64_t>(token) * kK);

        out_tile.data =
            out_base
            + static_cast<std::int64_t>(token) * kIntermediate;

        x_tile.ne[1] = 4;
        out_tile.ne[1] = 4;

        q3_linear_swiglu_small_t_pair_launch<4>(
            x_tile, w, out_tile, stream);

        token += 4;
    }

    const int remainder = tokens - token;

    if (remainder != 0) {
        Tensor x_tail = x;
        Tensor out_tail = out;

        x_tail.data = const_cast<__nv_bfloat16*>(
            x_base
            + static_cast<std::int64_t>(token) * kK);

        out_tail.data =
            out_base
            + static_cast<std::int64_t>(token) * kIntermediate;

        x_tail.ne[1] = remainder;
        out_tail.ne[1] = remainder;

        if (remainder == 1) {
            q3_linear_swiglu_gemv_pair_launch(
                x_tail, w, out_tail, stream);
        } else if (remainder == 2) {
            q3_linear_swiglu_small_t_pair_launch<2>(
                x_tail, w, out_tail, stream);
        } else {
            q3_linear_swiglu_small_t_pair_launch<3>(
                x_tail, w, out_tail, stream);
        }
    }
}

} // namespace ninfer::ops::detail
