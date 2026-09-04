#pragma once

#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <bool Pair>
__launch_bounds__(256) __global__
    void rmsnorm_rope_d128_kernel(const std::int32_t* __restrict__ positions,
                                  const __nv_bfloat16* __restrict__ q_norm_weight,
                                  const __nv_bfloat16* __restrict__ k_norm_weight,
                                  __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k) {
    constexpr int kHeadDim       = 128;
    constexpr int kHalf          = kHeadDim / 2;
    constexpr int kWeightPairs   = kHeadDim / 2;
    constexpr int kQueryHeads    = Pair ? 32 : 0;
    constexpr int kKeyHeads      = 8;
    constexpr int kCombinedHeads = kQueryHeads + kKeyHeads;
    constexpr int kWarps         = 8;
    constexpr float kEpsilon     = 1.0e-6F;

    const int token = static_cast<int>(blockIdx.x);

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    __shared__ __nv_bfloat162 q_weight_cache[Pair ? kWeightPairs : 1];
    __shared__ __nv_bfloat162 k_weight_cache[kWeightPairs];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        dflash_rope_sincos(positions, token, pair, &sin_cache[pair], &cos_cache[pair]);
        if constexpr (Pair) {
            q_weight_cache[pair] = reinterpret_cast<const __nv_bfloat162*>(q_norm_weight)[pair];
        }
        k_weight_cache[pair] = reinterpret_cast<const __nv_bfloat162*>(k_norm_weight)[pair];
    }
    __syncthreads();

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int local_head = static_cast<int>(threadIdx.x) >> 5;
    for (int combined_head = local_head; combined_head < kCombinedHeads; combined_head += kWarps) {
        int head;
        __nv_bfloat162* data;
        const __nv_bfloat162* weight_cache;
        std::int64_t token_stride_pairs;
        if constexpr (Pair) {
            if (combined_head < kQueryHeads) {
                head               = combined_head;
                data               = reinterpret_cast<__nv_bfloat162*>(q);
                weight_cache       = q_weight_cache;
                token_stride_pairs = static_cast<std::int64_t>(kQueryHeads) * kWeightPairs;
            } else {
                head               = combined_head - kQueryHeads;
                data               = reinterpret_cast<__nv_bfloat162*>(k);
                weight_cache       = k_weight_cache;
                token_stride_pairs = static_cast<std::int64_t>(kKeyHeads) * kWeightPairs;
            }
        } else {
            head               = combined_head;
            data               = reinterpret_cast<__nv_bfloat162*>(k);
            weight_cache       = k_weight_cache;
            token_stride_pairs = static_cast<std::int64_t>(kKeyHeads) * kWeightPairs;
        }
        const std::int64_t base = static_cast<std::int64_t>(token) * token_stride_pairs +
                                  static_cast<std::int64_t>(head) * kWeightPairs;

        const int pair0             = lane;
        const int pair1             = lane + 32;
        const __nv_bfloat162 input0 = data[base + pair0];
        const __nv_bfloat162 input1 = data[base + pair1];
        const float2 input0_f32     = __bfloat1622float2(input0);
        const float2 input1_f32     = __bfloat1622float2(input1);
        float sum                   = input0_f32.x * input0_f32.x + input0_f32.y * input0_f32.y +
                    input1_f32.x * input1_f32.x + input1_f32.y * input1_f32.y;
        sum                          = warp_reduce_sum(sum);
        float inverse                = lane == 0 ? rsqrtf(sum * (1.0F / 128.0F) + kEpsilon) : 0.0F;
        inverse                      = __shfl_sync(kFullWarpMask, inverse, 0);
        const __nv_bfloat162 weight0 = weight_cache[pair0];
        const __nv_bfloat162 weight1 = weight_cache[pair1];
        const float2 weight0_f32     = __bfloat1622float2(weight0);
        const float2 weight1_f32     = __bfloat1622float2(weight1);
        const float normalized0_x    = input0_f32.x * inverse * weight0_f32.x;
        const float normalized0_y    = input0_f32.y * inverse * weight0_f32.y;
        const float normalized1_x    = input1_f32.x * inverse * weight1_f32.x;
        const float normalized1_y    = input1_f32.y * inverse * weight1_f32.y;
        const int rotary_pair        = lane * 2;
        const float cosine0          = cos_cache[rotary_pair];
        const float cosine1          = cos_cache[rotary_pair + 1];
        const float sine0            = sin_cache[rotary_pair];
        const float sine1            = sin_cache[rotary_pair + 1];
        data[base + pair0] = __floats2bfloat162_rn(normalized0_x * cosine0 - normalized1_x * sine0,
                                                   normalized0_y * cosine1 - normalized1_y * sine1);
        data[base + pair1] = __floats2bfloat162_rn(normalized1_x * cosine0 + normalized0_x * sine0,
                                                   normalized1_y * cosine1 + normalized0_y * sine1);
    }
}

} // namespace ninfer::ops
