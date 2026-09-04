#pragma once

#include "ops/common/dflash_rope.cuh"
#include "ops/rmsnorm_rope/d128.cuh"

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

        const int pair0                          = lane;
        const int pair1                          = lane + 32;
        const __nv_bfloat162 input0              = data[base + pair0];
        const __nv_bfloat162 input1              = data[base + pair1];
        const __nv_bfloat162 weight0             = weight_cache[pair0];
        const __nv_bfloat162 weight1             = weight_cache[pair1];
        const detail::RmsnormRopeD128Pair output = detail::rmsnorm_rope_d128_head(
            input0, input1, weight0, weight1, cos_cache, sin_cache, lane);
        data[base + pair0] = output.first;
        data[base + pair1] = output.second;
    }
}

} // namespace ninfer::ops
