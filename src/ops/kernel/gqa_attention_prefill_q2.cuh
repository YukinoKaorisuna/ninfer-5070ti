#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

// Experimental INT2-G64 MTP KV append.
//
// One warp owns one {token, kv_head, G64 group}.
// Lanes 0..15 each load and pack four consecutive BF16 values.
// The remaining lanes participate in the warp-wide absmax reduction.
//
// Reconstruction levels:
//   {-1, -1/3, +1/3, +1} * FP16-rounded absmax.
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
void gqa_attention_prefill_fill_q2_kernel(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions,
    Metadata metadata,
    std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k,
    __half* __restrict__ scale_v,
    std::int32_t width) {

    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;

    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads * kGqaKvQuantGroups;

    if (unit >= units) return;

    const int group   = unit % kGqaKvQuantGroups;
    const int tmp     = unit / kGqaKvQuantGroups;
    const int kv_head = tmp % Geometry::KVHeads;
    const int token   = tmp / Geometry::KVHeads;

    const int position = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();

    int page = lane == 0
                   ? paged_kv_physical_page(block_table, position)
                   : 0;

    const int page_off = position & kPagedKVPageMask;
    const int d0       = group * kGqaKvQuantGroup + 4 * lane;

    float k0 = 0.0f;
    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;
    float v0 = 0.0f;
    float v1 = 0.0f;
    float v2 = 0.0f;
    float v3 = 0.0f;

    if (lane < 16) {
        const std::int64_t src0 =
            gqa_kv_quant_src_index<Geometry>(kv_head, d0 + 0, token);
        const std::int64_t src1 =
            gqa_kv_quant_src_index<Geometry>(kv_head, d0 + 1, token);
        const std::int64_t src2 =
            gqa_kv_quant_src_index<Geometry>(kv_head, d0 + 2, token);
        const std::int64_t src3 =
            gqa_kv_quant_src_index<Geometry>(kv_head, d0 + 3, token);

        k0 = __bfloat162float(k[src0]);
        k1 = __bfloat162float(k[src1]);
        k2 = __bfloat162float(k[src2]);
        k3 = __bfloat162float(k[src3]);

        v0 = __bfloat162float(v[src0]);
        v1 = __bfloat162float(v[src1]);
        v2 = __bfloat162float(v[src2]);
        v3 = __bfloat162float(v[src3]);
    }

    float k_abs = fmaxf(
        fmaxf(fabsf(k0), fabsf(k1)),
        fmaxf(fabsf(k2), fabsf(k3)));

    float v_abs = fmaxf(
        fmaxf(fabsf(v0), fabsf(v1)),
        fmaxf(fabsf(v2), fabsf(v3)));

    k_abs = warp_max(k_abs, FullMask);
    v_abs = warp_max(v_abs, FullMask);

    // Q2 scale semantics differ from Q4:
    // the FP16 scale is the reconstructed outer level itself (absmax),
    // not absmax / 7.
    const __half ksh = __float2half_rn(k_abs);
    const __half vsh = __float2half_rn(v_abs);

    const float ks = __half2float(ksh);
    const float vs = __half2float(vsh);

    page = __shfl_sync(FullMask, page, 0);

    if (lane < 16) {
        const std::int64_t code_off =
            gqa_kv_q2_code_index<Geometry>(
                page, kv_head, d0, page_off);

        cache_k[code_off] =
            gqa_kv_pack_q2(
                gqa_kv_quant_q2_code(k0, ks),
                gqa_kv_quant_q2_code(k1, ks),
                gqa_kv_quant_q2_code(k2, ks),
                gqa_kv_quant_q2_code(k3, ks));

        cache_v[code_off] =
            gqa_kv_pack_q2(
                gqa_kv_quant_q2_code(v0, vs),
                gqa_kv_quant_q2_code(v1, vs),
                gqa_kv_quant_q2_code(v2, vs),
                gqa_kv_quant_q2_code(v3, vs));
    }

    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index<Geometry>(
                page, kv_head, group, page_off);

        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

} // namespace ninfer::ops
