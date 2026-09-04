#include "ops/context_kv_materialize/launch.h"

#include "core/device.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"
#include "ops/rmsnorm_rope/d128.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kLayers      = static_cast<int>(kContextKVMaterializeLayers);
constexpr int kMatrices    = 2 * kLayers;
constexpr int kHidden      = 5120;
constexpr int kRows        = 1024;
constexpr int kHeadDim     = 128;
constexpr int kHeads       = 8;
constexpr int kWidth       = 8;
constexpr int kCapacity    = 2048;
constexpr int kRowsPerCta  = 8;
constexpr int kColsPerTile = 4;
constexpr int kStages      = 2;

struct DeviceLayerView {
    const std::uint8_t* key_codes;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_codes;
    const std::uint8_t* value_scales;
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

struct DeviceLayers {
    DeviceLayerView layer[kLayers];
};

__global__ __launch_bounds__(kRowsPerCta * 32) void context_kv_project_decode_kernel(
    const __nv_bfloat16* __restrict__ context, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ state_slots,
    DeviceLayers layers, __nv_bfloat16* __restrict__ key_scratch, std::int32_t batch_size,
    std::int32_t min_count, std::int32_t max_count) {
    using Schedule             = W8RowSplitSimtSchedule;
    constexpr int kPrefetch    = kStages - 1;
    constexpr int kHighU4Alloc = Schedule::kHighU4 > 0 ? Schedule::kHighU4 : 1;
    constexpr int kFullSlabs   = kHidden / 1024;

    __shared__ __align__(16) uint4 s_nib[kRowsPerCta][kStages][Schedule::kNibU4];
    __shared__ __align__(16) uint4 s_hi[kRowsPerCta][kStages][kHighU4Alloc];
    __shared__ __align__(16) std::uint32_t s_sc[kRowsPerCta][kStages][Schedule::kScaleU32];

    const int matrix            = static_cast<int>(blockIdx.z);
    const int layer_index       = matrix >> 1;
    const bool value            = (matrix & 1) != 0;
    const DeviceLayerView layer = layers.layer[layer_index];
    const std::uint8_t* codes   = value ? layer.value_codes : layer.key_codes;
    const std::uint8_t* scales  = value ? layer.value_scales : layer.key_scales;

    const int col0         = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int batch        = col0 / kWidth;
    const int local_column = col0 - batch * kWidth;
    if (batch >= batch_size) return;
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;
    const int ncols = min(kColsPerTile, max(0, count - local_column));
    if (ncols == 0) return;

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int row         = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    const int kg          = kHidden / Schedule::Codec::kGroupK;
    const auto* code_row  = codes + static_cast<std::int64_t>(row) * kg * 32;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * kg * 2;
    const auto* context0  = context + static_cast<std::int64_t>(col0) * kHidden;

    float accumulators[kColsPerTile]{};

#pragma unroll
    for (int stage = 0; stage < kPrefetch; ++stage) {
        w8_simt_issue_slab<Schedule>(s_nib[warp][stage], s_hi[warp][stage], s_sc[warp][stage],
                                     code_row, nullptr, scale_row, stage, lane);
    }

#pragma unroll 1
    for (int slab = 0; slab < kFullSlabs; ++slab) {
        const int fetch = slab + kPrefetch;
        if (fetch < kFullSlabs) {
            const int buffer = fetch % kStages;
            w8_simt_issue_slab<Schedule>(s_nib[warp][buffer], s_hi[warp][buffer],
                                         s_sc[warp][buffer], code_row, nullptr, scale_row, fetch,
                                         lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();
        const int buffer = slab % kStages;
        w8_simt_consume_slab<Schedule, kColsPerTile>(
            context0, static_cast<std::int64_t>(slab) * 1024, kHidden, ncols, s_nib[warp][buffer],
            s_hi[warp][buffer], s_sc[warp][buffer], lane, accumulators);
        __syncwarp();
    }

#pragma unroll
    for (int column = 0; column < kColsPerTile; ++column) {
        if (column >= ncols) continue;
        float result = warp_reduce_sum(accumulators[column]);
        if (lane != 0) continue;
        const int physical_column       = col0 + column;
        const __nv_bfloat16 represented = __float2bfloat16_rn(result);
        if (!value) {
            const std::int64_t destination =
                static_cast<std::int64_t>(row) +
                static_cast<std::int64_t>(kRows) *
                    (physical_column +
                     static_cast<std::int64_t>(kWidth * batch_size) * layer_index);
            key_scratch[destination] = represented;
            continue;
        }

        const int position = positions[physical_column];
        const int slot     = position & (kCapacity - 1);
        const int head     = row / kHeadDim;
        const int dim      = row - head * kHeadDim;
        const std::int64_t destination =
            static_cast<std::int64_t>(dim) +
            static_cast<std::int64_t>(kHeadDim) *
                (slot + static_cast<std::int64_t>(layer.padded_capacity) *
                            (head + kHeads * state_slots[batch]));
        layer.cache_v[destination] = __float2half_rn(__bfloat162float(represented));
    }
}

__global__ __launch_bounds__(256) void context_kv_key_post_decode_kernel(
    const __nv_bfloat16* __restrict__ key_scratch, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ state_slots,
    DeviceLayers layers, std::int32_t batch_size, std::int32_t min_count, std::int32_t max_count) {
    constexpr int kHalf        = kHeadDim / 2;
    constexpr int kWeightPairs = kHeadDim / 2;

    const int physical_column = static_cast<int>(blockIdx.x);
    const int layer_index     = static_cast<int>(blockIdx.y);
    const int batch           = physical_column / kWidth;
    const int local_column    = physical_column - batch * kWidth;
    if (batch >= batch_size) return;
    const int count = counts[batch];
    if (count < min_count || count > max_count || local_column >= count) return;

    const DeviceLayerView layer = layers.layer[layer_index];
    const int position          = positions[physical_column];
    const int slot              = position & (kCapacity - 1);
    const int state_slot        = state_slots[batch];

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    __shared__ __nv_bfloat162 weight_cache[kWeightPairs];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        dflash_rope_sincos(positions, physical_column, pair, &sin_cache[pair], &cos_cache[pair]);
        weight_cache[pair] = reinterpret_cast<const __nv_bfloat162*>(layer.key_norm)[pair];
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int head = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t token_base =
        static_cast<std::int64_t>(kRows) *
        (physical_column + static_cast<std::int64_t>(kWidth * batch_size) * layer_index);
    const auto* input =
        reinterpret_cast<const __nv_bfloat162*>(key_scratch + token_base + head * kHeadDim);
    const __nv_bfloat162 input0       = input[lane];
    const __nv_bfloat162 input1       = input[lane + 32];
    const RmsnormRopeD128Pair rotated = rmsnorm_rope_d128_head(
        input0, input1, weight_cache[lane], weight_cache[lane + 32], cos_cache, sin_cache, lane);

    const std::int64_t cache_base =
        static_cast<std::int64_t>(kHeadDim) *
        (slot + static_cast<std::int64_t>(layer.padded_capacity) * (head + kHeads * state_slot));
    auto* cache_output      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + cache_base);
    cache_output[lane]      = rotated.first;
    cache_output[lane + 32] = rotated.second;
}

DeviceLayers make_device_layers(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers) {
    DeviceLayers result{};
    for (int index = 0; index < kLayers; ++index) {
        const ContextKVMaterializeLayerView& source = layers[static_cast<std::size_t>(index)];
        result.layer[index]                         = {
            static_cast<const std::uint8_t*>(source.key_weight.qdata),
            static_cast<const std::uint8_t*>(source.key_weight.scales),
            static_cast<const std::uint8_t*>(source.value_weight.qdata),
            static_cast<const std::uint8_t*>(source.value_weight.scales),
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(source.cache.k.data),
            static_cast<__half*>(source.cache.v.data),
            static_cast<std::int32_t>(source.cache.padded_capacity),
        };
    }
    return result;
}

} // namespace

void context_kv_materialize_decode_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, Tensor& key_scratch, cudaStream_t stream) {
    const std::int32_t batch         = context.ne[2];
    const DeviceLayers device_layers = make_device_layers(layers);
    const dim3 projection_grid(kRows / kRowsPerCta, 2U * static_cast<unsigned>(batch), kMatrices);
    context_kv_project_decode_kernel<<<projection_grid, kRowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(context.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(state_slots.data), device_layers,
        static_cast<__nv_bfloat16*>(key_scratch.data), batch,
        static_cast<std::int32_t>(envelope.min_count),
        static_cast<std::int32_t>(envelope.max_count));
    CUDA_CHECK(cudaGetLastError());

    const dim3 post_grid(kWidth * static_cast<unsigned>(batch), kLayers, 1U);
    context_kv_key_post_decode_kernel<<<post_grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key_scratch.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(state_slots.data), device_layers, batch,
        static_cast<std::int32_t>(envelope.min_count),
        static_cast<std::int32_t>(envelope.max_count));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
