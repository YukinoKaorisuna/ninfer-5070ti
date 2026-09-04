#include "ninfer/ops/context_kv_materialize.h"

#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/rmsnorm_rope.h"

#include "core/layout.h"
#include "ops/context_kv_materialize/launch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden   = 5120;
constexpr std::int32_t kKVSize   = 1024;
constexpr std::int32_t kHeadDim  = 128;
constexpr std::int32_t kKVHeads  = 8;
constexpr std::int32_t kCapacity = 2048;
constexpr std::int32_t kWidth    = 8;
constexpr const char* kOp        = "context_kv_materialize";

enum class Route : std::uint8_t { DecodeDirect, PrefillComposed };

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_weight(const Weight& weight, const char* name) {
    constexpr std::uint64_t kCodeBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden);
    constexpr std::uint64_t kScaleBytes =
        static_cast<std::uint64_t>(kKVSize) * static_cast<std::uint64_t>(kHidden / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kKVSize || weight.k != kHidden ||
        weight.shape[0] != kKVSize || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kKVSize || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < kCodeBytes + kScaleBytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

Route require_profile(std::int32_t width, std::int32_t batch) {
    if (batch < 1 || batch > 8 || width < 1) {
        throw std::invalid_argument("context_kv_materialize: invalid W/B");
    }
    if (width == kWidth) return Route::DecodeDirect;
    if (batch == 1 && width <= kCapacity) return Route::PrefillComposed;
    throw std::invalid_argument("context_kv_materialize: unsupported W/B profile");
}

void require_interval(std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    if (batch < 1 || batch > 8 || min_width < 1 || max_width < min_width) {
        throw std::invalid_argument("context_kv_materialize workspace: invalid interval");
    }
    if (batch == 1) {
        if (max_width > kCapacity) {
            throw std::invalid_argument("context_kv_materialize workspace: W exceeds 2048");
        }
        return;
    }
    if (min_width != kWidth || max_width != kWidth) {
        throw std::invalid_argument(
            "context_kv_materialize workspace: batched profile requires W=8");
    }
}

template <class Allocator>
Tensor allocate_decode(Allocator& allocator, std::int32_t columns) {
    return allocator.alloc(
        DType::BF16, {kKVSize, columns, static_cast<std::int32_t>(kContextKVMaterializeLayers)});
}

struct ComposedWorkspace {
    Tensor key;
    Tensor value;
};

template <class Allocator>
ComposedWorkspace allocate_composed(Allocator& allocator, std::int32_t width) {
    return {
        allocator.alloc(DType::BF16, {kKVSize, width}),
        allocator.alloc(DType::BF16, {kKVSize, width}),
    };
}

std::size_t decode_capacity(std::int32_t batch) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_decode(layout, kWidth * batch);
    return layout.peak_bytes(1);
}

std::size_t composed_capacity(std::int32_t width) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_composed(layout, width);
    return layout.peak_bytes(1);
}

void validate_cache(const CyclicKVCacheLayerView& cache, std::int32_t padded,
                    std::int32_t lane_capacity) {
    if (cache.capacity != kCapacity ||
        cache.padded_capacity != static_cast<std::uint32_t>(padded) ||
        cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim ||
        cache.lane_capacity != lane_capacity || padded < kCapacity || lane_capacity <= 0) {
        throw std::invalid_argument("context_kv_materialize: invalid cyclic cache geometry");
    }
    require_tensor(cache.k, DType::BF16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache K");
    require_tensor(cache.v, DType::FP16, kHeadDim, padded, kKVHeads, lane_capacity, 16, "cache V");
}

void composed_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t width    = context.ne[1];
    auto scope                  = workspace.scope();
    ComposedWorkspace scratch   = allocate_composed(workspace, width);
    const Tensor flat_context   = context.view({kHidden, width});
    const Tensor flat_positions = positions.view({width});
    const KVCacheAppendPrefixExecutionEnvelope append_envelope{envelope.min_count,
                                                               envelope.max_count};
    for (const ContextKVMaterializeLayerView& layer : layers) {
        linear_pair(flat_context, layer.key_weight, layer.value_weight, scratch.key, scratch.value,
                    stream);
        Tensor key = scratch.key.view({kHeadDim, kKVHeads, width});
        rmsnorm_rope(flat_positions, layer.key_norm_weight, key, stream);
        Tensor key_batch   = key.view({kHeadDim, kKVHeads, width, 1});
        Tensor value_batch = scratch.value.view({kHeadDim, kKVHeads, width, 1});
        kv_cache_append_prefix(key_batch, value_batch, positions, counts, state_slots,
                               append_envelope, layer.cache, stream);
    }
}

} // namespace

std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                            std::int32_t min_width,
                                                            std::int32_t max_width) {
    require_interval(batch_size, min_width, max_width);
    if (batch_size > 1) return decode_capacity(batch_size);

    std::size_t capacity = 0;
    if (min_width <= kWidth && kWidth <= max_width) { capacity = decode_capacity(1); }
    if (min_width != kWidth || max_width != kWidth) {
        const std::int32_t largest_composed = max_width == kWidth ? kWidth - 1 : max_width;
        capacity = std::max(capacity, composed_capacity(largest_composed));
    }
    return capacity;
}

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t width = context.ne[1];
    const std::int32_t batch = context.ne[2];
    const Route route        = require_profile(width, batch);
    require_tensor(context, DType::BF16, kHidden, width, batch, 1, 16, "context");
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions");
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts");
    require_tensor(state_slots, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "state slots");
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > static_cast<std::uint32_t>(kCapacity)) {
        throw std::invalid_argument("context_kv_materialize: invalid execution envelope");
    }

    if (layers.front().cache.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("context_kv_materialize: padded capacity exceeds int32");
    }
    const std::int32_t padded = static_cast<std::int32_t>(layers.front().cache.padded_capacity);
    const std::int32_t lane_capacity = layers.front().cache.lane_capacity;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const ContextKVMaterializeLayerView& layer = layers[index];
        require_weight(layer.key_weight, "key weight");
        require_weight(layer.value_weight, "value weight");
        require_tensor(layer.key_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, 4, "key norm weight");
        validate_cache(layer.cache, padded, lane_capacity);
    }

    if (envelope.max_count == 0) return;
    if (route == Route::DecodeDirect) {
        auto scope         = workspace.scope();
        Tensor key_scratch = allocate_decode(workspace, width * batch);
        detail::context_kv_materialize_decode_launch(context, positions, counts, state_slots,
                                                     layers, envelope, key_scratch, stream);
        return;
    }
    composed_materialize(context, positions, counts, state_slots, layers, envelope, workspace,
                         stream);
}

} // namespace ninfer::ops
