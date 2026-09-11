#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

/**
 * Fixed cyclic BF16 K/V storage with absolute-position addressing.
 *
 * A logical absolute position p resides in physical slot p % capacity. The view deliberately
 * carries no mutable frontier: callers supply the live absolute interval to the consuming Op.
 */
struct CyclicKVCacheLayerView {
    Tensor k;
    Tensor v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
};

struct CyclicKVCacheLayout {
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    std::vector<TensorRegion> k;
    std::vector<TensorRegion> v;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] CyclicKVCacheLayout
plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                     std::int32_t num_kv_heads, std::int32_t head_dim, std::int32_t lane_capacity,
                     DType value_dtype = DType::BF16);

class CyclicKVCache {
public:
    CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout);

    CyclicKVCache(const CyclicKVCache&)            = delete;
    CyclicKVCache& operator=(const CyclicKVCache&) = delete;
    CyclicKVCache(CyclicKVCache&&)                 = delete;
    CyclicKVCache& operator=(CyclicKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t layer_count() const noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept { return padded_capacity_; }

    [[nodiscard]] std::int32_t num_kv_heads() const noexcept { return num_kv_heads_; }

    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }

    [[nodiscard]] std::int32_t lane_capacity() const noexcept { return lane_capacity_; }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer) const;

    // Exact byte size of one complete lane across every K/V layer.
    [[nodiscard]] std::size_t lane_bytes() const noexcept;

    // Copies one lane's complete fixed state. Source and destination must have identical layouts.
    void copy_lane_from(const CyclicKVCache& source, std::int32_t lane, cudaStream_t stream);

    // Exact complete-lane snapshots to/from pinned host storage.
    //
    // Host layout is deterministic:
    //   layer0 K, layer0 V,
    //   layer1 K, layer1 V,
    //   ...
    //
    // host_bytes must be at least lane_bytes().
    void copy_lane_to_host(std::int32_t lane, void* host, std::size_t host_bytes,
                           cudaStream_t stream) const;
    void copy_lane_from_host(const void* host, std::size_t host_bytes, std::int32_t lane,
                             cudaStream_t stream);

private:
    std::vector<Tensor> k_;
    std::vector<Tensor> v_;
    std::uint32_t capacity_        = 0;
    std::uint32_t padded_capacity_ = 0;
    std::int32_t num_kv_heads_     = 0;
    std::int32_t head_dim_         = 0;
    std::int32_t lane_capacity_    = 0;
};

} // namespace ninfer
