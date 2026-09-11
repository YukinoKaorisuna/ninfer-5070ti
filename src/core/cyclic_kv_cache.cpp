#include "core/cyclic_kv_cache.h"

#include "core/device.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

std::uint32_t align_up_u32(std::uint32_t value, std::uint32_t alignment) {
    const std::uint64_t mask    = static_cast<std::uint64_t>(alignment) - 1U;
    const std::uint64_t aligned = (static_cast<std::uint64_t>(value) + mask) & ~mask;
    if (aligned > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Cyclic KV padded capacity is out of range");
    }
    return static_cast<std::uint32_t>(aligned);
}

} // namespace

CyclicKVCacheLayout plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers,
                                         std::uint32_t capacity, std::int32_t num_kv_heads,
                                         std::int32_t head_dim, std::int32_t lane_capacity,
                                         DType value_dtype) {
    if (layers == 0 || capacity == 0 ||
        capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        num_kv_heads <= 0 || head_dim <= 0 || lane_capacity <= 0) {
        throw std::invalid_argument("Cyclic KV geometry is invalid");
    }
    if (value_dtype != DType::BF16 && value_dtype != DType::FP16) {
        throw std::invalid_argument("Cyclic KV V dtype must be BF16 or FP16");
    }

    CyclicKVCacheLayout layout;
    layout.capacity        = capacity;
    layout.padded_capacity = align_up_u32(capacity, 128);
    layout.num_kv_heads    = num_kv_heads;
    layout.head_dim        = head_dim;
    layout.lane_capacity   = lane_capacity;
    layout.k.reserve(layers);
    layout.v.reserve(layers);
    const auto padded = static_cast<std::int32_t>(layout.padded_capacity);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "Cyclic KV layer " + std::to_string(layer);
        layout.k.push_back(builder.add_tensor(DType::BF16,
                                              {head_dim, padded, num_kv_heads, lane_capacity},
                                              kArenaAlign, prefix + " K"));
        layout.v.push_back(builder.add_tensor(value_dtype,
                                              {head_dim, padded, num_kv_heads, lane_capacity},
                                              kArenaAlign, prefix + " V"));
    }
    return layout;
}

std::size_t CyclicKVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const TensorRegion& region : k) { total += region.region.bytes; }
    for (const TensorRegion& region : v) { total += region.region.bytes; }
    return total;
}

CyclicKVCache::CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout)
    : capacity_(layout.capacity), padded_capacity_(layout.padded_capacity),
      num_kv_heads_(layout.num_kv_heads), head_dim_(layout.head_dim),
      lane_capacity_(layout.lane_capacity) {
    if (layout.k.empty() || layout.v.size() != layout.k.size() || capacity_ == 0 ||
        padded_capacity_ < capacity_ || num_kv_heads_ <= 0 || head_dim_ <= 0 ||
        lane_capacity_ <= 0) {
        throw std::invalid_argument("Cyclic KV layout is inconsistent");
    }
    const std::array<std::int32_t, 4> expected_shape{
        head_dim_, static_cast<std::int32_t>(padded_capacity_), num_kv_heads_, lane_capacity_};
    const DType value_dtype = layout.v.front().dtype;
    if (value_dtype != DType::BF16 && value_dtype != DType::FP16) {
        throw std::invalid_argument("Cyclic KV V dtype is unsupported");
    }

    k_.reserve(layout.k.size());
    v_.reserve(layout.v.size());
    for (std::size_t layer = 0; layer < layout.k.size(); ++layer) {
        if (layout.k[layer].dtype != DType::BF16 || layout.v[layer].dtype != value_dtype ||
            layout.k[layer].shape != expected_shape || layout.v[layer].shape != expected_shape) {
            throw std::invalid_argument("Cyclic KV layer layout is inconsistent");
        }
        k_.push_back(layout.k[layer].bind(backing));
        v_.push_back(layout.v[layer].bind(backing));
    }
}

std::uint32_t CyclicKVCache::layer_count() const noexcept {
    return static_cast<std::uint32_t>(k_.size());
}

CyclicKVCacheLayerView CyclicKVCache::layer_view(std::uint32_t layer) const {
    if (layer >= layer_count()) { throw std::out_of_range("Cyclic KV layer is out of range"); }
    return {
        .k               = k_[layer],
        .v               = v_[layer],
        .capacity        = capacity_,
        .padded_capacity = padded_capacity_,
        .num_kv_heads    = num_kv_heads_,
        .head_dim        = head_dim_,
        .lane_capacity   = lane_capacity_,
    };
}

void CyclicKVCache::copy_lane_from(const CyclicKVCache& source, std::int32_t lane,
                                   cudaStream_t stream) {
    if (source.layer_count() != layer_count() || source.capacity_ != capacity_ ||
        source.padded_capacity_ != padded_capacity_ || source.num_kv_heads_ != num_kv_heads_ ||
        source.head_dim_ != head_dim_ || source.lane_capacity_ != lane_capacity_) {
        throw std::invalid_argument("Cyclic KV copy requires identical layouts");
    }
    if (lane < 0 || lane >= lane_capacity_) {
        throw std::out_of_range("Cyclic KV lane is out of range");
    }
    for (std::size_t layer = 0; layer < k_.size(); ++layer) {
        Tensor destination_k = k_[layer].slice(3, lane, 1);
        Tensor destination_v = v_[layer].slice(3, lane, 1);
        Tensor source_k      = source.k_[layer].slice(3, lane, 1);
        Tensor source_v      = source.v_[layer].slice(3, lane, 1);
        CUDA_CHECK(cudaMemcpyAsync(destination_k.data, source_k.data, destination_k.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(destination_v.data, source_v.data, destination_v.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

std::size_t CyclicKVCache::lane_bytes() const noexcept {
    std::size_t total = 0;

    for (std::size_t layer = 0; layer < k_.size(); ++layer) {
        // lane_capacity_ is guaranteed positive by construction and the
        // highest tensor dimension is the lane dimension, so each lane is
        // one contiguous region.
        total += k_[layer].bytes() /
                 static_cast<std::size_t>(lane_capacity_);

        total += v_[layer].bytes() /
                 static_cast<std::size_t>(lane_capacity_);
    }

    return total;
}

void CyclicKVCache::copy_lane_to_host(
    std::int32_t lane,
    void* host,
    std::size_t host_bytes,
    cudaStream_t stream) const {

    if (lane < 0 || lane >= lane_capacity_) {
        throw std::out_of_range(
            "Cyclic KV lane is out of range");
    }

    const std::size_t required = lane_bytes();

    if (host == nullptr || host_bytes < required) {
        throw std::invalid_argument(
            "Cyclic KV host snapshot storage is too small");
    }

    auto* destination =
        static_cast<unsigned char*>(host);

    std::size_t offset = 0;

    for (std::size_t layer = 0;
         layer < k_.size();
         ++layer) {

        const Tensor source_k =
            k_[layer].slice(3, lane, 1);

        const Tensor source_v =
            v_[layer].slice(3, lane, 1);

        CUDA_CHECK(
            cudaMemcpyAsync(
                destination + offset,
                source_k.data,
                source_k.bytes(),
                cudaMemcpyDeviceToHost,
                stream));

        offset += source_k.bytes();

        CUDA_CHECK(
            cudaMemcpyAsync(
                destination + offset,
                source_v.data,
                source_v.bytes(),
                cudaMemcpyDeviceToHost,
                stream));

        offset += source_v.bytes();
    }

    if (offset != required) {
        throw std::logic_error(
            "Cyclic KV host snapshot size is inconsistent");
    }
}

void CyclicKVCache::copy_lane_from_host(
    const void* host,
    std::size_t host_bytes,
    std::int32_t lane,
    cudaStream_t stream) {

    if (lane < 0 || lane >= lane_capacity_) {
        throw std::out_of_range(
            "Cyclic KV lane is out of range");
    }

    const std::size_t required = lane_bytes();

    if (host == nullptr || host_bytes < required) {
        throw std::invalid_argument(
            "Cyclic KV host snapshot storage is too small");
    }

    const auto* source =
        static_cast<const unsigned char*>(host);

    std::size_t offset = 0;

    for (std::size_t layer = 0;
         layer < k_.size();
         ++layer) {

        Tensor destination_k =
            k_[layer].slice(3, lane, 1);

        Tensor destination_v =
            v_[layer].slice(3, lane, 1);

        CUDA_CHECK(
            cudaMemcpyAsync(
                destination_k.data,
                source + offset,
                destination_k.bytes(),
                cudaMemcpyHostToDevice,
                stream));

        offset += destination_k.bytes();

        CUDA_CHECK(
            cudaMemcpyAsync(
                destination_v.data,
                source + offset,
                destination_v.bytes(),
                cudaMemcpyHostToDevice,
                stream));

        offset += destination_v.bytes();
    }

    if (offset != required) {
        throw std::logic_error(
            "Cyclic KV host restore size is inconsistent");
    }
}

} // namespace ninfer
