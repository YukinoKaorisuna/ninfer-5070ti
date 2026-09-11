#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlashPersistentState {
    CyclicKVCache local;
    std::optional<qwen3_6::PagedKVCache> full;
    Tensor prefill_projected;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    [[nodiscard]] std::size_t rewrite_checkpoint_lane_bytes() const noexcept;

    void save_rewrite_checkpoint(std::int32_t lane, void* host, std::size_t host_bytes,
                                 cudaStream_t stream) const;

    void restore_rewrite_checkpoint(const void* host, std::size_t host_bytes,
                                    std::int32_t lane, cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
