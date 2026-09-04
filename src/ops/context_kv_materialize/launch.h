#pragma once

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops::detail {

void context_kv_materialize_decode_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, Tensor& key_scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
