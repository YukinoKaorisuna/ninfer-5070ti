#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void constrained_choice_launch(const Tensor& logits,
                               const Tensor& candidate_ids,
                               Tensor& probabilities,
                               Tensor& winners,
                               std::int32_t valid_rows,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
