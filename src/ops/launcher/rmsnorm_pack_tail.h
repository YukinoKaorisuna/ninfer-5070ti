#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              std::int32_t batch, cudaStream_t stream);

} // namespace ninfer::ops::detail
