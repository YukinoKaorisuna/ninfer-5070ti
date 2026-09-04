#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void bf16_dynamic_grouped_conv_prepare_partial_launch(std::int32_t split_k,
                                                      const Tensor& normalized,
                                                      const Weight& kernel_projection_weight,
                                                      float* projection_partial,
                                                      cudaStream_t stream);

void bf16_dynamic_grouped_conv_prepare_reduce_launch(std::int32_t split_k,
                                                     const Tensor& base_kernel,
                                                     const float* projection_partial,
                                                     Tensor& prepared, Tensor& finish_delta,
                                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
