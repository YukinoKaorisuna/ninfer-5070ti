#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_dynamic_grouped_conv_add_materialized_launch(const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

} // namespace ninfer::ops::detail
