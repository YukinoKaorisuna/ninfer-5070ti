#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class W8DynamicConvAddSchedule { SmallT, Tiled32, Tiled72, Tiled80, Tiled88, MmaK128 };

void w8_dynamic_grouped_conv_add_materialized_launch(W8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

} // namespace ninfer::ops::detail
