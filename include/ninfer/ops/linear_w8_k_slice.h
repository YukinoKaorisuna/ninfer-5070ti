#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/*
 * Experimental W8G32 partial-K projection.
 *
 * x:
 *   contiguous BF16 [k_extent,T]
 *
 * parent:
 *   canonical W8G32_F16S RowSplit [N,parent_k]
 *
 * accumulator:
 *   contiguous FP32 [N,T]
 *
 * k_begin/k_extent are logical K values and must be group-aligned.
 */
void linear_w8_k_slice_accumulate(const Tensor& x,
                                  const Weight& parent,
                                  std::int32_t k_begin,
                                  std::int32_t k_extent,
                                  Tensor& accumulator,
                                  bool initialize,
                                  cudaStream_t stream);

/*
 * Final single conversion:
 *
 * FP32 [N,T] -> BF16 [N,T]
 */
void linear_w8_fp32_materialize(const Tensor& accumulator,
                                Tensor& output,
                                cudaStream_t stream);

} // namespace ninfer::ops
