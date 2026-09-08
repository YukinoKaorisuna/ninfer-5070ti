#pragma once

#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "core/arena.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q3_linear_swiglu_workspace_capacity_bytes(
    std::int32_t gate_up_rows,
    std::int32_t input_rows,
    LinearPolicy policy,
    std::int32_t min_tokens,
    std::int32_t max_tokens);

void q3_linear_swiglu_dispatch(
    const Tensor& x,
    const Weight& w,
    Tensor& out,
    LinearPolicy policy,
    WorkspaceArena& workspace,
    cudaStream_t stream);

} // namespace ninfer::ops::detail
