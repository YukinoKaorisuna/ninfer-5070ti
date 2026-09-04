#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"

#include "ninfer/ops/rmsnorm.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kWidth           = 8;
constexpr std::int32_t kCoefficientRows = 1280;

struct Plan {
    std::int32_t split_k;
    std::size_t workspace_bytes;
};

Plan resolve_plan(std::int32_t batch_size) {
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("dynamic grouped conv prepare: B must be in [1,8]");
    }
    const std::int32_t split_k = batch_size <= 6 ? 8 : 4;
    const std::size_t tokens =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(kWidth);
    const std::size_t workspace_bytes = static_cast<std::size_t>(split_k) * tokens *
                                        static_cast<std::size_t>(kCoefficientRows) * sizeof(float);
    return {split_k, workspace_bytes};
}

void execute_plan(const Plan& plan, const Tensor& residual, const Tensor& norm_weight, float eps,
                  const Tensor& base_kernel, const Weight& kernel_projection_weight,
                  Tensor& prepared, Tensor& finish_delta, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    auto scope                    = workspace.scope();
    DeviceSpan projection_partial = workspace.alloc_bytes(plan.workspace_bytes);
    rmsnorm(residual, norm_weight, eps, false, prepared, stream);
    bf16_dynamic_grouped_conv_prepare_partial_launch(
        plan.split_k, prepared, kernel_projection_weight,
        static_cast<float*>(projection_partial.data), stream);
    bf16_dynamic_grouped_conv_prepare_reduce_launch(
        plan.split_k, base_kernel, static_cast<const float*>(projection_partial.data), prepared,
        finish_delta, stream);
}

} // namespace

std::size_t
bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(std::int32_t min_batch_size,
                                                           std::int32_t max_batch_size) {
    if (min_batch_size < 1 || max_batch_size > 8 || max_batch_size < min_batch_size) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare workspace: invalid batch interval");
    }
    std::size_t maximum = 0;
    for (std::int32_t batch_size = min_batch_size; batch_size <= max_batch_size; ++batch_size) {
        maximum = std::max(maximum, resolve_plan(batch_size).workspace_bytes);
    }
    return maximum;
}

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm_weight,
                                                float eps, const Tensor& base_kernel,
                                                const Weight& kernel_projection_weight,
                                                Tensor& prepared, Tensor& finish_delta,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    execute_plan(resolve_plan(residual.ne[2]), residual, norm_weight, eps, base_kernel,
                 kernel_projection_weight, prepared, finish_delta, workspace, stream);
}

} // namespace ninfer::ops::detail
