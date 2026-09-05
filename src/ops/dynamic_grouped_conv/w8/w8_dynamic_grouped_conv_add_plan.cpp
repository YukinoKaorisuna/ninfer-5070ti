#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.h"

#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRows  = 5120;
constexpr std::int32_t kWidth = 8;

struct Plan {
    std::size_t workspace_bytes;
};

Plan resolve_plan(std::int32_t input_rows, std::int32_t batch_size) {
    if (input_rows != 4096 && input_rows != 17408) {
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("linear dynamic grouped conv add: B must be in [1,8]");
    }
    return {static_cast<std::size_t>(kRows) * kWidth * static_cast<std::size_t>(batch_size) *
            sizeof(std::uint16_t)};
}

} // namespace

std::size_t w8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_batch_size, std::int32_t max_batch_size) {
    if (min_batch_size < 1 || max_batch_size > 8 || max_batch_size < min_batch_size) {
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid batch interval");
    }
    std::size_t maximum = 0;
    for (std::int32_t batch_size = min_batch_size; batch_size <= max_batch_size; ++batch_size) {
        const Plan plan = resolve_plan(input_rows, batch_size);
        if (plan.workspace_bytes > maximum) { maximum = plan.workspace_bytes; }
    }
    return maximum;
}

const char* w8_linear_dynamic_grouped_conv_add_route_name(std::int32_t input_rows,
                                                          std::int32_t batch_size) {
    (void)resolve_plan(input_rows, batch_size);
    return "dynamic_grouped_conv_add.w8.small_t.materialized_bf16";
}

void w8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& projection_weight,
                                                 const Tensor& base_kernel,
                                                 const Tensor& finish_delta, Tensor& residual,
                                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const Plan plan          = resolve_plan(x.ne[0], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage.data, DType::BF16, {kRows, kWidth, x.ne[2]});
    w8_dynamic_grouped_conv_add_materialized_launch(x, projection_weight, base_kernel, finish_delta,
                                                    residual, projected, stream);
}

} // namespace ninfer::ops::detail
