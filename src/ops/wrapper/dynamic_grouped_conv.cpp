#include "ninfer/ops/dynamic_grouped_conv.h"

#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kWidth           = 8;
constexpr std::int32_t kGroups          = 320;
constexpr std::int32_t kTaps            = 2;
constexpr std::int32_t kSides           = 2;
constexpr std::int32_t kCoefficientRows = kGroups * kTaps * kSides;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t d0, std::int32_t d1,
                    std::int32_t d2, std::int32_t d3, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 ||
        tensor.ne[3] != d3 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("dynamic grouped conv prepare: invalid ") + label);
    }
}

void require_kernel_projection_weight(const Weight& weight) {
    constexpr std::uint64_t kPayloadBytes =
        static_cast<std::uint64_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < kPayloadBytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != kCoefficientRows || weight.k != kHidden ||
        weight.shape[0] != kCoefficientRows || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kCoefficientRows || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.scales != nullptr || weight.group_size != 0 ||
        weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: invalid kernel_projection_weight");
    }
}

struct Range {
    const void* pointer;
    std::size_t bytes;
    const char* label;
};

bool overlaps(const Range& lhs, const Range& rhs) {
    if (lhs.bytes == 0 || rhs.bytes == 0 || lhs.pointer == nullptr || rhs.pointer == nullptr) {
        return false;
    }
    const std::uintptr_t lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.pointer);
    const std::uintptr_t rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.pointer);
    if (lhs.bytes > std::numeric_limits<std::uintptr_t>::max() - lhs_begin ||
        rhs.bytes > std::numeric_limits<std::uintptr_t>::max() - rhs_begin) {
        throw std::overflow_error("dynamic grouped conv prepare: operand range overflows");
    }
    return lhs_begin < rhs_begin + rhs.bytes && rhs_begin < lhs_begin + lhs.bytes;
}

void require_nonoverlap(const Tensor& residual, const Tensor& norm_weight,
                        const Tensor& base_kernel, const Weight& kernel_projection_weight,
                        const Tensor& prepared, const Tensor& finish_delta,
                        const WorkspaceArena& workspace) {
    constexpr std::size_t kWeightBytes =
        static_cast<std::size_t>(kCoefficientRows) * kHidden * sizeof(std::uint16_t);
    const std::array<Range, 7> ranges{{
        {residual.data, residual.bytes(), "residual"},
        {norm_weight.data, norm_weight.bytes(), "norm_weight"},
        {base_kernel.data, base_kernel.bytes(), "base_kernel"},
        {kernel_projection_weight.qdata, kWeightBytes, "kernel_projection_weight"},
        {prepared.data, prepared.bytes(), "prepared"},
        {finish_delta.data, finish_delta.bytes(), "finish_delta"},
        {workspace.base(), workspace.capacity(), "workspace"},
    }};
    for (std::size_t first = 0; first < ranges.size(); ++first) {
        for (std::size_t second = first + 1; second < ranges.size(); ++second) {
            if (overlaps(ranges[first], ranges[second])) {
                throw std::invalid_argument(std::string("dynamic grouped conv prepare: ") +
                                            ranges[first].label + " overlaps " +
                                            ranges[second].label);
            }
        }
    }
}

} // namespace

std::size_t
rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(std::int32_t min_batch_size,
                                                              std::int32_t max_batch_size) {
    return detail::bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(min_batch_size,
                                                                              max_batch_size);
}

void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const std::int32_t batch_size = residual.ne[2];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "dynamic grouped conv prepare: eps must be positive and finite");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("dynamic grouped conv prepare: B must be in [1,8]");
    }
    require_tensor(residual, DType::BF16, kHidden, kWidth, batch_size, 1, "residual");
    require_tensor(norm_weight, DType::BF16, kHidden, 1, 1, 1, "norm_weight");
    require_tensor(base_kernel, DType::BF16, kHidden, kTaps, kSides, 1, "base_kernel");
    require_tensor(prepared, DType::BF16, kHidden, kWidth, batch_size, 1, "prepared");
    require_tensor(finish_delta, DType::BF16, kGroups, kTaps, kWidth, batch_size, "finish_delta");
    require_kernel_projection_weight(kernel_projection_weight);
    require_nonoverlap(residual, norm_weight, base_kernel, kernel_projection_weight, prepared,
                       finish_delta, workspace);

    detail::bf16_dynamic_grouped_conv_prepare_dispatch(residual, norm_weight, eps, base_kernel,
                                                       kernel_projection_weight, prepared,
                                                       finish_delta, workspace, stream);
}

} // namespace ninfer::ops
