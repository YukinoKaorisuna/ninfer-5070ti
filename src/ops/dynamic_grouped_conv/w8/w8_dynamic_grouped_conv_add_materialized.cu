#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRows   = 5120;
constexpr std::int32_t kWidth  = 8;
constexpr std::int32_t kGroups = 320;
using Launch                   = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <std::int32_t InputRows, int BatchSize>
struct ProjectionSchedule {
    // RTX 5090 cold-cache winners over the complete production domain. Attention uses eight
    // K-split warps through B=5 and four thereafter; bypassing L1 for activation staging wins at
    // B=4,5,7,8. The longer MLP projection switches from eight to four warps after B=4.
    static constexpr int kKWarps =
        InputRows == 4096 ? (BatchSize <= 5 ? 8 : 4) : (BatchSize <= 4 ? 8 : 4);
    static constexpr int kMinBlocks = kKWarps == 8 ? 2 : 3;
    static constexpr Cache kActivationCache =
        InputRows == 4096 && (BatchSize == 4 || BatchSize == 5 || BatchSize == 7 || BatchSize == 8)
            ? Cache::cg
            : Cache::ca;
    using Type = W8SmallTMmaSchedule<kKWarps, kWidth * BatchSize, kMinBlocks,
                                     W8SmallTMmaScaleAccess::Shared, kActivationCache>;
};

template <std::int32_t InputRows, int BatchSize>
void launch_projection(const Tensor& x, const Weight& weight, Tensor& projected,
                       cudaStream_t stream) {
    using Geometry        = W8LinearGeometry<kRows, InputRows>;
    using Schedule        = typename ProjectionSchedule<InputRows, BatchSize>::Type;
    constexpr int kCols   = kWidth * BatchSize;
    constexpr int kBlocks = kRows / W8SmallTMmaIdentityRows::kOutputRowsPerCta;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(projected.data), kRows};
    w8_small_t_mma_kernel<Geometry, kCols, Schedule, W8ContiguousOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t InputRows, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_projection<InputRows, 1 + static_cast<int>(Offsets)>...};
}

constexpr auto kAttentionLaunchers = make_launchers<4096>(std::make_index_sequence<8>{});
constexpr auto kMlpLaunchers       = make_launchers<17408>(std::make_index_sequence<8>{});

__global__ void finish_kernel(const __nv_bfloat16* __restrict__ projected,
                              const __nv_bfloat16* __restrict__ base_kernel,
                              const __nv_bfloat16* __restrict__ finish_delta,
                              __nv_bfloat16* __restrict__ residual) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int32_t col = static_cast<std::int32_t>(blockIdx.y);
    if (row >= kRows) { return; }

    const std::int32_t index         = col * kRows + row;
    const std::int32_t group         = row / 16;
    const std::int64_t delta0_offset = group + static_cast<std::int64_t>(kGroups) * (2 * col);
    float value                      = __bfloat162float(residual[index]);
    const float current_coefficient =
        __bfloat162float(base_kernel[row + static_cast<std::int64_t>(2) * kRows]) +
        __bfloat162float(finish_delta[delta0_offset]);
    value = fmaf(current_coefficient, __bfloat162float(projected[index]), value);
    if ((col & 7) != 0) {
        const float previous_coefficient =
            __bfloat162float(base_kernel[row + static_cast<std::int64_t>(3) * kRows]) +
            __bfloat162float(finish_delta[delta0_offset + kGroups]);
        value = fmaf(previous_coefficient, __bfloat162float(projected[index - kRows]), value);
    }
    residual[index] = __float2bfloat16_rn(value);
}

} // namespace

void w8_dynamic_grouped_conv_add_materialized_launch(const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream) {
    if (x.ne[2] < 1 || x.ne[2] > 8) {
        throw std::invalid_argument("W8 dynamic grouped conv add materialized: invalid B");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[2] - 1);
    const auto& launchers   = x.ne[0] == 4096 ? kAttentionLaunchers : kMlpLaunchers;
    launchers[index](x, weight, projected, stream);

    const dim3 grid(static_cast<unsigned>((kRows + 255) / 256),
                    static_cast<unsigned>(kWidth * x.ne[2]), 1U);
    finish_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                            static_cast<const __nv_bfloat16*>(base_kernel.data),
                                            static_cast<const __nv_bfloat16*>(finish_delta.data),
                                            static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
