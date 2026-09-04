#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                       = W8MtpGateUpProjectionGeometry;
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kFirstExactT  = 1;
constexpr std::int32_t kLastExactT   = 51;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
struct DFlash2SmallTSchedule {
    static_assert(ActiveTokens >= kFirstExactT && ActiveTokens <= kLastExactT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                       : ActiveTokens <= 48 ? 48
                                                            : 56;
    static constexpr int kKWarps     = ActiveTokens >= 22 && ActiveTokens <= 24 ? 8 : 4;
    static constexpr auto kActivationStage =
        ActiveTokens <= 4 || (ActiveTokens >= 9 && ActiveTokens <= 15)
            ? W8SmallTMmaActivationStage::PaddedZero
            : W8SmallTMmaActivationStage::ActiveOnly;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 2, W8SmallTMmaScaleAccess::Shared,
                                     Cache::ca, Cache::cg, kActivationStage>;
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule  = typename DFlash2SmallTSchedule<ActiveTokens>::Type;
    using RowPolicy = W8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, W8ContiguousOutput,
                          W8SwiGluDirectEpilogue, RowPolicy, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, row_policy);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void w8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFirstExactT);
    kLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
