#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                    = W8DFlash2AttentionProjectionGeometry;
constexpr std::int32_t kQueryRows = 4096;
constexpr std::int32_t kKvRows    = 1024;
using Output                      = W8SplitOutput3<kQueryRows, kKvRows, kKvRows>;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_small_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                        cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((kQueryRows % Schedule::kRowsPerCta) == 0);
    static_assert((kKvRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_small_exact<kW8DFlash2AttentionFirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallLaunchers = make_small_launchers(
    std::make_index_sequence<kW8DFlash2AttentionLastSmallT - kW8DFlash2AttentionFirstSmallT + 1>{});

template <class Schedule, bool Full>
void launch_mma_slice(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                      cudaStream_t stream) {
    static_assert((kQueryRows % Schedule::BM) == 0);
    static_assert((kKvRows % Schedule::BM) == 0);
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    const dim3 grid(Geometry::kOutputRows / Schedule::BM,
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Geometry::kOutputRows,
            Geometry::kInputRows, x.ne[1], Geometry::kInputRows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        if ((count % Schedule::BN) == 0) {
            launch_mma_slice<Schedule, true>(x_slice, weight, q_slice, k_slice, v_slice, stream);
        } else {
            launch_mma_slice<Schedule, false>(x_slice, weight, q_slice, k_slice, v_slice, stream);
        }
    });
}

} // namespace

void w8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kW8DFlash2AttentionFirstSmallT || x.ne[1] > kW8DFlash2AttentionLastSmallT) {
        throw std::invalid_argument("W8 DFlash2 attention input small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8DFlash2AttentionFirstSmallT);
    kSmallLaunchers[index](x, weight, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

void w8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
