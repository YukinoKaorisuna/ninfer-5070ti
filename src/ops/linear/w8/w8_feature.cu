#include "ops/linear/w8/w8_feature.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <int Rows>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Keep the predicated profile even on full tiles: the Full specialization regresses T=64.
    // A single activation stage makes K128 fit while retaining the common code/scale pipeline.
    using Schedule = W8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    w8_rowsplit_gemm_mma_kernel<Schedule, false><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_w8_feature_r16_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<16>(x, w, out, stream);
}

void launch_w8_feature_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<32>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
