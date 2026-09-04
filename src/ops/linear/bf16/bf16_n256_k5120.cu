#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_n256_k5120.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Schedule = Bf16N256K5120MmaSchedule;

void launch_chunk(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                  std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = 256 / Schedule::kOutputRowsPerCta;
    const dim3 grid(kRowTiles, (tokens + Schedule::kTileTokens - 1) / Schedule::kTileTokens);
    static const cudaError_t attr =
        cudaFuncSetAttribute(bf16_n256_k5120_mma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
    CUDA_CHECK(attr);
    bf16_n256_k5120_mma_kernel<<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
        x, weight, out, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_n256_k5120(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    constexpr std::int32_t kRows        = 256;
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kChunkTokens = 64;
    const auto* x_data                  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* weight_data             = static_cast<const __nv_bfloat16*>(weight.qdata);
    auto* out_data                      = static_cast<__nv_bfloat16*>(out.data);

    for (std::int32_t token0 = 0; token0 < x.ne[1];) {
        const std::int32_t tokens = std::min(kChunkTokens, x.ne[1] - token0);
        launch_chunk(x_data + static_cast<std::int64_t>(token0) * kHidden, weight_data,
                     out_data + static_cast<std::int64_t>(token0) * kRows, tokens, stream);
        token0 += tokens;
    }
}

} // namespace ninfer::ops::detail
