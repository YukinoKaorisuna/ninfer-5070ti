#include "ops/launcher/rmsnorm_pack_tail.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm_pack_tail.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              std::int32_t batch, cudaStream_t stream) {
    constexpr int kBlock = kRmsnormPackTailBlock;
    const dim3 grid(7, static_cast<unsigned int>(batch));
    rmsnorm_pack_tail_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(input.data),
        static_cast<const __nv_bfloat162*>(weight.data), static_cast<__nv_bfloat162*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
