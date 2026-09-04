#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden          = 5120;
constexpr int kWidth           = 8;
constexpr int kGroups          = 320;
constexpr int kTaps            = 2;
constexpr int kCoefficientRows = 1280;
constexpr int kThreads         = 128;

template <int SplitK>
__global__ __launch_bounds__(kThreads, 4) void dynamic_grouped_conv_prepare_reduce_kernel(
    const __nv_bfloat16* __restrict__ base_kernel, const float* __restrict__ projection_partial,
    __nv_bfloat16* __restrict__ prepared, __nv_bfloat16* __restrict__ finish_delta,
    int batch_size) {
    static_assert(SplitK == 4 || SplitK == 8);
    __shared__ float projected[4][kWidth];
    __shared__ __nv_bfloat16 normalized[kWidth][16];

    const int tid   = static_cast<int>(threadIdx.x);
    const int group = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);

    if (tid < 4 * kWidth) {
        const int coefficient = tid / kWidth;
        const int position    = tid - coefficient * kWidth;
        const int parent_row  = coefficient * kGroups + group;
        float value           = 0.0F;
#pragma unroll
        for (int split = 0; split < SplitK; ++split) {
            const std::int64_t offset =
                ((static_cast<std::int64_t>(split) * batch_size + batch) * kWidth + position) *
                    kCoefficientRows +
                parent_row;
            value += projection_partial[offset];
        }
        projected[coefficient][position] = value;
    }

    const int position = tid / 16;
    const int channel  = tid - position * 16;
    const int hidden   = group * 16 + channel;
    const std::int64_t current_offset =
        static_cast<std::int64_t>(batch * kWidth + position) * kHidden + hidden;
    normalized[position][channel] = prepared[current_offset];
    __syncthreads();

    if (tid < 2 * kWidth) {
        const int tap      = tid / kWidth;
        const int position = tid - tap * kWidth;
        const float delta  = projected[2 + tap][position];
        const std::int64_t offset =
            ((static_cast<std::int64_t>(batch) * kWidth + position) * kTaps + tap) * kGroups +
            group;
        finish_delta[offset] = __float2bfloat16_rn(delta);
    }

    const float current = __bfloat162float(normalized[position][channel]);
    const float delta0  = projected[0][position];
    const float delta1  = projected[1][position];
    const float base0   = __bfloat162float(base_kernel[hidden]);
    const float base1   = __bfloat162float(base_kernel[kHidden + hidden]);
    float output        = (base0 + delta0) * current;
    if (position > 0) {
        const float previous = __bfloat162float(normalized[position - 1][channel]);
        output               = fmaf(base1 + delta1, previous, output);
    }
    prepared[current_offset] = __float2bfloat16_rn(output);
}

template <int SplitK>
void launch(const Tensor& base_kernel, const float* projection_partial, Tensor& prepared,
            Tensor& finish_delta, cudaStream_t stream) {
    const int batch_size = prepared.ne[2];
    const dim3 grid(kGroups, static_cast<unsigned>(batch_size), 1u);
    dynamic_grouped_conv_prepare_reduce_kernel<SplitK><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(base_kernel.data), projection_partial,
        static_cast<__nv_bfloat16*>(prepared.data), static_cast<__nv_bfloat16*>(finish_delta.data),
        batch_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_dynamic_grouped_conv_prepare_reduce_launch(std::int32_t split_k,
                                                     const Tensor& base_kernel,
                                                     const float* projection_partial,
                                                     Tensor& prepared, Tensor& finish_delta,
                                                     cudaStream_t stream) {
    switch (split_k) {
    case 4:
        launch<4>(base_kernel, projection_partial, prepared, finish_delta, stream);
        return;
    case 8:
        launch<8>(base_kernel, projection_partial, prepared, finish_delta, stream);
        return;
    default:
        throw std::invalid_argument("dynamic grouped conv prepare: unsupported split-K");
    }
}

} // namespace ninfer::ops::detail
