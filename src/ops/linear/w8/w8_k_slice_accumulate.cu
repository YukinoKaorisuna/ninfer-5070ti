#include "ninfer/ops/linear_w8_k_slice.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

using FastSchedule =
    detail::W8RowSplitMmaGemmSchedule<
        64,   // BM
        128,  // BN
        64,   // WM
        16,   // WN
        2,    // minimum blocks
        2     // stages
    >;

constexpr std::int32_t kGroup = 32;

void validate_weight(
    const Weight& w)
{
    if (w.qtype != QType::W8G32_F16S ||
        w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 ||
        w.group_size != 32 ||
        w.group != 32 ||
        w.padded_shape[0] != w.n ||
        w.padded_shape[1] != w.k ||
        w.qdata == nullptr ||
        w.qhigh != nullptr ||
        w.scales == nullptr)
    {
        throw std::invalid_argument(
            "linear_w8_k_slice: expected canonical "
            "W8G32_F16S RowSplit weight");
    }
}

template <
    bool Full,
    detail::W8Epilogue Epilogue>
void launch_fast(
    const Tensor& x,
    const Weight& parent,
    std::int32_t k_begin,
    Tensor& accumulator,
    cudaStream_t stream)
{
    const std::int32_t rows =
        parent.n;

    const std::int32_t logical_k =
        x.ne[0];

    const std::int32_t cols =
        x.ne[1];

    /*
     * Critical geometry:
     *
     * qdata/scales move to the selected parent K window,
     * but padded_k stays at parent.k=25600.
     *
     * Therefore:
     *
     * row r code base =
     *   shifted_qdata + r * parent_k
     *
     * not:
     *   shifted_qdata + r * logical_k
     */
    const auto* codes =
        static_cast<const std::uint8_t*>(
            parent.qdata) +
        k_begin;

    const auto* scales =
        static_cast<const std::uint8_t*>(
            parent.scales) +
        static_cast<std::size_t>(
            k_begin / kGroup) *
            sizeof(std::uint16_t);

    detail::W8ContiguousFp32Output output{
        static_cast<float*>(
            accumulator.data),
        rows
    };

    const dim3 grid(
        static_cast<unsigned>(
            (rows + FastSchedule::BM - 1) /
            FastSchedule::BM),

        static_cast<unsigned>(
            (cols + FastSchedule::BN - 1) /
            FastSchedule::BN),

        1u);

    detail::w8_rowsplit_gemm_mma_kernel<
        FastSchedule,
        Full,
        Epilogue,
        detail::W8ContiguousFp32Output>
        <<<grid,
           FastSchedule::THREADS,
           0,
           stream>>>(
            static_cast<const __nv_bfloat16*>(
                x.data),
            codes,
            scales,
            output,
            rows,
            logical_k,
            cols,
            parent.padded_shape[1]);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void fp32_to_bf16_kernel(
    const float* __restrict__ source,
    __nv_bfloat16* __restrict__ destination,
    std::int64_t elements)
{
    const std::int64_t index =
        static_cast<std::int64_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (index < elements) {
        destination[index] =
            __float2bfloat16_rn(
                source[index]);
    }
}

} // namespace

void linear_w8_k_slice_accumulate(
    const Tensor& x,
    const Weight& parent,
    std::int32_t k_begin,
    std::int32_t k_extent,
    Tensor& accumulator,
    bool initialize,
    cudaStream_t stream)
{
    validate_weight(parent);

    if (x.dtype != DType::BF16 ||
        accumulator.dtype != DType::FP32)
    {
        throw std::invalid_argument(
            "linear_w8_k_slice: expected BF16 input "
            "and FP32 accumulator");
    }

    if (x.ne[0] != k_extent ||
        accumulator.ne[0] != parent.n ||
        accumulator.ne[1] != x.ne[1])
    {
        throw std::invalid_argument(
            "linear_w8_k_slice: shape mismatch");
    }

    if (k_begin < 0 ||
        k_extent <= 0 ||
        (k_begin % kGroup) != 0 ||
        (k_extent % kGroup) != 0 ||
        k_begin + k_extent > parent.k)
    {
        throw std::invalid_argument(
            "linear_w8_k_slice: invalid K window");
    }

    if ((k_extent % FastSchedule::BK) != 0)
    {
        throw std::invalid_argument(
            "linear_w8_k_slice: K extent must align "
            "to MMA BK");
    }

    const bool full =
        (parent.n % FastSchedule::BM) == 0 &&
        (x.ne[1] % FastSchedule::BN) == 0;

    if (initialize) {
        if (full) {
            launch_fast<
                true,
                detail::W8Epilogue::StoreFp32>(
                    x,
                    parent,
                    k_begin,
                    accumulator,
                    stream);
        } else {
            launch_fast<
                false,
                detail::W8Epilogue::StoreFp32>(
                    x,
                    parent,
                    k_begin,
                    accumulator,
                    stream);
        }
    } else {
        if (full) {
            launch_fast<
                true,
                detail::W8Epilogue::AddFp32>(
                    x,
                    parent,
                    k_begin,
                    accumulator,
                    stream);
        } else {
            launch_fast<
                false,
                detail::W8Epilogue::AddFp32>(
                    x,
                    parent,
                    k_begin,
                    accumulator,
                    stream);
        }
    }
}

void linear_w8_fp32_materialize(
    const Tensor& accumulator,
    Tensor& output,
    cudaStream_t stream)
{
    if (accumulator.dtype != DType::FP32 ||
        output.dtype != DType::BF16 ||
        accumulator.ne[0] != output.ne[0] ||
        accumulator.ne[1] != output.ne[1])
    {
        throw std::invalid_argument(
            "linear_w8_fp32_materialize: "
            "shape/dtype mismatch");
    }

    const std::int64_t elements =
        static_cast<std::int64_t>(
            accumulator.ne[0]) *
        accumulator.ne[1];

    constexpr std::int32_t threads = 256;

    const std::int32_t blocks =
        static_cast<std::int32_t>(
            (elements + threads - 1) /
            threads);

    fp32_to_bf16_kernel
        <<<blocks,
           threads,
           0,
           stream>>>(
            static_cast<const float*>(
                accumulator.data),
            static_cast<__nv_bfloat16*>(
                output.data),
            elements);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
