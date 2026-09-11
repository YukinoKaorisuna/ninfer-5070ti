#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_w8_k_slice.h"

#include "core/tensor.h"
#include "ops/linear/linear_test_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::Tensor;
using ninfer::Weight;

void cuda_check(
    cudaError_t result,
    const char* label)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(label) +
            ": " +
            cudaGetErrorString(result));
    }
}

struct DeviceBuffer {
    void* p = nullptr;
    std::size_t bytes = 0;

    explicit DeviceBuffer(
        std::size_t size)
        : bytes(size)
    {
        cuda_check(
            cudaMalloc(&p, bytes),
            "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    DeviceBuffer(
        const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(
        const DeviceBuffer&) = delete;
};

std::uint16_t bf16_bits(
    __nv_bfloat16 value)
{
    std::uint16_t result = 0;

    std::memcpy(
        &result,
        &value,
        sizeof(result));

    return result;
}

void run_case(int T)
{
    constexpr int N      = 5120;
    constexpr int K      = 25600;
    constexpr int SliceK = 5120;
    constexpr int Slices = 5;

    auto host_weight =
        ninfer::test::linear::
            make_w8g32_f16s_weight(
                N,
                K,
                991U);

    DeviceBuffer device_weight(
        host_weight.payload.size());

    cuda_check(
        cudaMemcpy(
            device_weight.p,
            host_weight.payload.data(),
            host_weight.payload.size(),
            cudaMemcpyHostToDevice),
        "copy W8 weight");

    const Weight weight =
        host_weight.device_weight(
            device_weight.p);

    std::vector<__nv_bfloat16> host_x(
        static_cast<std::size_t>(K) *
        T);

    std::mt19937 rng(
        1234U +
        static_cast<unsigned>(T));

    std::uniform_real_distribution<float>
        dist(-0.75F, 0.75F);

    for (auto& value : host_x) {
        value =
            __float2bfloat16_rn(
                dist(rng));
    }

    DeviceBuffer x_storage(
        host_x.size() *
        sizeof(__nv_bfloat16));

    cuda_check(
        cudaMemcpy(
            x_storage.p,
            host_x.data(),
            x_storage.bytes,
            cudaMemcpyHostToDevice),
        "copy activation");

    Tensor x(
        x_storage.p,
        DType::BF16,
        {K, T});

    /*
     * Production full [5120,25600] W8 path.
     */
    DeviceBuffer oracle_storage(
        static_cast<std::size_t>(N) *
        T *
        sizeof(__nv_bfloat16));

    Tensor oracle(
        oracle_storage.p,
        DType::BF16,
        {N, T});

    ninfer::ops::linear(
        x,
        weight,
        oracle,
        nullptr);

    /*
     * Partial-K FP32 accumulator.
     */
    DeviceBuffer accumulator_storage(
        static_cast<std::size_t>(N) *
        T *
        sizeof(float));

    Tensor accumulator(
        accumulator_storage.p,
        DType::FP32,
        {N, T});

    DeviceBuffer slice_storage(
        static_cast<std::size_t>(SliceK) *
        T *
        sizeof(__nv_bfloat16));

    Tensor slice_x(
        slice_storage.p,
        DType::BF16,
        {SliceK, T});

    for (int slice = 0;
         slice < Slices;
         ++slice)
    {
        /*
         * Full x is [K,T], K contiguous.
         * Extract one [5120,T] vertical K range.
         */
        const auto* source =
            static_cast<const std::byte*>(
                x.data) +
            static_cast<std::size_t>(
                slice * SliceK) *
                sizeof(__nv_bfloat16);

        cuda_check(
            cudaMemcpy2D(
                slice_x.data,
                static_cast<std::size_t>(
                    SliceK) *
                    sizeof(__nv_bfloat16),

                source,
                static_cast<std::size_t>(
                    K) *
                    sizeof(__nv_bfloat16),

                static_cast<std::size_t>(
                    SliceK) *
                    sizeof(__nv_bfloat16),

                static_cast<std::size_t>(T),

                cudaMemcpyDeviceToDevice),
            "copy K slice");

        ninfer::ops::
            linear_w8_k_slice_accumulate(
                slice_x,
                weight,
                slice * SliceK,
                SliceK,
                accumulator,
                slice == 0,
                nullptr);
    }

    DeviceBuffer result_storage(
        static_cast<std::size_t>(N) *
        T *
        sizeof(__nv_bfloat16));

    Tensor result(
        result_storage.p,
        DType::BF16,
        {N, T});

    ninfer::ops::
        linear_w8_fp32_materialize(
            accumulator,
            result,
            nullptr);

    cuda_check(
        cudaDeviceSynchronize(),
        "projection synchronize");

    std::vector<__nv_bfloat16> expected(
        static_cast<std::size_t>(N) *
        T);

    std::vector<__nv_bfloat16> actual(
        static_cast<std::size_t>(N) *
        T);

    cuda_check(
        cudaMemcpy(
            expected.data(),
            oracle.data,
            expected.size() *
                sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost),
        "copy full projection");

    cuda_check(
        cudaMemcpy(
            actual.data(),
            result.data,
            actual.size() *
                sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost),
        "copy partial projection");

    float max_abs = 0.0F;
    double sum_abs = 0.0;
    std::size_t mismatch = 0;

    double sum_sq_error = 0.0;
    double sum_sq_reference = 0.0;

    for (std::size_t index = 0;
         index < expected.size();
         ++index)
    {
        const float a =
            __bfloat162float(
                expected[index]);

        const float b =
            __bfloat162float(
                actual[index]);

        if (!std::isfinite(a) ||
            !std::isfinite(b))
        {
            throw std::runtime_error(
                "non-finite output");
        }

        const double error =
            static_cast<double>(b) -
            static_cast<double>(a);

        const float abs_error =
            std::abs(a - b);

        max_abs =
            std::max(
                max_abs,
                abs_error);

        sum_abs += abs_error;

        sum_sq_error +=
            error * error;

        sum_sq_reference +=
            static_cast<double>(a) *
            static_cast<double>(a);

        if (bf16_bits(expected[index]) !=
            bf16_bits(actual[index]))
        {
            ++mismatch;
        }
    }

    const double mean_abs =
        sum_abs /
        static_cast<double>(
            expected.size());

    const double relative_l2 =
        std::sqrt(
            sum_sq_error /
            std::max(
                sum_sq_reference,
                1.0e-30));

    const double mismatch_pct =
        100.0 *
        static_cast<double>(mismatch) /
        static_cast<double>(
            expected.size());

    std::cout
        << "T=" << T
        << " elements=" << expected.size()
        << " mismatch=" << mismatch
        << " mismatch_pct=" << mismatch_pct
        << " max_abs=" << max_abs
        << " mean_abs=" << mean_abs
        << " relative_l2=" << relative_l2
        << '\n';

    /*
     * Ninfer's A16 relative-L2 criterion is 1/256.
     * Use that same magnitude for Phase-1.
     */
    constexpr double kA16Tolerance =
        1.0 / 256.0;

    if (relative_l2 >
        kA16Tolerance)
    {
        throw std::runtime_error(
            "partial-K relative-L2 exceeds A16 tolerance");
    }
}

} // namespace

int main()
{
    try {
        for (const int T :
             {1, 7, 8, 32, 64, 896})
        {
            run_case(T);
        }

        std::cout
            << "OK linear_w8_k_slice\n";

        return 0;
    } catch (
        const std::exception& e)
    {
        std::cerr
            << "FAIL: "
            << e.what()
            << '\n';

        return 1;
    }
}
