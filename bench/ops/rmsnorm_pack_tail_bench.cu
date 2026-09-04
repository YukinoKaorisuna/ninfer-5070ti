// Public hot-cache benchmark for the exact fused tail-pack RMSNorm contract.
#include "ninfer/ops/rmsnorm_pack_tail.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(std::int32_t batch) {
    constexpr std::int32_t kRows       = 5120;
    constexpr std::int32_t kInputWidth = 8;
    constexpr std::int32_t kTailWidth  = 7;
    const std::size_t input_elements   = static_cast<std::size_t>(kRows) * kInputWidth * batch;
    const std::size_t output_elements  = static_cast<std::size_t>(kRows) * kTailWidth * batch;
    DeviceBuffer input                 = make_bf16(input_elements);
    DeviceBuffer weight                = make_bf16(kRows);
    DeviceBuffer output                = make_zeros(output_elements * sizeof(std::uint16_t));
    Tensor input_tensor(input.p, DType::BF16, {kRows, kInputWidth, batch});
    Tensor weight_tensor(weight.p, DType::BF16, {kRows});
    Tensor output_tensor(output.p, DType::BF16, {kRows, kTailWidth * batch});

    // Anchor rows are not read. As in the general RMSNorm benchmark, the shared weight is reused
    // across rows and excluded from effective payload bandwidth.
    const double payload_bytes = 2.0 * static_cast<double>(output_elements) * sizeof(std::uint16_t);
    const Result result        = bench_loop(
        [&](cudaStream_t stream) {
            ops::rmsnorm_pack_tail(input_tensor, weight_tensor, output_tensor, stream);
        },
        payload_bytes);
    char label[96];
    std::snprintf(label, sizeof(label), "rmsnorm_pack_tail B=%d rows=%d", batch,
                  kTailWidth * batch);
    print_result(label, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    if (argc == 1) {
        for (std::int32_t batch = 1; batch <= 8; ++batch) { run(batch); }
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s [B]\n", argv[0]);
        return 2;
    }
    const int batch = std::atoi(argv[1]);
    if (batch < 1 || batch > 8) {
        std::fprintf(stderr, "B must be in 1..8\n");
        return 2;
    }
    run(batch);
    return 0;
}
