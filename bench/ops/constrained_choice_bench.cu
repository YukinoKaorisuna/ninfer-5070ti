#include "ninfer/ops/constrained_choice.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kValidRows    = 248077;
constexpr std::int32_t kMaximumBatch = 8;
constexpr std::int32_t kMaximumK     = 16;

void run_case(std::int32_t candidates, std::int32_t batch) {
    DeviceBuffer logits =
        make_bf16(
            static_cast<std::size_t>(kPhysicalRows) *
            static_cast<std::size_t>(batch));

    std::vector<std::int32_t> host_candidate_ids(
        static_cast<std::size_t>(candidates) *
        static_cast<std::size_t>(batch));

    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t k = 0; k < candidates; ++k) {
            host_candidate_ids[
                static_cast<std::size_t>(b) * candidates + k] =
                (17 + b * 7919 + k * 3571) % kValidRows;
        }
    }

    DeviceBuffer candidate_ids(
        host_candidate_ids.size() * sizeof(std::int32_t));

    candidate_ids.copy_from_host(
        host_candidate_ids.data(),
        host_candidate_ids.size() * sizeof(std::int32_t));

    DeviceBuffer probabilities(
        host_candidate_ids.size() * sizeof(float));

    DeviceBuffer winners(
        static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    Tensor logits_tensor(
        logits.p,
        DType::BF16,
        {kPhysicalRows, batch});

    Tensor candidate_tensor(
        candidate_ids.p,
        DType::I32,
        {candidates, batch});

    Tensor probability_tensor(
        probabilities.p,
        DType::FP32,
        {candidates, batch});

    Tensor winner_tensor(
        winners.p,
        DType::I32,
        {batch});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::constrained_choice(
                logits_tensor,
                candidate_tensor,
                probability_tensor,
                winner_tensor,
                kValidRows,
                stream);
        },
        0.0,
        50,
        200,
        500);

    std::printf(
        "constrained_choice K=%-2d B=%-2d "
        "median=%8.3f us  min=%8.3f us  "
        "p95=%8.3f us  mean=%8.3f us  "
        "inner=%d runs=%d\n",
        candidates,
        batch,
        result.median_us,
        result.min_us,
        result.p95_us,
        result.mean_us,
        result.inner_iters,
        result.n_runs);
}

} // namespace

int main() {
    int count = 0;

    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    for (const std::int32_t candidates : {2, 4, 8, 16}) {
        run_case(candidates, 1);
    }

    for (const std::int32_t candidates : {2, 4, 8, 16}) {
        run_case(candidates, 4);
    }

    for (const std::int32_t candidates : {2, 4, 8, 16}) {
        run_case(candidates, 8);
    }

    return 0;
}
