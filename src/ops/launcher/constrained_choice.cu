#include "ops/launcher/constrained_choice.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarpSize      = 32;
constexpr int kMaxCandidates = 16;

__global__ void constrained_choice_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ candidate_ids,
    float* __restrict__ probabilities,
    std::int32_t* __restrict__ winners,
    std::int32_t physical_rows,
    std::int32_t valid_rows,
    std::int32_t candidates,
    std::int32_t batch) {

    const int b    = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    if (b >= batch || lane >= kWarpSize) {
        return;
    }

    __shared__ float scores[kMaxCandidates];
    __shared__ float denom;
    __shared__ float max_score;
    __shared__ std::int32_t winner;

    if (lane < candidates) {
        const std::size_t candidate_offset =
            static_cast<std::size_t>(b) * candidates + lane;

        const std::int32_t token_id = candidate_ids[candidate_offset];

        // The public contract requires IDs to be within valid_rows.
        // Keep the kernel memory-safe even if an internal caller violates it.
        const float score =
            (token_id >= 0 && token_id < valid_rows)
                ? __bfloat162float(
                      logits[static_cast<std::size_t>(b) * physical_rows +
                             static_cast<std::size_t>(token_id)])
                : -INFINITY;

        scores[lane] = score;
    }

    __syncwarp();

    if (lane == 0) {
        float local_max = scores[0];
        std::int32_t local_winner = 0;

        for (int k = 1; k < candidates; ++k) {
            const float value = scores[k];

            // Strict comparison preserves lowest candidate index on ties.
            if (value > local_max) {
                local_max    = value;
                local_winner = k;
            }
        }

        float local_denom = 0.0f;

        for (int k = 0; k < candidates; ++k) {
            local_denom += expf(scores[k] - local_max);
        }

        max_score = local_max;
        denom     = local_denom;
        winner    = local_winner;
    }

    __syncwarp();

    if (lane < candidates) {
        const std::size_t output_offset =
            static_cast<std::size_t>(b) * candidates + lane;

        probabilities[output_offset] =
            expf(scores[lane] - max_score) / denom;
    }

    if (lane == 0) {
        winners[b] = winner;
    }
}

} // namespace

void constrained_choice_launch(const Tensor& logits,
                               const Tensor& candidate_ids,
                               Tensor& probabilities,
                               Tensor& winners,
                               std::int32_t valid_rows,
                               cudaStream_t stream) {
    const std::int32_t candidates = candidate_ids.ne[0];
    const std::int32_t batch      = candidate_ids.ne[1];

    constrained_choice_kernel<<<
        static_cast<unsigned int>(batch),
        kWarpSize,
        0,
        stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data),
            static_cast<const std::int32_t*>(candidate_ids.data),
            static_cast<float*>(probabilities.data),
            static_cast<std::int32_t*>(winners.data),
            logits.ne[0],
            valid_rows,
            candidates,
            batch);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
