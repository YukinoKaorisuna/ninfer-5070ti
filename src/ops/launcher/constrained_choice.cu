#include "ops/launcher/constrained_choice.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarpSize          = 32;
constexpr int kDecisionBlockSize = 256;

__device__ bool better_candidate(
    float candidate_score,
    std::int32_t candidate_index,
    float current_score,
    std::int32_t current_index) {

    return candidate_score > current_score ||
           (candidate_score == current_score &&
            candidate_index < current_index);
}

__global__ void constrained_choice_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ candidate_ids,
    float* __restrict__ probabilities,
    std::int32_t* __restrict__ winners,
    std::int32_t physical_rows,
    std::int32_t valid_rows,
    std::int32_t candidates,
    std::int32_t batch) {

    const int b =
        static_cast<int>(blockIdx.x);

    const int lane =
        static_cast<int>(threadIdx.x);

    if (b >= batch) {
        return;
    }

    __shared__ float reduce_values[kDecisionBlockSize];
    __shared__ std::int32_t reduce_indices[kDecisionBlockSize];

    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ std::int32_t winner;

    float local_max = -INFINITY;

    // candidates is a convenient sentinel because every legal candidate
    // index lies in [0,candidates).
    std::int32_t local_winner =
        candidates;

    for (std::int32_t k = lane;
         k < candidates;
         k += static_cast<std::int32_t>(blockDim.x)) {

        const std::size_t candidate_offset =
            static_cast<std::size_t>(b) *
                static_cast<std::size_t>(candidates) +
            static_cast<std::size_t>(k);

        const std::int32_t token_id =
            candidate_ids[candidate_offset];

        // The public contract requires IDs in [0,valid_rows). Keep the
        // kernel memory-safe even if an internal caller violates it.
        const float score =
            token_id >= 0 &&
                    token_id < valid_rows
                ? __bfloat162float(
                      logits[
                          static_cast<std::size_t>(b) *
                              static_cast<std::size_t>(physical_rows) +
                          static_cast<std::size_t>(token_id)])
                : -INFINITY;

        if (better_candidate(
                score,
                k,
                local_max,
                local_winner)) {

            local_max = score;
            local_winner = k;
        }
    }

    reduce_values[lane] =
        local_max;

    reduce_indices[lane] =
        local_winner;

    __syncthreads();

    // Block-wide (score,lowest-index) max reduction.
    for (int stride =
             static_cast<int>(blockDim.x) / 2;
         stride > 0;
         stride >>= 1) {

        if (lane < stride) {
            const float other_score =
                reduce_values[lane + stride];

            const std::int32_t other_index =
                reduce_indices[lane + stride];

            if (better_candidate(
                    other_score,
                    other_index,
                    reduce_values[lane],
                    reduce_indices[lane])) {

                reduce_values[lane] =
                    other_score;

                reduce_indices[lane] =
                    other_index;
            }
        }

        __syncthreads();
    }

    if (lane == 0) {
        maximum =
            reduce_values[0];

        winner =
            reduce_indices[0];
    }

    __syncthreads();

    float local_sum = 0.0F;

    for (std::int32_t k = lane;
         k < candidates;
         k += static_cast<std::int32_t>(blockDim.x)) {

        const std::size_t candidate_offset =
            static_cast<std::size_t>(b) *
                static_cast<std::size_t>(candidates) +
            static_cast<std::size_t>(k);

        const std::int32_t token_id =
            candidate_ids[candidate_offset];

        const float score =
            token_id >= 0 &&
                    token_id < valid_rows
                ? __bfloat162float(
                      logits[
                          static_cast<std::size_t>(b) *
                              static_cast<std::size_t>(physical_rows) +
                          static_cast<std::size_t>(token_id)])
                : -INFINITY;

        local_sum +=
            expf(score - maximum);
    }

    reduce_values[lane] =
        local_sum;

    __syncthreads();

    for (int stride =
             static_cast<int>(blockDim.x) / 2;
         stride > 0;
         stride >>= 1) {

        if (lane < stride) {
            reduce_values[lane] +=
                reduce_values[lane + stride];
        }

        __syncthreads();
    }

    if (lane == 0) {
        denominator =
            reduce_values[0];
    }

    __syncthreads();

    for (std::int32_t k = lane;
         k < candidates;
         k += static_cast<std::int32_t>(blockDim.x)) {

        const std::size_t candidate_offset =
            static_cast<std::size_t>(b) *
                static_cast<std::size_t>(candidates) +
            static_cast<std::size_t>(k);

        const std::int32_t token_id =
            candidate_ids[candidate_offset];

        const float score =
            token_id >= 0 &&
                    token_id < valid_rows
                ? __bfloat162float(
                      logits[
                          static_cast<std::size_t>(b) *
                              static_cast<std::size_t>(physical_rows) +
                          static_cast<std::size_t>(token_id)])
                : -INFINITY;

        probabilities[candidate_offset] =
            expf(score - maximum) /
            denominator;
    }

    if (lane == 0) {
        winners[b] =
            winner;
    }
}

} // namespace

void constrained_choice_launch(
    const Tensor& logits,
    const Tensor& candidate_ids,
    Tensor& probabilities,
    Tensor& winners,
    std::int32_t valid_rows,
    cudaStream_t stream) {

    const std::int32_t candidates =
        candidate_ids.ne[0];

    const std::int32_t batch =
        candidate_ids.ne[1];

    // Preserve the original warp-sized fast path for small finite domains
    // while using a block-wide strided reduction for larger domains.
    const unsigned int threads =
        candidates <= kWarpSize
            ? static_cast<unsigned int>(kWarpSize)
            : static_cast<unsigned int>(kDecisionBlockSize);

    constrained_choice_kernel<<<
        static_cast<unsigned int>(batch),
        threads,
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

    CUDA_CHECK(
        cudaGetLastError());
}

} // namespace ninfer::ops::detail
