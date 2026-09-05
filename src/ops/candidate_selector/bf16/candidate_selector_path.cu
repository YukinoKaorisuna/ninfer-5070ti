#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kSteps        = 7;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kBlockThreads = 512;

struct CandidateSelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    std::int32_t predecessor;
    std::int32_t base_position;
    float temperature;
    unsigned long long seed;
};

__global__ __launch_bounds__(kBlockThreads, 1) void candidate_selector_path_kernel(
    const std::int32_t* __restrict__ candidate_ids, const float* __restrict__ unary_scores,
    const __nv_bfloat16* __restrict__ projected_hidden, const std::int32_t* __restrict__ anchors,
    const __nv_bfloat16* __restrict__ predecessor_codebook,
    const __nv_bfloat16* __restrict__ successor_codebook,
    const std::int32_t* __restrict__ base_positions, const SamplingConfig* __restrict__ configs,
    std::int32_t* __restrict__ drafts, float* __restrict__ proposal_q) {
    __shared__ __align__(16) CandidateSelectorShared shared;
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.x);
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;

    if (tid == 0) {
        shared.predecessor   = anchors[batch];
        shared.base_position = base_positions[batch];
        shared.temperature   = configs[batch].temperature;
        shared.seed          = configs[batch].seed;
    }

#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
        const std::int32_t column         = batch * kSteps + step;
        const std::int32_t candidate_base = column * kCandidates;
        std::int32_t successor            = lane == 0 ? candidate_ids[candidate_base + warp] : 0;
        successor                         = __shfl_sync(kFullWarpMask, successor, 0);
        cp_async<16, Cache::cg>(
            &shared.successors[warp * kRank + lane * 8],
            &successor_codebook[static_cast<std::int64_t>(successor) * kRank + lane * 8]);
        cp_commit();
        // Successor staging does not consume predecessor. On later steps it may begin while warp 0
        // finishes the prior draw; this barrier publishes that draw before product reads it.
        __syncthreads();

        if (tid < kRank) {
            const float predecessor = __bfloat162float(
                predecessor_codebook[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]);
            const float hidden =
                __bfloat162float(projected_hidden[static_cast<std::int64_t>(column) * kRank + tid]);
            shared.product[tid] = predecessor * hidden;
        }
        cp_wait<0>();
        __syncthreads();

        float sum = 0.0F;
#pragma unroll
        for (int rank = lane; rank < kRank; rank += 32) {
            sum = fmaf(shared.product[rank],
                       __bfloat162float(shared.successors[warp * kRank + rank]), sum);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) { shared.edge[warp] = unary_scores[candidate_base + warp] + sum; }
        // This both publishes every edge and ends all reads of the current successor tile.
        __syncthreads();

        if (warp == 0) {
            const bool stochastic = shared.temperature > 0.0F;
            const float edge      = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const float maximum   = warp_max(edge);
            int selected          = 0;
            if (stochastic) {
                const float weight =
                    lane < kCandidates ? __expf((edge - maximum) / shared.temperature) : 0.0F;
                const float probability = weight / warp_sum(weight);
                if (lane < kCandidates) { proposal_q[candidate_base + lane] = probability; }

                const float uniform =
                    lane == 0 ? sampling_uniform(shared.seed, shared.base_position + step,
                                                 kSamplePurposeDFlash2Proposal, 0U)
                              : 0.0F;
                float cumulative = 0.0F;
                bool found       = false;
                selected         = kCandidates - 1;
#pragma unroll
                for (int candidate = 0; candidate < kCandidates; ++candidate) {
                    const float value = __shfl_sync(kFullWarpMask, probability, candidate);
                    if (lane == 0 && !found) {
                        cumulative += value;
                        if (uniform < cumulative) {
                            selected = candidate;
                            found    = true;
                        }
                    }
                }
            } else {
                const unsigned winners =
                    __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
                selected = winners == 0U ? 0 : __ffs(winners) - 1;
                if (lane < kCandidates) {
                    proposal_q[candidate_base + lane] = lane == selected ? 1.0F : 0.0F;
                }
            }

            if (lane == 0) {
                const std::int32_t selected_token = candidate_ids[candidate_base + selected];
                drafts[column]                    = selected_token;
                shared.predecessor                = selected_token;
            }
        }
    }
}

} // namespace

void candidate_selector_path_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, cudaStream_t stream) {
    candidate_selector_path_kernel<<<candidate_ids.ne[2], kBlockThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(unary_scores.data),
        static_cast<const __nv_bfloat16*>(projected_hidden.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
        static_cast<const __nv_bfloat16*>(successor_codebook.data),
        static_cast<const std::int32_t*>(base_positions.data), configs,
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
