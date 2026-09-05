#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void require_batch_size(std::int32_t batch_size) {
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("candidate_selector_path: B must be in [1,8]");
    }
}

} // namespace

const char* candidate_selector_path_route_name(std::int32_t batch_size) {
    require_batch_size(batch_size);
    return "candidate_selector.bf16.staged.t512";
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, cudaStream_t stream) {
    candidate_selector_path_launch(candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, stream);
}

} // namespace ninfer::ops::detail
