#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void candidate_selector_path_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, cudaStream_t stream);

} // namespace ninfer::ops::detail
