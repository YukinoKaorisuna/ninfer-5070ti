#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Score a finite set of allowed next-token IDs from full-vocabulary logits.
 *
 * logits:
 *   contiguous BF16 [physical_vocab, B]
 *
 * candidate_ids:
 *   contiguous I32 [K, B]
 *
 * probabilities:
 *   contiguous FP32 [K, B]
 *
 * winners:
 *   contiguous I32 [B]
 *
 * For each batch column, only the supplied candidate token logits participate
 * in the softmax. Equal maxima select the lowest candidate index.
 *
 * Finite-choice domain:
 *   1 <= B <= 8
 *   2 <= K
 *   K is represented by the Tensor's int32 extent and is otherwise limited
 *   by caller/runtime resources rather than an arbitrary product ceiling.
 *   every candidate id is in [0, valid_rows)
 */
void constrained_choice(const Tensor& logits,
                        const Tensor& candidate_ids,
                        Tensor& probabilities,
                        Tensor& winners,
                        std::int32_t valid_rows,
                        cudaStream_t stream);

} // namespace ninfer::ops
