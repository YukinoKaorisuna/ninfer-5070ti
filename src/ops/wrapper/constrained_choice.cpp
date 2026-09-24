#include "ninfer/ops/constrained_choice.h"

#include "ops/launcher/constrained_choice.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void constrained_choice(const Tensor& logits,
                        const Tensor& candidate_ids,
                        Tensor& probabilities,
                        Tensor& winners,
                        std::int32_t valid_rows,
                        cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument(
            "constrained_choice: logits must be BF16");
    }

    if (candidate_ids.dtype != DType::I32) {
        throw std::invalid_argument(
            "constrained_choice: candidate_ids must be I32");
    }

    if (probabilities.dtype != DType::FP32) {
        throw std::invalid_argument(
            "constrained_choice: probabilities must be FP32");
    }

    if (winners.dtype != DType::I32) {
        throw std::invalid_argument(
            "constrained_choice: winners must be I32");
    }

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "constrained_choice: logits must be rank-2 [physical_vocab,B]");
    }

    if (candidate_ids.ne[2] != 1 || candidate_ids.ne[3] != 1) {
        throw std::invalid_argument(
            "constrained_choice: candidate_ids must be rank-2 [K,B]");
    }

    if (probabilities.ne[2] != 1 || probabilities.ne[3] != 1) {
        throw std::invalid_argument(
            "constrained_choice: probabilities must be rank-2 [K,B]");
    }

    if (winners.ne[1] != 1 ||
        winners.ne[2] != 1 ||
        winners.ne[3] != 1) {
        throw std::invalid_argument(
            "constrained_choice: winners must be rank-1 [B]");
    }

    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t batch         = logits.ne[1];
    const std::int32_t candidates    = candidate_ids.ne[0];

    if (physical_rows <= 0) {
        throw std::invalid_argument(
            "constrained_choice: physical vocabulary must be positive");
    }

    if (valid_rows <= 0 || valid_rows > physical_rows) {
        throw std::invalid_argument(
            "constrained_choice: valid_rows must be in [1, physical_vocab]");
    }

    if (batch < 1 || batch > 8) {
        throw std::invalid_argument(
            "constrained_choice: B must be in [1,8]");
    }

    if (candidates < 2) {
        throw std::invalid_argument(
            "constrained_choice: K must be at least 2");
    }

    if (candidate_ids.ne[1] != batch) {
        throw std::invalid_argument(
            "constrained_choice: candidate_ids B must match logits B");
    }

    if (probabilities.ne[0] != candidates ||
        probabilities.ne[1] != batch) {
        throw std::invalid_argument(
            "constrained_choice: probabilities must have shape [K,B]");
    }

    if (winners.ne[0] != batch) {
        throw std::invalid_argument(
            "constrained_choice: winners must have shape [B]");
    }

    if (!logits.is_contiguous() ||
        !candidate_ids.is_contiguous() ||
        !probabilities.is_contiguous() ||
        !winners.is_contiguous()) {
        throw std::invalid_argument(
            "constrained_choice: all tensors must be contiguous");
    }

    if (logits.data == nullptr ||
        candidate_ids.data == nullptr ||
        probabilities.data == nullptr ||
        winners.data == nullptr) {
        throw std::invalid_argument(
            "constrained_choice: all tensor data must be non-null");
    }

    if (probabilities.data == logits.data ||
        probabilities.data == candidate_ids.data ||
        winners.data == logits.data ||
        winners.data == candidate_ids.data ||
        winners.data == probabilities.data) {
        throw std::invalid_argument(
            "constrained_choice: outputs must not alias inputs or each other");
    }

    detail::constrained_choice_launch(
        logits,
        candidate_ids,
        probabilities,
        winners,
        valid_rows,
        stream);
}

} // namespace ninfer::ops
