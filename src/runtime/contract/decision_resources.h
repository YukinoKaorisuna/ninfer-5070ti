#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {

inline constexpr std::size_t
    kDecisionScorerAlignment = 256;

inline std::size_t checked_decision_add(
    std::size_t lhs,
    std::size_t rhs) {

    if (rhs >
        std::numeric_limits<std::size_t>::max() -
            lhs) {

        throw std::overflow_error(
            "decision scorer workspace addition overflow");
    }

    return lhs + rhs;
}

inline std::size_t checked_decision_multiply(
    std::size_t lhs,
    std::size_t rhs) {

    if (lhs != 0 &&
        rhs >
            std::numeric_limits<std::size_t>::max() /
                lhs) {

        throw std::overflow_error(
            "decision scorer workspace multiplication overflow");
    }

    return lhs * rhs;
}

inline std::size_t align_decision_workspace(
    std::size_t value) {

    constexpr std::size_t alignment =
        kDecisionScorerAlignment;

    constexpr std::size_t mask =
        alignment - 1;

    return checked_decision_add(
               value,
               mask) &
           ~mask;
}

inline std::size_t
decision_scorer_workspace_required_bytes(
    std::size_t candidate_count) {

    if (candidate_count < 2) {
        throw std::invalid_argument(
            "decision scorer requires at least two candidates");
    }

    if (candidate_count >
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {

        throw std::length_error(
            "decision scorer candidate count exceeds int32 tensor extent");
    }

    const std::size_t ids_bytes =
        checked_decision_multiply(
            candidate_count,
            sizeof(std::int32_t));

    const std::size_t probabilities_bytes =
        checked_decision_multiply(
            candidate_count,
            sizeof(float));

    std::size_t offset = 0;

    // candidate_ids: I32[K,1], default DeviceArena alignment 256.
    offset =
        align_decision_workspace(
            offset);

    offset =
        checked_decision_add(
            offset,
            ids_bytes);

    // probabilities: FP32[K,1].
    offset =
        align_decision_workspace(
            offset);

    offset =
        checked_decision_add(
            offset,
            probabilities_bytes);

    // winner: I32[1].
    offset =
        align_decision_workspace(
            offset);

    offset =
        checked_decision_add(
            offset,
            sizeof(std::int32_t));

    return offset;
}

inline std::size_t
decision_scorer_max_candidates(
    std::size_t workspace_capacity_bytes) {

    if (decision_scorer_workspace_required_bytes(2) >
        workspace_capacity_bytes) {

        return 0;
    }

    std::size_t low = 2;

    std::size_t high =
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max());

    std::size_t best = 0;

    while (low <= high) {
        const std::size_t middle =
            low + (high - low) / 2;

        const std::size_t required =
            decision_scorer_workspace_required_bytes(
                middle);

        if (required <=
            workspace_capacity_bytes) {

            best = middle;

            if (middle == high) {
                break;
            }

            low = middle + 1;

        } else {
            high = middle - 1;
        }
    }

    return best;
}

struct DecisionScorerWorkspaceProjection {
    std::size_t requested_candidates = 0;
    std::size_t required_bytes = 0;
    std::size_t available_bytes = 0;
    std::size_t maximum_candidates = 0;
    bool fits = false;
};

inline DecisionScorerWorkspaceProjection
project_decision_scorer_workspace(
    std::size_t candidate_count,
    std::size_t workspace_capacity_bytes) {

    DecisionScorerWorkspaceProjection out;

    out.requested_candidates =
        candidate_count;

    out.required_bytes =
        decision_scorer_workspace_required_bytes(
            candidate_count);

    out.available_bytes =
        workspace_capacity_bytes;

    out.maximum_candidates =
        decision_scorer_max_candidates(
            workspace_capacity_bytes);

    out.fits =
        out.required_bytes <=
            workspace_capacity_bytes &&
        candidate_count <=
            out.maximum_candidates;

    return out;
}

} // namespace ninfer::runtime
