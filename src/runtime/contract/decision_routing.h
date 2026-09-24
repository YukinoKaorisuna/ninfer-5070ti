#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

struct DecisionRoutingFinal {
    std::vector<float> routing_probabilities;
    std::int32_t winner_index = -1;
};

inline void apply_decision_routing_branch(
    std::vector<double>& candidate_products,
    const std::vector<float>& branch_probabilities,
    const std::vector<std::vector<std::uint32_t>>&
        descendant_candidate_indices) {

    if (branch_probabilities.size() !=
        descendant_candidate_indices.size()) {

        throw std::logic_error(
            "decision routing branch/mapping sizes disagree");
    }

    for (std::size_t edge = 0;
         edge < branch_probabilities.size();
         ++edge) {

        const double probability =
            static_cast<double>(
                branch_probabilities[edge]);

        if (!std::isfinite(probability) ||
            probability < 0.0 ||
            probability > 1.0) {

            throw std::logic_error(
                "decision routing branch probability is invalid");
        }

        for (const std::uint32_t candidate :
             descendant_candidate_indices[edge]) {

            if (candidate >=
                candidate_products.size()) {

                throw std::logic_error(
                    "decision routing descendant index is out of range");
            }

            candidate_products[candidate] *=
                probability;
        }
    }
}

inline DecisionRoutingFinal finalize_decision_routing(
    const std::vector<double>& candidate_products) {

    if (candidate_products.empty()) {
        throw std::logic_error(
            "decision routing candidate domain is empty");
    }

    double probability_sum = 0.0;

    for (const double probability :
         candidate_products) {

        if (!std::isfinite(probability) ||
            probability < 0.0) {

            throw std::logic_error(
                "decision routing candidate product is invalid");
        }

        probability_sum += probability;
    }

    if (!std::isfinite(probability_sum) ||
        probability_sum <= 0.0) {

        throw std::logic_error(
            "decision routing candidate product sum is invalid");
    }

    DecisionRoutingFinal result;

    result.routing_probabilities.reserve(
        candidate_products.size());

    std::size_t winner_index = 0;

    for (std::size_t candidate = 0;
         candidate < candidate_products.size();
         ++candidate) {

        result.routing_probabilities.push_back(
            static_cast<float>(
                candidate_products[candidate] /
                probability_sum));

        // Strict greater-than preserves lowest semantic index on an exact tie.
        if (candidate != 0 &&
            candidate_products[candidate] >
                candidate_products[winner_index]) {

            winner_index = candidate;
        }
    }

    result.winner_index =
        static_cast<std::int32_t>(
            winner_index);

    return result;
}

} // namespace ninfer::runtime
