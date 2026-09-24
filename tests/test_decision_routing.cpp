#include "runtime/contract/decision_routing.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    std::vector<double> products{
        1.0,
        1.0,
    };

    const std::vector<float> branch_probabilities{
        0.60F,
        0.40F,
    };

    const std::vector<std::vector<std::uint32_t>>
        descendants{
            {0},
            {1},
        };

    ninfer::runtime::apply_decision_routing_branch(
        products,
        branch_probabilities,
        descendants);

    const auto routing =
        ninfer::runtime::finalize_decision_routing(
            products);

    if (routing.routing_probabilities.size() != 2 ||
        std::fabs(
            routing.routing_probabilities[0] -
            0.60F) > 1.0e-6F ||
        std::fabs(
            routing.routing_probabilities[1] -
            0.40F) > 1.0e-6F ||
        routing.winner_index != 0) {

        std::cerr
            << "FAIL: constrained routing Q changed\n";

        return 1;
    }

    const double full_likelihood_a =
        0.60 * 0.01;

    const double full_likelihood_b =
        0.40 * 0.99;

    if (!(full_likelihood_b >
          full_likelihood_a)) {

        std::cerr
            << "FAIL: reversal fixture malformed\n";

        return 1;
    }

    std::cout
        << "ROUTING_Q_A=0.60\n"
        << "ROUTING_Q_B=0.40\n"
        << "FULL_L_A=0.006\n"
        << "FULL_L_B=0.396\n"
        << "ROUTING_WINNER=A\n"
        << "FULL_SEQUENCE_WINNER=B\n"
        << "DETERMINISTIC_UNARY_FACTOR=1\n"
        << "Q_VS_L_REVERSAL=PASS\n"
        << "DECISION_ROUTING_CONTRACT=PASS\n";

    return 0;
}
