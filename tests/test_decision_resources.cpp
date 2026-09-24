#include "runtime/contract/decision_resources.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace ninfer::runtime;

    const std::size_t required =
        decision_scorer_workspace_required_bytes(
            257);

    const auto exact =
        project_decision_scorer_workspace(
            257,
            required);

    if (!exact.fits ||
        exact.required_bytes != required ||
        exact.maximum_candidates < 257) {

        std::cerr
            << "FAIL: exact boundary rejected\n";

        return 1;
    }

    if (required == 0) {
        return 1;
    }

    const auto short_by_one =
        project_decision_scorer_workspace(
            257,
            required - 1);

    if (short_by_one.fits) {
        std::cerr
            << "FAIL: one-byte-short workspace accepted\n";

        return 1;
    }

    bool representation_rejected = false;

    try {
        (void)
            decision_scorer_workspace_required_bytes(
                static_cast<std::size_t>(
                    std::numeric_limits<
                        std::int32_t>::max()) +
                1U);

    } catch (const std::length_error&) {
        representation_rejected = true;
    }

    if (!representation_rejected) {
        std::cerr
            << "FAIL: int32 representation overflow accepted\n";

        return 1;
    }

    const std::size_t maximum =
        decision_scorer_max_candidates(
            required);

    if (maximum < 257) {
        std::cerr
            << "FAIL: maximum-K projection below exact boundary\n";

        return 1;
    }

    if (maximum >=
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {

        std::cerr
            << "FAIL: finite resource-boundary fixture unexpectedly reached int32 maximum\n";

        return 1;
    }

    const auto maximum_projection =
        project_decision_scorer_workspace(
            maximum,
            required);

    if (!maximum_projection.fits ||
        maximum_projection.maximum_candidates !=
            maximum ||
        maximum_projection.required_bytes >
            required) {

        std::cerr
            << "FAIL: computed maximum K does not fit its workspace\n";

        return 1;
    }

    const auto maximum_plus_one_projection =
        project_decision_scorer_workspace(
            maximum + 1U,
            required);

    if (maximum_plus_one_projection.fits ||
        maximum_plus_one_projection.maximum_candidates !=
            maximum ||
        maximum_plus_one_projection.required_bytes <=
            required) {

        std::cerr
            << "FAIL: computed maximum K+1 was not rejected\n";

        return 1;
    }

    std::cout
        << "K257_REQUIRED_BYTES="
        << required
        << "\n"
        << "K257_EXACT_BOUNDARY=PASS\n"
        << "K257_MINUS_ONE_REJECTION=PASS\n"
        << "INT32_REPRESENTATION_REJECTION=PASS\n"
        << "WORKSPACE_MAXIMUM_K="
        << maximum
        << "\n"
        << "WORKSPACE_MAX_K_BOUNDARY=PASS\n"
        << "WORKSPACE_MAX_K_PLUS_ONE_REJECTION=PASS\n"
        << "DECISION_RESOURCE_CONTRACT=PASS\n";

    return 0;
}
