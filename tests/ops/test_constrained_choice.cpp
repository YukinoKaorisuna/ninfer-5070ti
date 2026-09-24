#include "ninfer/ops/constrained_choice.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct OracleResult {
    std::vector<float> probabilities;
    std::vector<std::int32_t> winners;
};

OracleResult oracle(const std::vector<std::uint16_t>& logits,
                    std::int32_t physical_rows,
                    const std::vector<std::int32_t>& candidate_ids,
                    std::int32_t candidates,
                    std::int32_t batch) {
    OracleResult out;
    out.probabilities.resize(
        static_cast<std::size_t>(candidates) * batch);
    out.winners.resize(static_cast<std::size_t>(batch));

    for (std::int32_t b = 0; b < batch; ++b) {
        const std::size_t candidate_base =
            static_cast<std::size_t>(b) * candidates;

        const std::size_t logits_base =
            static_cast<std::size_t>(b) * physical_rows;

        float maximum = -INFINITY;
        std::int32_t winner = 0;

        for (std::int32_t k = 0; k < candidates; ++k) {
            const std::int32_t token =
                candidate_ids[candidate_base + k];

            const float score =
                bf16_to_f32(logits[
                    logits_base +
                    static_cast<std::size_t>(token)]);

            if (k == 0 || score > maximum) {
                maximum = score;
                winner  = k;
            }
        }

        float denominator = 0.0f;

        for (std::int32_t k = 0; k < candidates; ++k) {
            const std::int32_t token =
                candidate_ids[candidate_base + k];

            const float score =
                bf16_to_f32(logits[
                    logits_base +
                    static_cast<std::size_t>(token)]);

            denominator += std::exp(score - maximum);
        }

        for (std::int32_t k = 0; k < candidates; ++k) {
            const std::int32_t token =
                candidate_ids[candidate_base + k];

            const float score =
                bf16_to_f32(logits[
                    logits_base +
                    static_cast<std::size_t>(token)]);

            out.probabilities[candidate_base + k] =
                std::exp(score - maximum) / denominator;
        }

        out.winners[static_cast<std::size_t>(b)] = winner;
    }

    return out;
}

int verify_probabilities(const char* label,
                         const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float tolerance = 2.0e-6f) {
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL " << label
                  << ": size mismatch\n";
        return 1;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error =
            std::fabs(actual[i] - expected[i]);

        if (!(error <= tolerance)) {
            std::cerr
                << "FAIL " << label
                << " index=" << i
                << " actual=" << actual[i]
                << " expected=" << expected[i]
                << " error=" << error
                << "\n";
            return 1;
        }
    }

    return 0;
}

int run_case(std::int32_t candidates,
             std::int32_t batch,
             bool balanced_fixture = false) {
    const std::int32_t physical_rows =
        std::max<std::int32_t>(
            64,
            candidates + 17);

    const std::int32_t valid_rows =
        physical_rows - 3;

    std::vector<std::uint16_t> logits(
        static_cast<std::size_t>(physical_rows) * batch);

    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t row = 0;
             row < physical_rows;
             ++row) {
            const float value =
                balanced_fixture
                    ? -1.25f +
                          static_cast<float>(
                              (row * 13 + b * 7) % 19) *
                              0.125f
                    : -6.0f +
                          static_cast<float>(
                              (row * 17 + b * 23) % 41) *
                              0.1875f;

            logits[
                static_cast<std::size_t>(b) *
                    physical_rows +
                row] = f32_to_bf16(value);
        }
    }

    std::vector<std::int32_t> candidate_ids(
        static_cast<std::size_t>(candidates) * batch);

    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t k = 0;
             k < candidates;
             ++k) {
            candidate_ids[
                static_cast<std::size_t>(b) *
                    candidates +
                k] =
                (3 + b * 11 + k) %
                valid_rows;
        }
    }

    if (!balanced_fixture) {
        // Force a deterministic winner for each batch column.
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t winning_k =
                b % candidates;

            const std::int32_t winning_token =
                candidate_ids[
                    static_cast<std::size_t>(b) *
                        candidates +
                    winning_k];

            logits[
                static_cast<std::size_t>(b) *
                    physical_rows +
                winning_token] =
                f32_to_bf16(9.0f + static_cast<float>(b));
        }

        // Explicit tie in the first column: lowest candidate
        // index must win.
        if (candidates >= 2) {
            const std::int32_t token0 = candidate_ids[0];
            const std::int32_t token1 = candidate_ids[1];

            logits[token0] = f32_to_bf16(12.0f);
            logits[token1] = f32_to_bf16(12.0f);
        }
    }

    const OracleResult expected =
        oracle(
            logits,
            physical_rows,
            candidate_ids,
            candidates,
            batch);

    GuardedDeviceBuffer device_logits(
        logits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_candidates(
        candidate_ids.size() * sizeof(std::int32_t));

    GuardedDeviceBuffer device_probabilities(
        candidate_ids.size() * sizeof(float));

    GuardedDeviceBuffer device_winners(
        static_cast<std::size_t>(batch) *
        sizeof(std::int32_t));

    device_logits.copy_from_host(
        logits.data(),
        logits.size() * sizeof(std::uint16_t));

    device_candidates.copy_from_host(
        candidate_ids.data(),
        candidate_ids.size() *
            sizeof(std::int32_t));

    device_probabilities.fill(0xcd);
    device_winners.fill(0xcd);

    Tensor logits_tensor(
        device_logits.data(),
        DType::BF16,
        {physical_rows, batch});

    Tensor candidate_tensor(
        device_candidates.data(),
        DType::I32,
        {candidates, batch});

    Tensor probability_tensor(
        device_probabilities.data(),
        DType::FP32,
        {candidates, batch});

    Tensor winner_tensor(
        device_winners.data(),
        DType::I32,
        {batch});

    ops::constrained_choice(
        logits_tensor,
        candidate_tensor,
        probability_tensor,
        winner_tensor,
        valid_rows,
        nullptr);

    cuda_synchronize();

    const auto actual_probabilities =
        from_device<float>(
            device_probabilities.data(),
            candidate_ids.size());

    const auto actual_winners =
        from_device<std::int32_t>(
            device_winners.data(),
            static_cast<std::size_t>(batch));

    const auto logits_after =
        from_device<std::uint16_t>(
            device_logits.data(),
            logits.size());

    const auto candidates_after =
        from_device<std::int32_t>(
            device_candidates.data(),
            candidate_ids.size());

    const std::string label =
        std::string("constrained_choice ") +
        (balanced_fixture ? "balanced " : "") +
        "K=" +
        std::to_string(candidates) +
        " B=" +
        std::to_string(batch);

    int failures = 0;

    failures += verify_probabilities(
        (label + " probabilities").c_str(),
        actual_probabilities,
        expected.probabilities);

    for (std::int32_t b = 0; b < batch; ++b) {
        float sum = 0.0f;
        for (std::int32_t k = 0; k < candidates; ++k) {
            sum += actual_probabilities[
                static_cast<std::size_t>(b) * candidates + k];
        }

        const float error = std::fabs(sum - 1.0f);
        if (!(error <= 2.0e-6f)) {
            std::cerr
                << "FAIL " << label
                << " probability sum"
                << " B=" << b
                << " sum=" << sum
                << " error=" << error
                << "\n";
            ++failures;
        }

        if (balanced_fixture) {
            float maximum_probability = 0.0f;

            for (std::int32_t k = 0; k < candidates; ++k) {
                maximum_probability =
                    std::max(
                        maximum_probability,
                        actual_probabilities[
                            static_cast<std::size_t>(b) *
                                candidates +
                            k]);
            }

            // This fixture exists specifically to exercise a genuinely
            // distributed finite-choice softmax rather than an effectively
            // saturated argmax case.
            if (!(maximum_probability > 0.125f &&
                  maximum_probability < 0.40f)) {
                std::cerr
                    << "FAIL " << label
                    << " expected non-saturated distribution"
                    << " B=" << b
                    << " max_probability="
                    << maximum_probability
                    << "\n";
                ++failures;
            }
        }
    }

    failures += verify_exact(
        (label + " winners").c_str(),
        actual_winners,
        expected.winners);

    failures += verify_exact(
        (label + " preserves logits").c_str(),
        logits_after,
        logits);

    failures += verify_exact(
        (label + " preserves candidate ids").c_str(),
        candidates_after,
        candidate_ids);

    failures +=
        device_logits.verify_guards(
            (label + " logits").c_str());

    failures +=
        device_candidates.verify_guards(
            (label + " candidate_ids").c_str());

    failures +=
        device_probabilities.verify_guards(
            (label + " probabilities").c_str());

    failures +=
        device_winners.verify_guards(
            (label + " winners").c_str());

    return failures;
}


int run_invalid_candidate_id_safety_case(
    std::int32_t invalid_id,
    const char* marker) {

    constexpr std::int32_t physical_rows = 8;
    constexpr std::int32_t valid_rows = 5;
    constexpr std::int32_t candidates = 3;
    constexpr std::int32_t batch = 1;

    std::vector<std::uint16_t> logits(
        static_cast<std::size_t>(physical_rows),
        f32_to_bf16(-4.0f));

    // Two legal candidates.
    logits[1] = f32_to_bf16(1.0f);
    logits[3] = f32_to_bf16(2.0f);

    // Row valid_rows physically exists but is outside the legal token
    // domain. Give it a huge score so the >=valid_rows test proves the
    // scorer masks by valid_rows rather than merely physical_rows.
    logits[valid_rows] =
        f32_to_bf16(100.0f);

    const std::vector<std::int32_t> candidate_ids{
        1,
        invalid_id,
        3,
    };

    GuardedDeviceBuffer device_logits(
        logits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_candidates(
        candidate_ids.size() * sizeof(std::int32_t));

    GuardedDeviceBuffer device_probabilities(
        candidate_ids.size() * sizeof(float));

    GuardedDeviceBuffer device_winners(
        sizeof(std::int32_t));

    device_logits.copy_from_host(
        logits.data(),
        logits.size() * sizeof(std::uint16_t));

    device_candidates.copy_from_host(
        candidate_ids.data(),
        candidate_ids.size() *
            sizeof(std::int32_t));

    device_probabilities.fill(0xcd);
    device_winners.fill(0xcd);

    Tensor logits_tensor(
        device_logits.data(),
        DType::BF16,
        {physical_rows, batch});

    Tensor candidate_tensor(
        device_candidates.data(),
        DType::I32,
        {candidates, batch});

    Tensor probability_tensor(
        device_probabilities.data(),
        DType::FP32,
        {candidates, batch});

    Tensor winner_tensor(
        device_winners.data(),
        DType::I32,
        {batch});

    ops::constrained_choice(
        logits_tensor,
        candidate_tensor,
        probability_tensor,
        winner_tensor,
        valid_rows,
        nullptr);

    cuda_synchronize();

    const auto probabilities =
        from_device<float>(
            device_probabilities.data(),
            candidate_ids.size());

    const auto winners =
        from_device<std::int32_t>(
            device_winners.data(),
            1);

    const auto logits_after =
        from_device<std::uint16_t>(
            device_logits.data(),
            logits.size());

    const auto candidates_after =
        from_device<std::int32_t>(
            device_candidates.data(),
            candidate_ids.size());

    int failures = 0;

    if (probabilities.size() != 3 ||
        winners.size() != 1) {

        std::cerr
            << "FAIL " << marker
            << ": output size mismatch\n";

        ++failures;

    } else {
        const float probability_sum =
            probabilities[0] +
            probabilities[1] +
            probabilities[2];

        if (!std::isfinite(probabilities[0]) ||
            !std::isfinite(probabilities[1]) ||
            !std::isfinite(probabilities[2]) ||
            !std::isfinite(probability_sum)) {

            std::cerr
                << "FAIL " << marker
                << ": invalid-ID fixture produced non-finite output\n";

            ++failures;
        }

        if (std::fabs(probabilities[1]) >
            1.0e-7F) {

            std::cerr
                << "FAIL " << marker
                << ": invalid candidate received probability "
                << probabilities[1]
                << "\n";

            ++failures;
        }

        if (std::fabs(
                probability_sum -
                1.0F) >
            2.0e-6F) {

            std::cerr
                << "FAIL " << marker
                << ": probability sum="
                << probability_sum
                << "\n";

            ++failures;
        }

        // Legal token 3 has a larger score than legal token 1.
        // The invalid middle candidate must therefore never win.
        if (winners[0] != 2) {
            std::cerr
                << "FAIL " << marker
                << ": invalid candidate affected winner; winner="
                << winners[0]
                << "\n";

            ++failures;
        }
    }

    failures += verify_exact(
        (std::string(marker) +
         " preserves logits").c_str(),
        logits_after,
        logits);

    failures += verify_exact(
        (std::string(marker) +
         " preserves candidate ids").c_str(),
        candidates_after,
        candidate_ids);

    failures +=
        device_logits.verify_guards(
            (std::string(marker) +
             " logits").c_str());

    failures +=
        device_candidates.verify_guards(
            (std::string(marker) +
             " candidate_ids").c_str());

    failures +=
        device_probabilities.verify_guards(
            (std::string(marker) +
             " probabilities").c_str());

    failures +=
        device_winners.verify_guards(
            (std::string(marker) +
             " winners").c_str());

    if (failures == 0) {
        std::cout
            << marker
            << "=PASS\n";
    }

    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout
            << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    failures += run_case(2, 1);
    failures += run_case(4, 3);
    failures += run_case(8, 8);
    failures += run_case(16, 8);

    // Boundary and wide-domain coverage. K=17 is the regression which proves
    // the former fixed-16 implementation ceiling has been crossed.
    failures += run_case(17, 1);
    failures += run_case(31, 1);
    failures += run_case(32, 8);
    failures += run_case(33, 3);
    failures += run_case(64, 1);
    failures += run_case(127, 4);
    failures += run_case(257, 1);

    // Independent CPU oracle + moderate logits. No forced dominant winner
    // and no forced tie: several choices retain meaningful probability mass.
    failures += run_case(8, 4, true);

    // Defensive CUDA memory-safety coverage for callers which violate the
    // public candidate-ID precondition. These do not make invalid IDs valid
    // API inputs; they prove the kernel never indexes logits with them.
    failures +=
        run_invalid_candidate_id_safety_case(
            -1,
            "INVALID_CANDIDATE_NEGATIVE_MEMORY_SAFE");

    failures +=
        run_invalid_candidate_id_safety_case(
            5,
            "INVALID_CANDIDATE_GE_VALID_ROWS_MEMORY_SAFE");

    if (failures == 0) {
        std::cout
            << "LOW_LEVEL_INVALID_ID_MEMORY_SAFETY=PASS\n";
    }

    std::cout
        << (failures == 0 ? "OK" : "FAIL")
        << " constrained_choice public contract\n";

    return failures == 0 ? 0 : 1;
}
