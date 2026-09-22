#include "ninfer/ops/constrained_choice.h"
#include "ops/op_tester.h"

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
             std::int32_t batch) {
    constexpr std::int32_t physical_rows = 64;
    constexpr std::int32_t valid_rows    = 61;

    std::vector<std::uint16_t> logits(
        static_cast<std::size_t>(physical_rows) * batch);

    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t row = 0;
             row < physical_rows;
             ++row) {
            const float value =
                -6.0f +
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
                (3 + b * 11 + k * 7) % valid_rows;
        }
    }

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
        "constrained_choice K=" +
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

    std::cout
        << (failures == 0 ? "OK" : "FAIL")
        << " constrained_choice public contract\n";

    return failures == 0 ? 0 : 1;
}
