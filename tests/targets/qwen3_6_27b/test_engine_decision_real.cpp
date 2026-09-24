#include <ninfer/engine.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::DecisionFieldResult;
using ninfer::DecisionFieldSpec;
using ninfer::DecisionResult;
using ninfer::TokenId;

bool valid_probability_vector(
    const std::vector<float>& probabilities,
    const char* label) {

    if (probabilities.empty()) {
        std::cerr << "FAIL " << label << ": empty probabilities\n";
        return false;
    }

    double sum = 0.0;

    for (const float value : probabilities) {
        if (!std::isfinite(value) ||
            value < 0.0F ||
            value > 1.0F) {

            std::cerr
                << "FAIL " << label
                << ": invalid probability "
                << value
                << "\n";

            return false;
        }

        sum += static_cast<double>(value);
    }

    std::cout
        << label
        << "_SUM="
        << std::setprecision(10)
        << sum
        << "\n";

    return std::fabs(sum - 1.0) <= 1.0e-5;
}

float max_abs_error(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {

    if (lhs.size() != rhs.size()) {
        return INFINITY;
    }

    float error = 0.0F;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        error = std::max(
            error,
            std::fabs(lhs[i] - rhs[i]));
    }

    return error;
}

void print_field(
    const DecisionFieldResult& field,
    std::size_t index) {

    std::cout
        << "FIELD_" << index
        << "_NAME=" << field.name
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_FRONTIER=" << field.frontier
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_SUFFIX_TOKENS=" << field.suffix_tokens
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_WINNER_INDEX=" << field.winner_index
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_WINNER_TOKEN=" << field.winner_token
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_PROBABILITIES=";

    for (std::size_t i = 0;
         i < field.routing_probabilities.size();
         ++i) {

        if (i != 0) {
            std::cout << ",";
        }

        std::cout
            << std::setprecision(9)
            << field.routing_probabilities[i];
    }

    std::cout << "\n";

    std::cout
        << "FIELD_" << index
        << "_CAPTURE_MS="
        << std::fixed
        << std::setprecision(3)
        << field.capture_seconds * 1000.0
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_SUFFIX_MS="
        << field.suffix_seconds * 1000.0
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_SCORE_MS="
        << field.score_seconds * 1000.0
        << "\n";

    std::cout
        << "FIELD_" << index
        << "_RESTORE_MS="
        << field.restore_seconds * 1000.0
        << "\n";
}

DecisionFieldSpec make_field(
    std::string name,
    std::vector<TokenId> suffix,
    std::vector<TokenId> candidates) {

    DecisionFieldSpec field;

    field.name             = std::move(name);
    field.suffix_tokens    = std::move(suffix);
    field.candidate_tokens = std::move(candidates);

    return field;
}

int run_real_test(const char* artifact) {
    ninfer::EngineOptions options;

    options.artifact_path       = artifact;
    options.max_context         = 4096;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency     = 1;
    options.prefill_chunk       = 896;
    options.kv_cache            =
        ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend =
        ninfer::SpeculativeBackend::None;
    options.enable_vision       = false;
    options.use_cuda_graph      = false;

    ninfer::Engine engine(options);

    const auto load = engine.load_summary();

    std::cout
        << "MODEL_ID=" << load.model_id
        << "\n";

    std::cout
        << "TARGET=" << load.target
        << "\n";

    std::cout
        << "WEIGHTS_ID=" << load.weights_id
        << "\n";

    // Deliberately stop at frontier 63. The two-token B field
    // crosses the 64-token paged-KV boundary.
    std::vector<TokenId> trunk(63, 198);

    auto prompt =
        engine.prepare_tokens(
            std::move(trunk),
            true);

    std::vector<DecisionFieldSpec> fields;

    fields.push_back(
        make_field(
            "A1",
            {198},
            {198, 846, 5834}));

    fields.push_back(
        make_field(
            "B",
            {5834, 198},
            {198, 846, 5834}));

    fields.push_back(
        make_field(
            "A2",
            {198},
            {198, 846, 5834}));

    const DecisionResult result =
        engine.decide(
            std::move(prompt),
            std::move(fields));

    std::cout
        << "PROMPT_TOKENS="
        << result.prompt.prompt_tokens
        << "\n";

    std::cout
        << "FIELD_COUNT="
        << result.fields.size()
        << "\n";

    std::cout
        << "REUSED_PROMPT_TOKENS="
        << result.reused_prompt_tokens
        << "\n";

    std::cout
        << "PREPARE_MS="
        << std::fixed
        << std::setprecision(3)
        << result.prepare_seconds * 1000.0
        << "\n";

    std::cout
        << "TOTAL_MS="
        << result.total_seconds * 1000.0
        << "\n";

    if (result.prompt.prompt_tokens != 63) {
        std::cerr
            << "FAIL: prompt frontier was not 63\n";
        return 1;
    }

    if (result.fields.size() != 3) {
        std::cerr
            << "FAIL: expected three decision fields\n";
        return 1;
    }

    for (std::size_t i = 0;
         i < result.fields.size();
         ++i) {

        print_field(
            result.fields[i],
            i);

        const std::string label =
            "FIELD_" + std::to_string(i);

        if (!valid_probability_vector(
                result.fields[i].routing_probabilities,
                label.c_str())) {

            return 1;
        }
    }

    const auto& a1 = result.fields[0];
    const auto& b  = result.fields[1];
    const auto& a2 = result.fields[2];

    if (a1.name != "A1" ||
        b.name != "B" ||
        a2.name != "A2") {

        std::cerr
            << "FAIL: field ordering changed\n";

        return 1;
    }

    if (a1.frontier != 63 ||
        b.frontier != 63 ||
        a2.frontier != 63) {

        std::cerr
            << "FAIL: shared frontier changed\n";

        return 1;
    }

    if (a1.suffix_tokens != 1 ||
        b.suffix_tokens != 2 ||
        a2.suffix_tokens != 1) {

        std::cerr
            << "FAIL: suffix accounting incorrect\n";

        return 1;
    }

    if (a1.candidate_tokens != a2.candidate_tokens) {
        std::cerr
            << "FAIL: A candidate echo changed\n";

        return 1;
    }

    if (a1.winner_index != a2.winner_index ||
        a1.winner_token != a2.winner_token) {

        std::cerr
            << "FAIL: repeated A winner changed\n";

        return 1;
    }

    const float repeat_error =
        max_abs_error(
            a1.routing_probabilities,
            a2.routing_probabilities);

    std::cout
        << "A_REPEAT_MAX_ABS_ERROR="
        << std::setprecision(10)
        << repeat_error
        << "\n";

    if (repeat_error > 1.0e-6F) {
        std::cerr
            << "FAIL: repeated A probabilities changed\n";

        return 1;
    }

    std::cout
        << "A_REPEAT_WINNER_MATCH=YES\n";

    std::cout
        << "A_REPEAT_PROBABILITY_MATCH=YES\n";

    std::cout
        << "SHARED_FRONTIER_STABLE=YES\n";

    std::cout
        << "CROSS_PAGE_FIELD_PRESENT=YES\n";

    const ninfer::RuntimeStats stats =
        engine.runtime_stats();

    std::cout
        << "RUNTIME_COMPUTED_PREFILL_TOKENS="
        << stats.computed_prefill_tokens
        << "\n";

    std::cout
        << "RUNTIME_COMMITTED_DECODE_TOKENS="
        << stats.committed_decode_tokens
        << "\n";

    std::cout
        << "RUNTIME_DECODE_ROUNDS="
        << stats.decode_rounds
        << "\n";

    std::cout
        << "RUNTIME_DECODE_ROW_ROUNDS="
        << stats.decode_row_rounds
        << "\n";

    std::cout
        << "RUNTIME_RUNNING_REQUESTS="
        << stats.running_requests
        << "\n";

    std::cout
        << "RUNTIME_WAITING_REQUESTS="
        << stats.waiting_requests
        << "\n";

    if (stats.decode_rounds != 0 ||
        stats.decode_row_rounds != 0 ||
        stats.committed_decode_tokens != 0) {

        std::cerr
            << "FAIL: decision request entered normal decode\n";

        return 1;
    }

    if (stats.running_requests != 0 ||
        stats.waiting_requests != 0) {

        std::cerr
            << "FAIL: completed decision remained scheduled\n";

        return 1;
    }

    std::cout
        << "NORMAL_DECODE_USED=NO\n";

    std::cout
        << "REQUEST_COMPLETED=YES\n";

    std::cout
        << "M1C1_ENGINE_DECISION=PASS\n";

    return 0;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr ||
        *artifact == '\0') {

        std::cout
            << "skip: "
               "NINFER_QWEN3_8_27B_DECISION_WEIGHTS "
               "is not set\n";

        return 77;
    }

    try {
        return run_real_test(artifact);

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
