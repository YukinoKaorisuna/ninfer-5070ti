#include <ninfer/engine.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::DecisionFieldInput;
using ninfer::DecisionFieldResult;
using ninfer::DecisionFieldType;
using ninfer::DecisionResult;
using ninfer::TokenId;

bool valid_probs(const DecisionFieldResult& field) {
    if (field.routing_probabilities.size() != field.candidate_values.size()) {
        std::cerr << "FAIL: probability/value size mismatch\n";
        return false;
    }

    double sum = 0.0;

    for (float p : field.routing_probabilities) {
        if (!std::isfinite(p) || p < 0.0F || p > 1.0F) {
            std::cerr << "FAIL: invalid probability\n";
            return false;
        }
        sum += p;
    }

    std::cout
        << "FIELD_" << field.name
        << "_PROB_SUM="
        << std::setprecision(10)
        << sum
        << "\n";

    return std::fabs(sum - 1.0) <= 1.0e-5;
}

void print_field(const DecisionFieldResult& field) {
    std::cout << "FIELD_NAME=" << field.name << "\n";
    std::cout << "SELECTED_VALUE=" << field.selected_value << "\n";
    std::cout << "WINNER_INDEX=" << field.winner_index << "\n";
    std::cout << "WINNER_TOKEN=" << field.winner_token << "\n";
    std::cout << "FRONTIER=" << field.frontier << "\n";
    std::cout << "SUFFIX_TOKENS=" << field.suffix_tokens << "\n";

    std::cout << "CANDIDATE_VALUES=";
    for (std::size_t i = 0; i < field.candidate_values.size(); ++i) {
        if (i != 0) std::cout << ",";
        std::cout << field.candidate_values[i];
    }
    std::cout << "\n";

    std::cout << "CANDIDATE_TOKENS=";
    for (std::size_t i = 0; i < field.candidate_tokens.size(); ++i) {
        if (i != 0) std::cout << ",";
        std::cout << field.candidate_tokens[i];
    }
    std::cout << "\n";

    std::cout << "PROBABILITIES=";
    for (std::size_t i = 0; i < field.routing_probabilities.size(); ++i) {
        if (i != 0) std::cout << ",";
        std::cout << std::setprecision(9) << field.routing_probabilities[i];
    }
    std::cout << "\n";
}

int run(const char* artifact) {
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

    // Same 63-token retained frontier as the M1-C1 qualification.
    std::vector<TokenId> trunk(63, 198);

    auto prompt =
        engine.prepare_tokens(
            std::move(trunk),
            true);

    std::vector<DecisionFieldInput> fields;

    DecisionFieldInput bool_field;
    bool_field.name   = "approved";
    bool_field.type   = DecisionFieldType::Boolean;
    bool_field.suffix = " approved: ";
    fields.push_back(std::move(bool_field));

    DecisionFieldInput enum_field;
    enum_field.name   = "route";
    enum_field.type   = DecisionFieldType::Enum;
    enum_field.suffix = " route: ";
    enum_field.values = {
        "local",
        "remote",
        "human"
    };
    fields.push_back(std::move(enum_field));

    const DecisionResult result =
        engine.decide(
            std::move(prompt),
            std::move(fields));

    std::cout << "FIELD_COUNT=" << result.fields.size() << "\n";
    std::cout << "PROMPT_TOKENS=" << result.prompt.prompt_tokens << "\n";
    std::cout << "REUSED_PROMPT_TOKENS=" << result.reused_prompt_tokens << "\n";

    if (result.fields.size() != 2) {
        std::cerr << "FAIL: expected two typed fields\n";
        return 1;
    }

    const auto& approved = result.fields[0];
    const auto& route    = result.fields[1];

    print_field(approved);
    print_field(route);

    if (!valid_probs(approved) || !valid_probs(route)) {
        return 1;
    }

    if (approved.type != DecisionFieldType::Boolean) {
        std::cerr << "FAIL: bool type not preserved\n";
        return 1;
    }

    if (approved.candidate_values !=
        std::vector<std::string>{"false", "true"}) {
        std::cerr << "FAIL: canonical bool values not preserved\n";
        return 1;
    }

    if (approved.winner_index < 0 ||
        static_cast<std::size_t>(approved.winner_index) >=
            approved.candidate_values.size()) {
        std::cerr << "FAIL: bool winner index invalid\n";
        return 1;
    }

    if (approved.selected_value !=
        approved.candidate_values[
            static_cast<std::size_t>(approved.winner_index)]) {
        std::cerr << "FAIL: bool selected_value mismatch\n";
        return 1;
    }

    if (route.type != DecisionFieldType::Enum) {
        std::cerr << "FAIL: enum type not preserved\n";
        return 1;
    }

    if (route.candidate_values !=
        std::vector<std::string>{"local", "remote", "human"}) {
        std::cerr << "FAIL: enum values not preserved\n";
        return 1;
    }

    if (route.winner_index < 0 ||
        static_cast<std::size_t>(route.winner_index) >=
            route.candidate_values.size()) {
        std::cerr << "FAIL: enum winner index invalid\n";
        return 1;
    }

    if (route.selected_value !=
        route.candidate_values[
            static_cast<std::size_t>(route.winner_index)]) {
        std::cerr << "FAIL: enum selected_value mismatch\n";
        return 1;
    }

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

    if (stats.committed_decode_tokens != 0 ||
        stats.decode_rounds != 0 ||
        stats.decode_row_rounds != 0) {
        std::cerr << "FAIL: typed decision entered normal decode\n";
        return 1;
    }

    std::cout << "TYPED_BOOL=PASS\n";
    std::cout << "TYPED_ENUM=PASS\n";
    std::cout << "SELECTED_VALUE_MAPPING=PASS\n";
    std::cout << "NORMAL_DECODE_USED=NO\n";
    std::cout << "M1C2_TYPED_ENGINE=PASS\n";

    return 0;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv("NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr || *artifact == '\0') {
        std::cout
            << "skip: NINFER_QWEN3_8_27B_DECISION_WEIGHTS is not set\n";
        return 77;
    }

    try {
        return run(artifact);
    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";
        return 1;
    }
}
