#include <ninfer/engine.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::CompiledDecisionPlan;
using ninfer::DecisionFieldInput;
using ninfer::DecisionFieldResult;
using ninfer::DecisionFieldType;
using ninfer::DecisionResult;
using ninfer::StructuredDecisionSchema;
using ninfer::TokenId;

ninfer::EngineOptions make_options(const char* artifact) {
    ninfer::EngineOptions options;

    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk = 896;
    options.kv_cache =
        ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend =
        ninfer::SpeculativeBackend::None;
    options.enable_vision = false;
    options.use_cuda_graph = false;

    return options;
}

StructuredDecisionSchema make_schema() {
    StructuredDecisionSchema schema;

    DecisionFieldInput approved;
    approved.name = "approved";
    approved.type = DecisionFieldType::Boolean;
    approved.suffix = " approved: ";

    DecisionFieldInput route;
    route.name = "route";
    route.type = DecisionFieldType::Enum;
    route.suffix = " route: ";
    route.values = {"local", "remote", "human"};

    schema.fields.push_back(std::move(approved));
    schema.fields.push_back(std::move(route));

    return schema;
}

bool validate_result(
    const DecisionResult& result,
    std::uint32_t expected_frontier) {

    if (result.fields.size() != 2) {
        std::cerr << "FAIL: expected two fields\n";
        return false;
    }

    for (const DecisionFieldResult& field :
         result.fields) {

        if (field.frontier != expected_frontier) {
            std::cerr
                << "FAIL: frontier="
                << field.frontier
                << " expected="
                << expected_frontier
                << "\n";
            return false;
        }

        if (field.candidate_values.empty() ||
            field.candidate_values.size() !=
                field.candidate_tokens.size() ||
            field.candidate_values.size() !=
                field.probabilities.size()) {

            std::cerr
                << "FAIL: candidate metadata mismatch\n";
            return false;
        }

        if (field.winner_index < 0 ||
            static_cast<std::size_t>(
                field.winner_index) >=
                field.candidate_values.size()) {

            std::cerr
                << "FAIL: invalid winner index\n";
            return false;
        }

        if (field.selected_value !=
            field.candidate_values[
                static_cast<std::size_t>(
                    field.winner_index)]) {

            std::cerr
                << "FAIL: selected value mismatch\n";
            return false;
        }

        double sum = 0.0;

        for (float p : field.probabilities) {
            if (!std::isfinite(p) ||
                p < 0.0F ||
                p > 1.0F) {
                std::cerr
                    << "FAIL: invalid probability\n";
                return false;
            }

            sum += p;
        }

        if (std::fabs(sum - 1.0) > 1.0e-5) {
            std::cerr
                << "FAIL: probability sum="
                << sum
                << "\n";
            return false;
        }
    }

    return true;
}

void print_result(
    const char* label,
    const DecisionResult& result) {

    for (std::size_t i = 0;
         i < result.fields.size();
         ++i) {

        const auto& field = result.fields[i];

        std::cout
            << label
            << "_FIELD_" << i
            << "_NAME="
            << field.name
            << "\n";

        std::cout
            << label
            << "_FIELD_" << i
            << "_FRONTIER="
            << field.frontier
            << "\n";

        std::cout
            << label
            << "_FIELD_" << i
            << "_SELECTED="
            << field.selected_value
            << "\n";

        std::cout
            << label
            << "_FIELD_" << i
            << "_CANDIDATE_TOKENS=";

        for (std::size_t j = 0;
             j < field.candidate_tokens.size();
             ++j) {
            if (j != 0) std::cout << ",";
            std::cout << field.candidate_tokens[j];
        }

        std::cout << "\n";
    }
}

int run(const char* artifact) {
    CompiledDecisionPlan plan;

    std::vector<TokenId> bool_tokens;
    std::vector<TokenId> enum_tokens;

    // Engine A: compile once and reuse twice.
    {
        ninfer::Engine engine(
            make_options(artifact));

        plan =
            engine.compile_decision_plan(
                make_schema());

        std::cout
            << "PLAN_VALID="
            << (plan ? "YES" : "NO")
            << "\n";

        std::cout
            << "PLAN_FIELD_COUNT="
            << plan.field_count()
            << "\n";

        if (!plan ||
            plan.empty() ||
            plan.field_count() != 2) {
            std::cerr
                << "FAIL: plan metadata invalid\n";
            return 1;
        }

        const DecisionResult run63 =
            engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(63, 198),
                    true),
                plan);

        const DecisionResult run64 =
            engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(64, 198),
                    true),
                plan);

        print_result("RUN63", run63);
        print_result("RUN64", run64);

        if (!validate_result(run63, 63) ||
            !validate_result(run64, 64)) {
            return 1;
        }

        if (run63.fields[0].candidate_values !=
                run64.fields[0].candidate_values ||
            run63.fields[0].candidate_tokens !=
                run64.fields[0].candidate_tokens ||
            run63.fields[1].candidate_values !=
                run64.fields[1].candidate_values ||
            run63.fields[1].candidate_tokens !=
                run64.fields[1].candidate_tokens) {

            std::cerr
                << "FAIL: compiled metadata changed\n";
            return 1;
        }

        bool_tokens =
            run63.fields[0].candidate_tokens;

        enum_tokens =
            run63.fields[1].candidate_tokens;

        const ninfer::RuntimeStats stats =
            engine.runtime_stats();

        if (stats.committed_decode_tokens != 0 ||
            stats.decode_rounds != 0 ||
            stats.decode_row_rounds != 0) {

            std::cerr
                << "FAIL: Engine A entered normal decode\n";
            return 1;
        }

        std::cout
            << "ENGINE_A_COMMITTED_DECODE_TOKENS=0\n";

        std::cout
            << "ENGINE_A_DECODE_ROUNDS=0\n";

        std::cout
            << "ENGINE_A_PLAN_EXECUTIONS=2\n";
    }

    std::cout
        << "ENGINE_A_DESTROYED=YES\n";

    // Engine B is created only after A has been destroyed.
    bool foreign_rejected = false;

    {
        ninfer::Engine engine(
            make_options(artifact));

        try {
            (void)engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(63, 198),
                    true),
                plan);
        } catch (const std::invalid_argument&) {
            foreign_rejected = true;
        }

        const CompiledDecisionPlan own_plan =
            engine.compile_decision_plan(
                make_schema());

        const DecisionResult own_result =
            engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(63, 198),
                    true),
                own_plan);

        if (!validate_result(own_result, 63)) {
            std::cerr
                << "FAIL: Engine B own plan failed\n";
            return 1;
        }

        if (own_result.fields[0].candidate_tokens !=
                bool_tokens ||
            own_result.fields[1].candidate_tokens !=
                enum_tokens) {

            std::cerr
                << "FAIL: same artifact compiled different token metadata\n";
            return 1;
        }

        const ninfer::RuntimeStats stats =
            engine.runtime_stats();

        if (stats.committed_decode_tokens != 0 ||
            stats.decode_rounds != 0 ||
            stats.decode_row_rounds != 0) {

            std::cerr
                << "FAIL: Engine B entered normal decode\n";
            return 1;
        }
    }

    std::cout
        << "FOREIGN_ENGINE_PLAN_REJECTED="
        << (foreign_rejected ? "YES" : "NO")
        << "\n";

    if (!foreign_rejected) {
        std::cerr
            << "FAIL: expired/foreign plan accepted\n";
        return 1;
    }

    std::cout
        << "PLAN_COMPILED_ONCE_FOR_ENGINE_A=YES\n";

    std::cout
        << "ENGINE_A_PLAN_EXECUTIONS=2\n";

    std::cout
        << "ENGINE_A_FRONTIERS=63,64\n";

    std::cout
        << "COMPILED_METADATA_STABLE=YES\n";

    std::cout
        << "ENGINE_B_OWN_PLAN=PASS\n";

    std::cout
        << "NORMAL_DECODE_USED=NO\n";

    std::cout
        << "COMPILED_PLAN_REUSE=PASS\n";

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
            << "skip: model environment variable not set\n";
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
