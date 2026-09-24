#include <ninfer/engine.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using ninfer::CompiledDecisionPlan;
using ninfer::DecisionFieldResult;
using ninfer::DecisionModelPresentation;
using ninfer::DecisionResult;
using ninfer::FiniteChoice;
using ninfer::FiniteChoicePresentation;
using ninfer::SemanticNodeId;
using ninfer::SemanticValue;
using ninfer::SemanticValueKind;
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

struct DecisionDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;
};

DecisionDefinition make_definition() {
    DecisionDefinition definition;

    FiniteChoice approved;
    approved.label = "approved";
    approved.choices = {
        SemanticValue::boolean(false),
        SemanticValue::boolean(true),
    };

    const SemanticNodeId approved_node =
        definition.schema.add_finite_choice(
            std::move(approved));

    definition.presentation.set_finite_choice(
        approved_node,
        FiniteChoicePresentation{
            " approved: ",
            {"false", "true"},
        });

    FiniteChoice route;
    route.label = "route";
    route.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    const SemanticNodeId route_node =
        definition.schema.add_finite_choice(
            std::move(route));

    definition.presentation.set_finite_choice(
        route_node,
        FiniteChoicePresentation{
            " route: ",
            {"local", "remote", "human"},
        });

    return definition;
}

bool validate_v2a_host_api() {
    try {
        const SemanticValue boolean_true =
            SemanticValue::boolean(true);

        const SemanticValue integer_one =
            SemanticValue::integer(1);

        const SemanticValue number_one =
            SemanticValue::number(1.0);

        const SemanticValue negative_zero =
            SemanticValue::number(-0.0);

        const SemanticValue positive_zero =
            SemanticValue::number(0.0);

        const SemanticValue string_true =
            SemanticValue::string("true");

        if (boolean_true.kind() !=
                SemanticValueKind::Boolean ||
            !boolean_true.boolean_value()) {

            std::cerr
                << "FAIL: SemanticValue boolean contract\n";
            return false;
        }

        if (integer_one.kind() !=
                SemanticValueKind::Integer ||
            integer_one.integer_value() != 1) {

            std::cerr
                << "FAIL: SemanticValue integer contract\n";
            return false;
        }

        if (number_one.kind() !=
                SemanticValueKind::Number ||
            number_one.number_value() != 1.0) {

            std::cerr
                << "FAIL: SemanticValue number contract\n";
            return false;
        }

        if (integer_one == number_one) {
            std::cerr
                << "FAIL: integer and number semantic identities collapsed\n";
            return false;
        }

        if (boolean_true == string_true) {
            std::cerr
                << "FAIL: boolean and string semantic identities collapsed\n";
            return false;
        }

        if (!(negative_zero == positive_zero)) {
            std::cerr
                << "FAIL: negative zero was not canonicalized\n";
            return false;
        }

        bool non_finite_rejected = false;

        try {
            (void)SemanticValue::number(
                std::numeric_limits<double>::infinity());
        } catch (const std::invalid_argument&) {
            non_finite_rejected = true;
        }

        if (!non_finite_rejected) {
            std::cerr
                << "FAIL: non-finite semantic number accepted\n";
            return false;
        }

        StructuredDecisionSchema schema;

        FiniteChoice first;
        first.label = "first";
        first.choices = {
            SemanticValue::boolean(false),
            SemanticValue::boolean(true),
        };

        const SemanticNodeId first_id =
            schema.add_finite_choice(
                std::move(first));

        FiniteChoice second;
        second.label = "second";
        second.choices = {
            SemanticValue::string("a"),
            SemanticValue::string("b"),
        };

        const SemanticNodeId second_id =
            schema.add_finite_choice(
                std::move(second));

        if (!first_id.valid() ||
            !second_id.valid() ||
            first_id.value != 1 ||
            second_id.value != 2 ||
            schema.node_count() != 2 ||
            schema.empty()) {

            std::cerr
                << "FAIL: semantic node ID/schema contract\n";
            return false;
        }

        const StructuredDecisionSchema copied =
            schema;

        if (copied.node_count() != 2 ||
            copied.empty()) {

            std::cerr
                << "FAIL: semantic schema copy contract\n";
            return false;
        }

        bool duplicate_rejected = false;

        try {
            StructuredDecisionSchema duplicate_schema;

            FiniteChoice duplicate;
            duplicate.label = "duplicate";
            duplicate.choices = {
                SemanticValue::string("same"),
                SemanticValue::string("same"),
            };

            (void)duplicate_schema.add_finite_choice(
                std::move(duplicate));
        } catch (const std::invalid_argument&) {
            duplicate_rejected = true;
        }

        if (!duplicate_rejected) {
            std::cerr
                << "FAIL: duplicate semantic choices accepted\n";
            return false;
        }

        DecisionModelPresentation presentation;

        presentation.set_finite_choice(
            first_id,
            FiniteChoicePresentation{
                " first: ",
                {"false", "true"},
            });

        presentation.set_finite_choice(
            second_id,
            FiniteChoicePresentation{
                " second: ",
                {"a", "b"},
            });

        std::cout
            << "V2A_HOST_API=PASS\n";

        std::cout
            << "SEMANTIC_NODE_IDS=1,2\n";

        std::cout
            << "SEMANTIC_SCHEMA_COPY=PASS\n";

        std::cout
            << "SEMANTIC_DUPLICATE_REJECTION=PASS\n";

        std::cout
            << "SEMANTIC_NUMERIC_IDENTITY=TYPE_SENSITIVE\n";

        std::cout
            << "SEMANTIC_NEGATIVE_ZERO=CANONICALIZED\n";

        std::cout
            << "SEMANTIC_NONFINITE=REJECTED\n";

        return true;

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL: V2-A host API exception: "
            << error.what()
            << "\n";

        return false;
    }
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
                field.routing_probabilities.size()) {

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

        for (float p : field.routing_probabilities) {
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

        const DecisionDefinition definition =
            make_definition();

        plan =
            engine.compile_decision_plan(
                definition.schema,
                definition.presentation);

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

        const DecisionDefinition own_definition =
            make_definition();

        const CompiledDecisionPlan own_plan =
            engine.compile_decision_plan(
                own_definition.schema,
                own_definition.presentation);

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
    if (!validate_v2a_host_api()) {
        return 1;
    }

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
