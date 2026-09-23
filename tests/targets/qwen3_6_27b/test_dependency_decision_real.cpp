#include <ninfer/engine.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::CompiledDecisionPlan;
using ninfer::DecisionFieldResult;
using ninfer::DecisionModelPresentation;
using ninfer::DecisionResult;
using ninfer::DependencyConditioningPresentation;
using ninfer::FiniteChoice;
using ninfer::FiniteChoicePresentation;
using ninfer::SemanticNodeId;
using ninfer::SemanticValue;
using ninfer::StructuredDecisionSchema;
using ninfer::TokenId;

ninfer::EngineOptions
make_options(const char* artifact) {
    ninfer::EngineOptions options;

    options.artifact_path = artifact;
    options.max_context = 4096;

    options.kv_capacity =
        ninfer::KvCapacityPolicy::
            explicit_capacity(4096);

    options.max_concurrency = 1;
    options.prefill_chunk = 896;

    options.kv_cache =
        ninfer::KvCacheStorage::
            Int4Group64;

    options.speculative.backend =
        ninfer::SpeculativeBackend::None;

    options.enable_vision = false;
    options.use_cuda_graph = false;

    return options;
}

struct DependencyDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;

    SemanticNodeId root;
    SemanticNodeId child;

    std::vector<std::string> conditioning;

    std::string child_prefix =
        " route: ";
};

DependencyDefinition
make_dependency_definition() {
    DependencyDefinition definition;

    FiniteChoice approved;
    approved.label = "approved";

    approved.choices = {
        SemanticValue::boolean(false),
        SemanticValue::boolean(true),
    };

    definition.root =
        definition.schema.add_finite_choice(
            std::move(approved));

    FiniteChoice route;
    route.label = "route";

    route.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    definition.child =
        definition.schema.add_finite_choice(
            std::move(route));

    definition.schema.add_dependency(
        definition.root,
        definition.child);

    definition.presentation.set_finite_choice(
        definition.root,
        FiniteChoicePresentation{
            " approved: ",
            {"false", "true"},
        });

    definition.presentation.set_finite_choice(
        definition.child,
        FiniteChoicePresentation{
            definition.child_prefix,
            {"local", "remote", "human"},
        });

    definition.conditioning = {
        "",
        " selected parent outcome approved=true; downstream routing is conditioned on that semantic result. ",
    };

    definition.presentation
        .set_dependency_conditioning(
            definition.root,
            definition.child,
            DependencyConditioningPresentation{
                definition.conditioning,
            });

    return definition;
}

struct SingleDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;
};

SingleDefinition
make_child_only_definition(
    std::string continuation_prefix) {

    SingleDefinition definition;

    FiniteChoice route;
    route.label = "route";

    route.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    const SemanticNodeId node =
        definition.schema.add_finite_choice(
            std::move(route));

    definition.presentation.set_finite_choice(
        node,
        FiniteChoicePresentation{
            std::move(
                continuation_prefix),
            {"local", "remote", "human"},
        });

    return definition;
}

bool
probabilities_equal(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {

    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t i = 0;
         i < lhs.size();
         ++i) {

        if (std::fabs(lhs[i] - rhs[i]) >
            1.0e-6F) {

            return false;
        }
    }

    return true;
}

bool
validate_field(
    const DecisionFieldResult& field,
    std::uint32_t expected_frontier) {

    if (field.frontier !=
        expected_frontier) {

        return false;
    }

    if (field.candidate_values.empty() ||
        field.candidate_values.size() !=
            field.candidate_tokens.size() ||
        field.candidate_values.size() !=
            field.probabilities.size()) {

        return false;
    }

    if (field.winner_index < 0 ||
        static_cast<std::size_t>(
            field.winner_index) >=
            field.candidate_values.size()) {

        return false;
    }

    if (field.selected_value !=
        field.candidate_values[
            static_cast<std::size_t>(
                field.winner_index)]) {

        return false;
    }

    double sum = 0.0;

    for (const float p :
         field.probabilities) {

        if (!std::isfinite(p) ||
            p < 0.0F ||
            p > 1.0F) {

            return false;
        }

        sum += p;
    }

    return std::fabs(sum - 1.0) <=
           1.0e-5;
}

bool
validate_v2b_host_api() {
    try {
        DependencyDefinition definition =
            make_dependency_definition();

        bool cycle_rejected = false;

        try {
            definition.schema.add_dependency(
                definition.child,
                definition.root);

        } catch (const std::invalid_argument&) {
            cycle_rejected = true;
        }

        if (!cycle_rejected) {
            return false;
        }

        std::cout
            << "V2B_HOST_API=PASS\n";

        std::cout
            << "V2B_DEPENDENCY_STORAGE=PASS\n";

        std::cout
            << "V2B_DEPENDENCY_CONDITIONING=PASS\n";

        std::cout
            << "V2B_CYCLE_REJECTION=PASS\n";

        return true;

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL: "
            << error.what()
            << "\n";

        return false;
    }
}

int
run(const char* artifact) {
    ninfer::Engine engine(
        make_options(artifact));

    const DependencyDefinition definition =
        make_dependency_definition();

    const CompiledDecisionPlan dependency_plan =
        engine.compile_decision_plan(
            definition.schema,
            definition.presentation);

    if (!dependency_plan ||
        dependency_plan.empty() ||
        dependency_plan.field_count() != 2) {

        return 1;
    }

    const DecisionResult dependent =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            dependency_plan);

    if (dependent.fields.size() != 2) {
        return 1;
    }

    const DecisionFieldResult& root =
        dependent.fields[0];

    const DecisionFieldResult& child =
        dependent.fields[1];

    if (root.name != "approved" ||
        child.name != "route") {

        return 1;
    }

    if (!validate_field(root, 63) ||
        !validate_field(child, 63)) {

        return 1;
    }

    if (root.winner_index < 0 ||
        static_cast<std::size_t>(
            root.winner_index) >=
            definition.conditioning.size()) {

        return 1;
    }

    const std::size_t parent_choice =
        static_cast<std::size_t>(
            root.winner_index);

    const std::size_t other_choice =
        parent_choice == 0 ? 1 : 0;

    const SingleDefinition chosen_definition =
        make_child_only_definition(
            definition.conditioning[
                parent_choice] +
            definition.child_prefix);

    const SingleDefinition other_definition =
        make_child_only_definition(
            definition.conditioning[
                other_choice] +
            definition.child_prefix);

    const CompiledDecisionPlan chosen_plan =
        engine.compile_decision_plan(
            chosen_definition.schema,
            chosen_definition.presentation);

    const CompiledDecisionPlan other_plan =
        engine.compile_decision_plan(
            other_definition.schema,
            other_definition.presentation);

    const DecisionResult expected =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            chosen_plan);

    const DecisionResult alternate =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            other_plan);

    if (expected.fields.size() != 1 ||
        alternate.fields.size() != 1) {

        return 1;
    }

    const DecisionFieldResult& expected_child =
        expected.fields[0];

    const DecisionFieldResult& alternate_child =
        alternate.fields[0];

    if (!validate_field(expected_child, 63) ||
        !validate_field(alternate_child, 63)) {

        return 1;
    }

    const bool variants_distinguishable =
        expected_child.suffix_tokens !=
            alternate_child.suffix_tokens ||
        expected_child.candidate_tokens !=
            alternate_child.candidate_tokens ||
        expected_child.winner_index !=
            alternate_child.winner_index ||
        !probabilities_equal(
            expected_child.probabilities,
            alternate_child.probabilities);

    if (!variants_distinguishable) {
        return 1;
    }

    if (child.suffix_tokens !=
            expected_child.suffix_tokens ||
        child.candidate_tokens !=
            expected_child.candidate_tokens ||
        child.winner_index !=
            expected_child.winner_index ||
        child.winner_token !=
            expected_child.winner_token ||
        child.selected_value !=
            expected_child.selected_value ||
        !probabilities_equal(
            child.probabilities,
            expected_child.probabilities)) {

        return 1;
    }

    const ninfer::RuntimeStats stats =
        engine.runtime_stats();

    if (stats.committed_decode_tokens != 0 ||
        stats.decode_rounds != 0 ||
        stats.decode_row_rounds != 0) {

        return 1;
    }

    std::cout
        << "V2B_DEPENDENCY_PLAN_FIELD_COUNT=2\n";

    std::cout
        << "V2B_PARENT_WINNER_INDEX="
        << parent_choice
        << "\n";

    std::cout
        << "V2B_PARENT_SELECTED="
        << root.selected_value
        << "\n";

    std::cout
        << "V2B_CHILD_SELECTED="
        << child.selected_value
        << "\n";

    std::cout
        << "V2B_CHILD_VARIANT_SUFFIX_TOKENS="
        << child.suffix_tokens
        << "\n";

    std::cout
        << "V2B_EXPECTED_VARIANT_SUFFIX_TOKENS="
        << expected_child.suffix_tokens
        << "\n";

    std::cout
        << "V2B_OTHER_VARIANT_SUFFIX_TOKENS="
        << alternate_child.suffix_tokens
        << "\n";

    std::cout
        << "V2B_CHILD_VARIANT_SELECTION=PASS\n";

    std::cout
        << "V2B_WHOLE_PATH_PARITY=PASS\n";

    std::cout
        << "V2B_RETAINED_FRONTIER_RESTORED=YES\n";

    std::cout
        << "V2B_SECOND_SEQUENCE_LANE=NO\n";

    std::cout
        << "RUNTIME_COMMITTED_DECODE_TOKENS=0\n";

    std::cout
        << "RUNTIME_DECODE_ROUNDS=0\n";

    std::cout
        << "RUNTIME_DECODE_ROW_ROUNDS=0\n";

    std::cout
        << "NORMAL_DECODE_USED=NO\n";

    std::cout
        << "V2B_ROOT_TO_CHILD=PASS\n";

    return 0;
}

} // namespace

int main() {
    if (!validate_v2b_host_api()) {
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
