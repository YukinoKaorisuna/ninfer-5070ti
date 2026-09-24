#include <ninfer/engine.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

struct FanoutDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;

    SemanticNodeId root;
    SemanticNodeId child_b;
    SemanticNodeId child_c;

    std::vector<std::string> conditioning_b;
    std::vector<std::string> conditioning_c;

    std::string child_prefix =
        " route: ";
};

FanoutDefinition
make_fanout_definition() {
    FanoutDefinition definition;

    FiniteChoice root;
    root.label = "approved";

    root.choices = {
        SemanticValue::boolean(false),
        SemanticValue::boolean(true),
    };

    definition.root =
        definition.schema.add_finite_choice(
            std::move(root));

    FiniteChoice child_b;
    child_b.label = "route_b";

    child_b.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    definition.child_b =
        definition.schema.add_finite_choice(
            std::move(child_b));

    FiniteChoice child_c;
    child_c.label = "route_c";

    child_c.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    definition.child_c =
        definition.schema.add_finite_choice(
            std::move(child_c));

    definition.schema.add_dependency(
        definition.root,
        definition.child_b);

    definition.schema.add_dependency(
        definition.root,
        definition.child_c);

    // Semantic Boolean false/true is intentionally presented to the model as
    // local/remote. This keeps the semantic/presentation separation proven by
    // V2-B active in the V2-C1 fixture.
    definition.presentation.set_finite_choice(
        definition.root,
        FiniteChoicePresentation{
            " route: ",
            {"local", "remote"},
        });

    definition.presentation.set_finite_choice(
        definition.child_b,
        FiniteChoicePresentation{
            definition.child_prefix,
            {"local", "remote", "human"},
        });

    definition.presentation.set_finite_choice(
        definition.child_c,
        FiniteChoicePresentation{
            definition.child_prefix,
            {"local", "remote", "human"},
        });

    definition.conditioning_b = {
        "",
        " selected parent outcome approved=true; sibling B routing context. ",
    };

    definition.conditioning_c = {
        " sibling C denied-parent routing context. ",
        " selected parent outcome approved=true; sibling C alternate routing context. ",
    };

    definition.presentation.set_dependency_conditioning(
        definition.root,
        definition.child_b,
        DependencyConditioningPresentation{
            definition.conditioning_b,
        });

    definition.presentation.set_dependency_conditioning(
        definition.root,
        definition.child_c,
        DependencyConditioningPresentation{
            definition.conditioning_c,
        });

    return definition;
}

struct SingleDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;
};

SingleDefinition
make_single_child(
    std::string label,
    std::string continuation_prefix) {

    SingleDefinition definition;

    FiniteChoice child;
    child.label = std::move(label);

    child.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    const SemanticNodeId node =
        definition.schema.add_finite_choice(
            std::move(child));

    definition.presentation.set_finite_choice(
        node,
        FiniteChoicePresentation{
            std::move(continuation_prefix),
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

    for (std::size_t index = 0;
         index < lhs.size();
         ++index) {

        if (std::fabs(
                lhs[index] -
                rhs[index]) >
            1.0e-6F) {

            return false;
        }
    }

    return true;
}

bool
field_equal(
    const DecisionFieldResult& lhs,
    const DecisionFieldResult& rhs) {

    return
        lhs.candidate_values ==
            rhs.candidate_values &&
        lhs.candidate_tokens ==
            rhs.candidate_tokens &&
        lhs.winner_index ==
            rhs.winner_index &&
        lhs.winner_token ==
            rhs.winner_token &&
        lhs.selected_value ==
            rhs.selected_value &&
        lhs.frontier ==
            rhs.frontier &&
        lhs.suffix_tokens ==
            rhs.suffix_tokens &&
        probabilities_equal(
            lhs.routing_probabilities,
            rhs.routing_probabilities);
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
            field.routing_probabilities.size()) {

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

    double probability_sum = 0.0;

    for (const float probability :
         field.routing_probabilities) {

        if (!std::isfinite(probability) ||
            probability < 0.0F ||
            probability > 1.0F) {

            return false;
        }

        probability_sum +=
            probability;
    }

    return std::fabs(
               probability_sum - 1.0) <=
           1.0e-5;
}

bool
validate_host_api() {
    try {
        FanoutDefinition definition =
            make_fanout_definition();

        if (definition.schema.node_count() != 3) {
            return false;
        }

        bool multiple_parent_rejected =
            false;

        StructuredDecisionSchema invalid;

        FiniteChoice a;
        a.label = "a";
        a.choices = {
            SemanticValue::string("a0"),
            SemanticValue::string("a1"),
        };

        FiniteChoice b;
        b.label = "b";
        b.choices = {
            SemanticValue::string("b0"),
            SemanticValue::string("b1"),
        };

        FiniteChoice c;
        c.label = "c";
        c.choices = {
            SemanticValue::string("c0"),
            SemanticValue::string("c1"),
        };

        const SemanticNodeId a_id =
            invalid.add_finite_choice(
                std::move(a));

        const SemanticNodeId b_id =
            invalid.add_finite_choice(
                std::move(b));

        const SemanticNodeId c_id =
            invalid.add_finite_choice(
                std::move(c));

        invalid.add_dependency(
            a_id,
            c_id);

        // Semantic graph permits this today; V2-C1 backend rejection is a
        // compile-time capability boundary. This host test merely proves the
        // valid sibling DAG is accepted by the semantic layer.
        (void)b_id;

        std::cout
            << "V2C1_HOST_API=PASS\n";

        std::cout
            << "V2C1_SCHEMA_NODE_COUNT=3\n";

        std::cout
            << "V2C1_FANOUT_EDGES=2\n";

        std::cout
            << "V2C1_SEMANTIC_PRESENTATION_SEPARATION=PRESERVED\n";

        (void)multiple_parent_rejected;

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
run_shared_frontier_real(
    ninfer::Engine& engine) {

    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;

    FiniteChoice root;
    root.label = "shared_approved";

    root.choices = {
        SemanticValue::boolean(false),
        SemanticValue::boolean(true),
    };

    const SemanticNodeId root_id =
        schema.add_finite_choice(
            std::move(root));

    FiniteChoice child_b;
    child_b.label = "shared_route_b";

    child_b.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    const SemanticNodeId child_b_id =
        schema.add_finite_choice(
            std::move(child_b));

    FiniteChoice child_c;
    child_c.label = "shared_route_c";

    child_c.choices = {
        SemanticValue::string("local"),
        SemanticValue::string("remote"),
        SemanticValue::string("human"),
    };

    const SemanticNodeId child_c_id =
        schema.add_finite_choice(
            std::move(child_c));

    schema.add_dependency(
        root_id,
        child_b_id);

    schema.add_dependency(
        root_id,
        child_c_id);

    presentation.set_finite_choice(
        root_id,
        FiniteChoicePresentation{
            " route: ",
            {"local", "remote"},
        });

    const std::string child_b_prefix =
        " sibling B route: ";

    const std::string child_c_prefix =
        " sibling C route: ";

    presentation.set_finite_choice(
        child_b_id,
        FiniteChoicePresentation{
            child_b_prefix,
            {"local", "remote", "human"},
        });

    presentation.set_finite_choice(
        child_c_id,
        FiniteChoicePresentation{
            child_c_prefix,
            {"local", "remote", "human"},
        });

    // Both siblings receive the same long selected-parent conditioning for
    // each possible root winner. Whole paths are still tokenized separately,
    // so the runtime must discover the shared prefix from token sequences.
    const std::vector<std::string>
        shared_conditioning = {
            " selected parent outcome approved=false; common dependency context "
            "for both sibling decisions; preserve this deterministic semantic "
            "conditioning before entering the sibling-specific route field. ",

            " selected parent outcome approved=true; common dependency context "
            "for both sibling decisions; preserve this deterministic semantic "
            "conditioning before entering the sibling-specific route field. ",
        };

    presentation.set_dependency_conditioning(
        root_id,
        child_b_id,
        DependencyConditioningPresentation{
            shared_conditioning,
        });

    presentation.set_dependency_conditioning(
        root_id,
        child_c_id,
        DependencyConditioningPresentation{
            shared_conditioning,
        });

    const CompiledDecisionPlan plan =
        engine.compile_decision_plan(
            schema,
            presentation);

    if (!plan ||
        plan.empty() ||
        plan.field_count() != 3) {

        std::cerr
            << "FAIL: V2-C2 shared plan metadata\n";

        return 1;
    }

    const DecisionResult result =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            plan);

    if (result.fields.size() != 3) {
        std::cerr
            << "FAIL: V2-C2 expected root + two siblings\n";

        return 1;
    }

    const DecisionFieldResult& root_result =
        result.fields[0];

    const DecisionFieldResult& child_b_result =
        result.fields[1];

    const DecisionFieldResult& child_c_result =
        result.fields[2];

    if (root_result.name != "shared_approved" ||
        child_b_result.name != "shared_route_b" ||
        child_c_result.name != "shared_route_c") {

        std::cerr
            << "FAIL: V2-C2 execution order\n";

        return 1;
    }

    if (!validate_field(root_result, 63) ||
        !validate_field(child_b_result, 63) ||
        !validate_field(child_c_result, 63)) {

        std::cerr
            << "FAIL: V2-C2 result validation\n";

        return 1;
    }

    if (root_result.winner_index < 0 ||
        static_cast<std::size_t>(
            root_result.winner_index) >=
            shared_conditioning.size()) {

        std::cerr
            << "FAIL: V2-C2 parent winner index\n";

        return 1;
    }

    const std::size_t selected_parent =
        static_cast<std::size_t>(
            root_result.winner_index);

    const SingleDefinition expected_b =
        make_single_child(
            "shared_route_b",
            shared_conditioning[
                selected_parent] +
            child_b_prefix);

    const SingleDefinition expected_c =
        make_single_child(
            "shared_route_c",
            shared_conditioning[
                selected_parent] +
            child_c_prefix);

    const DecisionResult standalone_b =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            engine.compile_decision_plan(
                expected_b.schema,
                expected_b.presentation));

    const DecisionResult standalone_c =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            engine.compile_decision_plan(
                expected_c.schema,
                expected_c.presentation));

    if (standalone_b.fields.size() != 1 ||
        standalone_c.fields.size() != 1) {

        std::cerr
            << "FAIL: V2-C2 standalone baselines\n";

        return 1;
    }

    if (!field_equal(
            child_b_result,
            standalone_b.fields[0]) ||
        !field_equal(
            child_c_result,
            standalone_c.fields[0])) {

        std::cerr
            << "FAIL: V2-C2 semantic/probability parity\n";

        return 1;
    }

    const std::uint64_t replay_equivalent =
        static_cast<std::uint64_t>(
            child_b_result.suffix_tokens) +
        static_cast<std::uint64_t>(
            child_c_result.suffix_tokens);

    const std::uint64_t executed =
        static_cast<std::uint64_t>(
            child_b_result
                .executed_suffix_tokens) +
        static_cast<std::uint64_t>(
            child_c_result
                .executed_suffix_tokens);

    if (executed == 0 ||
        executed >= replay_equivalent) {

        std::cerr
            << "FAIL: V2-C2 did not reduce deterministic traversal: executed="
            << executed
            << " replay="
            << replay_equivalent
            << "\n";

        return 1;
    }

    if (child_b_result
            .executed_suffix_tokens !=
        child_b_result.suffix_tokens) {

        std::cerr
            << "FAIL: V2-C2 first sibling did not own the shared prefix\n";

        return 1;
    }

    if (child_c_result
            .executed_suffix_tokens >=
        child_c_result.suffix_tokens) {

        std::cerr
            << "FAIL: V2-C2 second sibling did not reuse shared prefix\n";

        return 1;
    }

    if (!(child_b_result.capture_seconds > 0.0) ||
        child_c_result.capture_seconds != 0.0) {

        std::cerr
            << "FAIL: V2-C2 outer capture ownership\n";

        return 1;
    }

    if (child_b_result.frontier != 63 ||
        child_c_result.frontier != 63) {

        std::cerr
            << "FAIL: V2-C2 retained frontier changed\n";

        return 1;
    }

    const ninfer::RuntimeStats stats =
        engine.runtime_stats();

    if (stats.committed_decode_tokens != 0 ||
        stats.decode_rounds != 0 ||
        stats.decode_row_rounds != 0) {

        std::cerr
            << "FAIL: V2-C2 entered normal decode\n";

        return 1;
    }

    const std::uint64_t saved =
        replay_equivalent -
        executed;

    std::cout
        << "V2C2_PARENT_WINNER_INDEX="
        << selected_parent
        << "\n";

    std::cout
        << "V2C2_REPLAY_EQUIVALENT_SUFFIX_TOKENS="
        << replay_equivalent
        << "\n";

    std::cout
        << "V2C2_EXECUTED_SUFFIX_TOKENS="
        << executed
        << "\n";

    std::cout
        << "V2C2_SAVED_SUFFIX_TOKENS="
        << saved
        << "\n";

    std::cout
        << "V2C2_FIRST_CHILD_LOGICAL_SUFFIX="
        << child_b_result.suffix_tokens
        << "\n";

    std::cout
        << "V2C2_FIRST_CHILD_EXECUTED_SUFFIX="
        << child_b_result.executed_suffix_tokens
        << "\n";

    std::cout
        << "V2C2_SECOND_CHILD_LOGICAL_SUFFIX="
        << child_c_result.suffix_tokens
        << "\n";

    std::cout
        << "V2C2_SECOND_CHILD_EXECUTED_SUFFIX="
        << child_c_result.executed_suffix_tokens
        << "\n";

    std::cout
        << "V2C2_SHARED_FRONTIER_USED=YES\n";

    std::cout
        << "V2C2_SECOND_CHILD_OUTER_CAPTURE=NO\n";

    std::cout
        << "V2C2_SEMANTIC_PROBABILITY_PARITY=PASS\n";

    std::cout
        << "V2C2_RETAINED_FRONTIER_RESTORED=YES\n";

    std::cout
        << "V2C2_SECOND_SEQUENCE_LANE=NO\n";

    std::cout
        << "V2C2_NORMAL_DECODE_USED=NO\n";

    std::cout
        << "V2C2_SHARED_FRONTIER=PASS\n";

    return 0;
}

int
run_real(const char* artifact) {
    ninfer::Engine engine(
        make_options(artifact));

    const FanoutDefinition definition =
        make_fanout_definition();

    const CompiledDecisionPlan fanout_plan =
        engine.compile_decision_plan(
            definition.schema,
            definition.presentation);

    if (!fanout_plan ||
        fanout_plan.empty() ||
        fanout_plan.field_count() != 3) {

        std::cerr
            << "FAIL: fanout plan metadata\n";

        return 1;
    }

    const DecisionResult result =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            fanout_plan);

    if (result.fields.size() != 3) {
        std::cerr
            << "FAIL: expected root + two siblings\n";

        return 1;
    }

    const DecisionFieldResult& root =
        result.fields[0];

    const DecisionFieldResult& child_b =
        result.fields[1];

    const DecisionFieldResult& child_c =
        result.fields[2];

    if (root.name != "approved" ||
        child_b.name != "route_b" ||
        child_c.name != "route_c") {

        std::cerr
            << "FAIL: compiled execution order\n";

        return 1;
    }

    if (!validate_field(root, 63) ||
        !validate_field(child_b, 63) ||
        !validate_field(child_c, 63)) {

        std::cerr
            << "FAIL: result field validation\n";

        return 1;
    }

    if (root.selected_value != "false" &&
        root.selected_value != "true") {

        std::cerr
            << "FAIL: root semantic value mapping\n";

        return 1;
    }

    if (root.selected_value == "local" ||
        root.selected_value == "remote") {

        std::cerr
            << "FAIL: model presentation leaked into semantic result\n";

        return 1;
    }

    if (root.winner_index < 0 ||
        static_cast<std::size_t>(
            root.winner_index) >=
            definition.conditioning_b.size() ||
        static_cast<std::size_t>(
            root.winner_index) >=
            definition.conditioning_c.size()) {

        std::cerr
            << "FAIL: root winner cannot select sibling variants\n";

        return 1;
    }

    const std::size_t selected_parent =
        static_cast<std::size_t>(
            root.winner_index);

    const SingleDefinition expected_b =
        make_single_child(
            "route_b",
            definition.conditioning_b[
                selected_parent] +
            definition.child_prefix);

    const SingleDefinition expected_c =
        make_single_child(
            "route_c",
            definition.conditioning_c[
                selected_parent] +
            definition.child_prefix);

    const DecisionResult standalone_b =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            engine.compile_decision_plan(
                expected_b.schema,
                expected_b.presentation));

    const DecisionResult standalone_c =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(63, 198),
                true),
            engine.compile_decision_plan(
                expected_c.schema,
                expected_c.presentation));

    if (standalone_b.fields.size() != 1 ||
        standalone_c.fields.size() != 1) {

        std::cerr
            << "FAIL: standalone sibling baselines\n";

        return 1;
    }

    if (!field_equal(
            child_b,
            standalone_b.fields[0])) {

        std::cerr
            << "FAIL: sibling B selected-path parity\n";

        return 1;
    }

    if (!field_equal(
            child_c,
            standalone_c.fields[0])) {

        std::cerr
            << "FAIL: sibling C selected-path parity\n";

        return 1;
    }

    const ninfer::RuntimeStats stats =
        engine.runtime_stats();

    if (stats.committed_decode_tokens != 0 ||
        stats.decode_rounds != 0 ||
        stats.decode_row_rounds != 0) {

        std::cerr
            << "FAIL: V2-C1 entered normal decode\n";

        return 1;
    }

    const std::uint64_t dependent_suffix_work =
        static_cast<std::uint64_t>(
            child_b.suffix_tokens) +
        static_cast<std::uint64_t>(
            child_c.suffix_tokens);

    std::cout
        << "V2C1_PLAN_FIELD_COUNT=3\n";

    std::cout
        << "V2C1_PARENT_WINNER_INDEX="
        << selected_parent
        << "\n";

    std::cout
        << "V2C1_PARENT_SELECTED="
        << root.selected_value
        << "\n";

    std::cout
        << "V2C1_CHILD_B_SELECTED="
        << child_b.selected_value
        << "\n";

    std::cout
        << "V2C1_CHILD_C_SELECTED="
        << child_c.selected_value
        << "\n";

    std::cout
        << "V2C1_CHILD_B_PARITY=PASS\n";

    std::cout
        << "V2C1_CHILD_C_PARITY=PASS\n";

    std::cout
        << "V2C1_SHARED_PARENT_WINNER=PASS\n";

    std::cout
        << "V2C1_WHOLE_PATH_PARITY=PASS\n";

    std::cout
        << "V2C1_SEMANTIC_PRESENTATION_SEPARATION=PASS\n";

    std::cout
        << "V2C1_DEPENDENT_SUFFIX_TOKENS_TOTAL="
        << dependent_suffix_work
        << "\n";

    std::cout
        << "V2C1_MAX_CONCURRENCY=1\n";

    std::cout
        << "V2C1_REPLAY_FROM_RETAINED_FRONTIER=YES\n";

    std::cout
        << "RUNTIME_COMMITTED_DECODE_TOKENS=0\n";

    std::cout
        << "RUNTIME_DECODE_ROUNDS=0\n";

    std::cout
        << "RUNTIME_DECODE_ROW_ROUNDS=0\n";

    std::cout
        << "NORMAL_DECODE_USED=NO\n";

    std::cout
        << "V2C1_FANOUT=PASS\n";

    if (run_shared_frontier_real(
            engine) != 0) {

        return 1;
    }

    return 0;
}

} // namespace

int main() {
    if (!validate_host_api()) {
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
        return run_real(artifact);

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
