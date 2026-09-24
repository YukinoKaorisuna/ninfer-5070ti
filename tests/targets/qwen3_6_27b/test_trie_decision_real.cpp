#include <ninfer/engine.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::CompiledDecisionPlan;
using ninfer::DecisionFieldInput;
using ninfer::DecisionFieldResult;
using ninfer::DecisionHandle;
using ninfer::DecisionModelPresentation;
using ninfer::DecisionResult;
using ninfer::DependencyConditioningPresentation;
using ninfer::FiniteChoice;
using ninfer::FiniteChoicePresentation;
using ninfer::SemanticNodeId;
using ninfer::SemanticValue;
using ninfer::StructuredDecisionSchema;
using ninfer::TokenId;

struct Definition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;
    std::vector<std::string> semantic_values;
};

struct TrieMetrics {
    bool valid = false;
    bool multi_token_paths = false;
    bool shared_internal_prefix = false;
    bool deterministic_unary_tail = false;

    std::size_t common_prefix_tokens = 0;
    std::size_t ambiguous_probe_count = 0;
    std::size_t ambiguity_depth_count = 0;
    std::size_t max_probe_suffix_tokens = 0;
    std::uint64_t executed_suffix_tokens = 0;
};

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

Definition make_definition(
    const std::string& prefix,
    const std::vector<std::string>& candidate_texts) {

    Definition definition;

    FiniteChoice choice;
    choice.label = "v2d_route";

    for (std::size_t i = 0;
         i < candidate_texts.size();
         ++i) {

        const std::string semantic =
            "semantic-" +
            std::to_string(i);

        definition.semantic_values.push_back(
            semantic);

        choice.choices.push_back(
            SemanticValue::string(
                semantic));
    }

    const SemanticNodeId node =
        definition.schema.add_finite_choice(
            std::move(choice));

    definition.presentation.set_finite_choice(
        node,
        FiniteChoicePresentation{
            prefix,
            candidate_texts,
        });

    return definition;
}

std::size_t common_prefix_length(
    const std::vector<std::vector<TokenId>>& paths) {

    if (paths.empty()) {
        return 0;
    }

    std::size_t common =
        paths.front().size();

    for (std::size_t path_index = 1;
         path_index < paths.size();
         ++path_index) {

        common =
            std::min(
                common,
                paths[path_index].size());

        std::size_t matched = 0;

        while (
            matched < common &&
            paths.front()[matched] ==
                paths[path_index][matched]) {

            ++matched;
        }

        common = matched;
    }

    return common;
}

TrieMetrics analyze_paths(
    const std::vector<std::vector<TokenId>>& paths) {

    TrieMetrics metrics;

    if (paths.size() < 4) {
        return metrics;
    }

    metrics.common_prefix_tokens =
        common_prefix_length(paths);

    if (metrics.common_prefix_tokens == 0) {
        return metrics;
    }

    struct Node {
        std::map<TokenId, std::size_t> children;
        std::int32_t terminal = -1;
    };

    std::vector<Node> nodes(1);

    for (std::size_t candidate_index = 0;
         candidate_index < paths.size();
         ++candidate_index) {

        const auto& path =
            paths[candidate_index];

        if (path.size() <=
            metrics.common_prefix_tokens) {

            return metrics;
        }

        if (path.size() >
            metrics.common_prefix_tokens + 1) {

            metrics.multi_token_paths = true;
        }

        std::size_t node_index = 0;

        for (std::size_t token_index =
                 metrics.common_prefix_tokens;
             token_index < path.size();
             ++token_index) {

            const TokenId token =
                path[token_index];

            auto it =
                nodes[node_index]
                    .children
                    .find(token);

            if (it ==
                nodes[node_index]
                    .children.end()) {

                const std::size_t child =
                    nodes.size();

                nodes[node_index]
                    .children
                    .emplace(
                        token,
                        child);

                nodes.emplace_back();

                node_index = child;

            } else {
                node_index =
                    it->second;
            }

            if (nodes[node_index].terminal >= 0 &&
                token_index + 1 <
                    path.size()) {

                return metrics;
            }
        }

        if (nodes[node_index].terminal >= 0 ||
            !nodes[node_index].children.empty()) {

            return metrics;
        }

        nodes[node_index].terminal =
            static_cast<std::int32_t>(
                candidate_index);
    }

    std::set<std::size_t>
        ambiguity_depths;

    std::function<void(
        std::size_t,
        std::size_t,
        std::optional<std::size_t>)>
        visit;

    visit =
        [&](std::size_t node_index,
            std::size_t depth,
            std::optional<std::size_t>
                last_ambiguity_depth) {

        const Node& node =
            nodes[node_index];

        if (node.children.size() >= 2) {
            ++metrics.ambiguous_probe_count;

            ambiguity_depths.insert(
                depth);

            const std::size_t
                probe_suffix_tokens =
                    metrics.common_prefix_tokens +
                    depth;

            metrics.max_probe_suffix_tokens =
                std::max(
                    metrics.max_probe_suffix_tokens,
                    probe_suffix_tokens);

            metrics.executed_suffix_tokens +=
                static_cast<std::uint64_t>(
                    probe_suffix_tokens);

            if (depth > 0) {
                metrics.shared_internal_prefix =
                    true;
            }

            last_ambiguity_depth =
                depth;
        }

        if (node.terminal >= 0 &&
            last_ambiguity_depth.has_value() &&
            depth >
                *last_ambiguity_depth + 1) {

            metrics.deterministic_unary_tail =
                true;
        }

        for (const auto& [token, child] :
             node.children) {

            (void)token;

            visit(
                child,
                depth + 1,
                last_ambiguity_depth);
        }
    };

    visit(
        0,
        0,
        std::nullopt);

    metrics.ambiguity_depth_count =
        ambiguity_depths.size();

    metrics.valid =
        metrics.multi_token_paths &&
        metrics.ambiguous_probe_count >= 2 &&
        metrics.ambiguity_depth_count >= 2 &&
        metrics.shared_internal_prefix &&
        metrics.deterministic_unary_tail;

    return metrics;
}

bool valid_probability_vector(
    const DecisionFieldResult& field,
    double* sum_out) {

    if (field.routing_probabilities.size() !=
        field.candidate_values.size()) {

        return false;
    }

    double sum = 0.0;

    for (const float probability :
         field.routing_probabilities) {

        if (!std::isfinite(probability) ||
            probability < 0.0F ||
            probability > 1.0F) {

            return false;
        }

        sum +=
            static_cast<double>(
                probability);
    }

    if (sum_out != nullptr) {
        *sum_out = sum;
    }

    return std::fabs(sum - 1.0) <=
        1.0e-5;
}

bool probabilities_equal(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {

    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t i = 0;
         i < lhs.size();
         ++i) {

        if (std::fabs(
                lhs[i] - rhs[i]) >
            1.0e-6F) {

            return false;
        }
    }

    return true;
}

void print_paths(
    const std::vector<std::string>& texts,
    const std::vector<std::vector<TokenId>>& paths) {

    for (std::size_t i = 0;
         i < paths.size();
         ++i) {

        std::cout
            << "V2D_CANDIDATE_"
            << i
            << "_TEXT="
            << texts[i]
            << "\n";

        std::cout
            << "V2D_CANDIDATE_"
            << i
            << "_TOKEN_PATH=";

        for (std::size_t j = 0;
             j < paths[i].size();
             ++j) {

            if (j != 0) {
                std::cout << ",";
            }

            std::cout
                << paths[i][j];
        }

        std::cout << "\n";
    }
}

bool exact_prefix_rejected(
    ninfer::Engine& engine) {

    const std::vector<
        std::pair<std::string, std::string>>
        pairs = {
            {"local", "local alpine"},
            {"remote", "remote coastal"},
            {"alpha", "alpha beta"},
            {"north", "north east"},
            {"one", "one two"},
            {"route", "route north"},
        };

    for (const auto& [short_text, long_text] :
         pairs) {

        try {
            Definition definition =
                make_definition(
                    " qualification: ",
                    {short_text, long_text});

            (void)engine.compile_decision_plan(
                definition.schema,
                definition.presentation);

        } catch (const std::invalid_argument& error) {
            const std::string message =
                error.what();

            if (message.find(
                    "exact token prefix") !=
                    std::string::npos ||
                message.find(
                    "exact-prefix") !=
                    std::string::npos) {

                std::cout
                    << "V2D_EXACT_PREFIX_PAIR="
                    << short_text
                    << "|"
                    << long_text
                    << "\n";

                return true;
            }
        }
    }

    return false;
}

bool duplicate_path_rejected(
    ninfer::Engine& engine) {

    // These first two strings are byte-distinct (NFC versus NFD) but the
    // Qwen tokenizer normalizes them to the same model-facing token path.
    // A third divergent candidate keeps the global common prefix shorter than
    // the duplicate paths so rejection occurs in trie construction rather
    // than presentation-text or exact-prefix validation.
    const std::string nfc =
        "caf\xc3\xa9 route";
    const std::string nfd =
        "cafe\xcc\x81 route";

    try {
        Definition definition =
            make_definition(
                " qualification: ",
                {
                    nfc,
                    nfd,
                    "remote route",
                });

        (void)engine.compile_decision_plan(
            definition.schema,
            definition.presentation);

    } catch (const std::invalid_argument& error) {
        const std::string message =
            error.what();

        if (message.find(
                "duplicate complete token paths") !=
            std::string::npos) {

            std::cout
                << "V2D_DUPLICATE_PATH_FIXTURE=NFC_NFD_NORMALIZATION\n";

            return true;
        }
    }

    return false;
}

bool validate_trie_field(
    const DecisionFieldResult& field,
    std::string_view expected_name,
    std::size_t candidate_count,
    std::uint32_t expected_frontier) {

    if (field.name != expected_name ||
        field.frontier != expected_frontier ||
        field.candidate_values.size() != candidate_count ||
        field.candidate_token_paths.size() != candidate_count ||
        !field.candidate_tokens.empty() ||
        field.routing_probabilities.size() != candidate_count ||
        field.winner_token != -1 ||
        field.winner_index < 0 ||
        static_cast<std::size_t>(field.winner_index) >= candidate_count ||
        field.selected_value !=
            field.candidate_values[
                static_cast<std::size_t>(
                    field.winner_index)]) {

        return false;
    }

    double probability_sum = 0.0;

    return valid_probability_vector(
        field,
        &probability_sum);
}

bool validate_dependent_trie_async(
    ninfer::Engine& engine,
    const std::vector<std::string>& candidate_texts) {

    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;

    FiniteChoice root;
    root.label = "v2d_root";

    FiniteChoice child;
    child.label = "v2d_child";

    for (std::size_t i = 0;
         i < candidate_texts.size();
         ++i) {

        root.choices.push_back(
            SemanticValue::string(
                "root-semantic-" +
                std::to_string(i)));

        child.choices.push_back(
            SemanticValue::string(
                "child-semantic-" +
                std::to_string(i)));
    }

    const SemanticNodeId root_id =
        schema.add_finite_choice(
            std::move(root));

    const SemanticNodeId child_id =
        schema.add_finite_choice(
            std::move(child));

    schema.add_dependency(
        root_id,
        child_id);

    presentation.set_finite_choice(
        root_id,
        FiniteChoicePresentation{
            " route: ",
            candidate_texts,
        });

    presentation.set_finite_choice(
        child_id,
        FiniteChoicePresentation{
            " route: ",
            candidate_texts,
        });

    const std::vector<std::string> conditioning = {
        " parent outcome alpha; ",
        " parent outcome beta; ",
        " parent outcome gamma; ",
        " parent outcome delta; ",
    };

    if (conditioning.size() != candidate_texts.size()) {
        return false;
    }

    presentation.set_dependency_conditioning(
        root_id,
        child_id,
        DependencyConditioningPresentation{
            conditioning,
        });

    const CompiledDecisionPlan plan =
        engine.compile_decision_plan(
            schema,
            presentation);

    DecisionHandle handle =
        engine.submit_decision(
            engine.prepare_tokens(
                std::vector<TokenId>(
                    63,
                    198),
                true),
            plan);

    if (!handle) {
        return false;
    }

    const DecisionResult result =
        handle.wait();

    if (result.fields.size() != 2 ||
        !validate_trie_field(
            result.fields[0],
            "v2d_root",
            candidate_texts.size(),
            63) ||
        !validate_trie_field(
            result.fields[1],
            "v2d_child",
            candidate_texts.size(),
            63)) {

        return false;
    }

    const DecisionFieldResult& root_result =
        result.fields[0];

    const DecisionFieldResult& child_result =
        result.fields[1];

    const std::size_t parent_choice =
        static_cast<std::size_t>(
            root_result.winner_index);

    Definition expected_child =
        make_definition(
            conditioning[parent_choice] +
                " route: ",
            candidate_texts);

    const CompiledDecisionPlan expected_plan =
        engine.compile_decision_plan(
            expected_child.schema,
            expected_child.presentation);

    const DecisionResult expected =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(
                    63,
                    198),
                true),
            expected_plan);

    if (expected.fields.size() != 1 ||
        !validate_trie_field(
            expected.fields[0],
            "v2d_route",
            candidate_texts.size(),
            63)) {

        return false;
    }

    const DecisionFieldResult& expected_field =
        expected.fields[0];

    if (child_result.candidate_token_paths !=
            expected_field.candidate_token_paths ||
        child_result.winner_index !=
            expected_field.winner_index ||
        !probabilities_equal(
            child_result.routing_probabilities,
            expected_field.routing_probabilities)) {

        return false;
    }

    std::cout
        << "V2D_DEPENDENT_TRIE_ROOT=PASS\n";
    std::cout
        << "V2D_DEPENDENT_TRIE_CHILD=PASS\n";
    std::cout
        << "V2D_DEPENDENT_TRIE_VARIANT_SELECTION=PASS\n";
    std::cout
        << "V2D_ASYNC_DECISION_HANDLE=PASS\n";

    return true;
}

bool validate_typed_multitoken_input(
    ninfer::Engine& engine,
    const std::vector<std::string>& candidate_texts) {

    DecisionFieldInput field;
    field.name = "typed_multitoken";
    field.type = ninfer::DecisionFieldType::Enum;
    field.suffix = " route: ";
    field.values = candidate_texts;

    std::vector<DecisionFieldInput> fields;
    fields.push_back(
        std::move(field));

    const DecisionResult result =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(
                    63,
                    198),
                true),
            std::move(fields));

    if (result.fields.size() != 1) {
        return false;
    }

    const DecisionFieldResult& resolved =
        result.fields.front();

    if (!validate_trie_field(
            resolved,
            "typed_multitoken",
            candidate_texts.size(),
            63) ||
        resolved.candidate_values !=
            candidate_texts) {

        return false;
    }

    std::cout
        << "V2D_TYPED_MULTITOKEN_INPUT=PASS\n";

    return true;
}


bool validate_wide_candidate_domain(
    ninfer::Engine& engine) {

    std::vector<std::string>
        candidate_texts;

    candidate_texts.reserve(26);

    for (char letter = 'A';
         letter <= 'Z';
         ++letter) {

        std::string candidate;

        candidate.push_back(
            letter);

        candidate +=
            " route terminal";

        candidate_texts.push_back(
            std::move(candidate));
    }

    Definition definition =
        make_definition(
            " wide choice: ",
            candidate_texts);

    const CompiledDecisionPlan plan =
        engine.compile_decision_plan(
            definition.schema,
            definition.presentation);

    const DecisionResult result =
        engine.decide(
            engine.prepare_tokens(
                std::vector<TokenId>(
                    63,
                    198),
                true),
            plan);

    if (result.fields.size() != 1) {
        return false;
    }

    const DecisionFieldResult& field =
        result.fields.front();

    if (field.candidate_values !=
            definition.semantic_values ||
        field.candidate_values.size() != 26 ||
        field.routing_probabilities.size() != 26 ||
        field.frontier != 63 ||
        field.winner_index < 0 ||
        static_cast<std::size_t>(
            field.winner_index) >=
                field.candidate_values.size() ||
        field.selected_value !=
            field.candidate_values[
                static_cast<std::size_t>(
                    field.winner_index)]) {

        return false;
    }

    double probability_sum = 0.0;

    if (!valid_probability_vector(
            field,
            &probability_sum)) {

        return false;
    }

    std::size_t outgoing_degree = 0;

    if (!field.candidate_tokens.empty()) {
        if (!field.candidate_token_paths.empty()) {
            return false;
        }

        outgoing_degree =
            field.candidate_tokens.size();

    } else {
        if (field.candidate_token_paths.size() !=
            candidate_texts.size()) {

            return false;
        }

        const std::size_t common =
            common_prefix_length(
                field.candidate_token_paths);

        std::set<TokenId> outgoing;

        for (const auto& path :
             field.candidate_token_paths) {

            if (path.size() <= common) {
                return false;
            }

            outgoing.insert(
                path[common]);
        }

        outgoing_degree =
            outgoing.size();

        if (field.winner_token != -1) {
            return false;
        }
    }

    if (outgoing_degree <= 16) {
        std::cerr
            << "FAIL: wide choice did not create an ambiguity degree above 16; degree="
            << outgoing_degree
            << "\n";

        return false;
    }

    std::cout
        << "V2D_WIDE_CANDIDATE_COUNT="
        << field.candidate_values.size()
        << "\n";

    std::cout
        << "V2D_WIDE_OUTGOING_DEGREE="
        << outgoing_degree
        << "\n";

    std::cout
        << "V2D_WIDE_PROBABILITY_SUM="
        << std::setprecision(10)
        << probability_sum
        << "\n";

    std::cout
        << "V2D_K16_CAP_REMOVED=PASS\n";

    return true;
}


int run(const char* artifact) {
    ninfer::Engine engine(
        make_options(artifact));

    if (!exact_prefix_rejected(engine)) {
        std::cerr
            << "FAIL: no exact-prefix candidate pair was rejected\n";

        return 1;
    }

    std::cout
        << "V2D_EXACT_PREFIX_REJECTION=PASS\n";

    if (!duplicate_path_rejected(engine)) {
        std::cerr
            << "FAIL: duplicate model-facing token path accepted\n";

        return 1;
    }

    std::cout
        << "V2D_DUPLICATE_TOKEN_PATH_REJECTION=PASS\n";

    const std::vector<
        std::vector<std::string>>
        candidate_sets = {

        {
            "local alpine ridge north",
            "local alpine valley south",
            "remote coastal ridge east",
            "remote coastal valley west",
        },

        {
            "alpha northern mountain road",
            "alpha northern river trail",
            "beta southern mountain path",
            "beta southern river route",
        },

        {
            "group red apple orchard north",
            "group red berry orchard south",
            "group blue apple valley east",
            "group blue berry valley west",
        },
    };

    for (std::size_t set_index = 0;
         set_index < candidate_sets.size();
         ++set_index) {

        const auto& candidate_texts =
            candidate_sets[set_index];

        Definition definition =
            make_definition(
                " route: ",
                candidate_texts);

        CompiledDecisionPlan plan;

        try {
            plan =
                engine.compile_decision_plan(
                    definition.schema,
                    definition.presentation);

        } catch (const std::exception& error) {
            std::cout
                << "V2D_SET_"
                << set_index
                << "_COMPILE_REJECTED="
                << error.what()
                << "\n";

            continue;
        }

        const DecisionResult first =
            engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(
                        63,
                        198),
                    true),
                plan);

        if (first.fields.size() != 1) {
            std::cerr
                << "FAIL: D1 plan produced unexpected field count\n";

            return 1;
        }

        const DecisionFieldResult& field =
            first.fields.front();

        if (field.candidate_token_paths.size() !=
                candidate_texts.size() ||
            !field.candidate_tokens.empty()) {

            std::cout
                << "V2D_SET_"
                << set_index
                << "_NOT_TRIE=YES\n";

            continue;
        }

        const TrieMetrics metrics =
            analyze_paths(
                field.candidate_token_paths);

        std::cout
            << "V2D_SET_"
            << set_index
            << "_AMBIGUOUS_PROBES="
            << metrics.ambiguous_probe_count
            << "\n";

        std::cout
            << "V2D_SET_"
            << set_index
            << "_AMBIGUITY_DEPTHS="
            << metrics.ambiguity_depth_count
            << "\n";

        if (!metrics.valid) {
            print_paths(
                candidate_texts,
                field.candidate_token_paths);

            continue;
        }

        if (field.candidate_values !=
            definition.semantic_values) {

            std::cerr
                << "FAIL: semantic candidate values changed\n";

            return 1;
        }

        double probability_sum = 0.0;

        if (!valid_probability_vector(
                field,
                &probability_sum)) {

            std::cerr
                << "FAIL: D1 probability vector invalid\n";

            return 1;
        }

        if (field.winner_index < 0 ||
            static_cast<std::size_t>(
                field.winner_index) >=
                field.candidate_values.size()) {

            std::cerr
                << "FAIL: D1 winner index invalid\n";

            return 1;
        }

        if (field.selected_value !=
            field.candidate_values[
                static_cast<std::size_t>(
                    field.winner_index)]) {

            std::cerr
                << "FAIL: D1 semantic winner mapping invalid\n";

            return 1;
        }

        if (field.winner_token != -1) {
            std::cerr
                << "FAIL: D1 fabricated a singular winner token\n";

            return 1;
        }

        if (field.frontier != 63) {
            std::cerr
                << "FAIL: retained frontier changed; frontier="
                << field.frontier
                << "\n";

            return 1;
        }

        if (field.suffix_tokens !=
            metrics.max_probe_suffix_tokens) {

            std::cerr
                << "FAIL: max probe suffix accounting mismatch; result="
                << field.suffix_tokens
                << " expected="
                << metrics.max_probe_suffix_tokens
                << "\n";

            return 1;
        }

        if (field.executed_suffix_tokens !=
            metrics.executed_suffix_tokens) {

            std::cerr
                << "FAIL: executed trie suffix accounting mismatch; result="
                << field.executed_suffix_tokens
                << " expected="
                << metrics.executed_suffix_tokens
                << "\n";

            return 1;
        }

        const DecisionResult repeat =
            engine.decide(
                engine.prepare_tokens(
                    std::vector<TokenId>(
                        63,
                        198),
                    true),
                plan);

        if (repeat.fields.size() != 1) {
            std::cerr
                << "FAIL: repeated D1 result field count changed\n";

            return 1;
        }

        const DecisionFieldResult& repeated =
            repeat.fields.front();

        if (repeated.frontier !=
                field.frontier ||
            repeated.candidate_token_paths !=
                field.candidate_token_paths ||
            repeated.winner_index !=
                field.winner_index ||
            repeated.selected_value !=
                field.selected_value ||
            !probabilities_equal(
                repeated.routing_probabilities,
                field.routing_probabilities)) {

            std::cerr
                << "FAIL: repeated D1 execution changed semantic result\n";

            return 1;
        }

        if (!validate_dependent_trie_async(
                engine,
                candidate_texts)) {

            std::cerr
                << "FAIL: dependent trie / async DecisionHandle qualification failed\n";

            return 1;
        }

        if (!validate_typed_multitoken_input(
                engine,
                candidate_texts)) {

            std::cerr
                << "FAIL: typed multi-token convenience path qualification failed\n";

            return 1;
        }

        if (!validate_wide_candidate_domain(
                engine)) {

            std::cerr
                << "FAIL: >16 finite-choice qualification failed\n";

            return 1;
        }

        const ninfer::RuntimeStats stats =
            engine.runtime_stats();

        if (stats.committed_decode_tokens != 0 ||
            stats.decode_rounds != 0 ||
            stats.decode_row_rounds != 0) {

            std::cerr
                << "FAIL: D1 entered ordinary decode\n";

            return 1;
        }

        print_paths(
            candidate_texts,
            field.candidate_token_paths);

        std::cout
            << "V2D_CHOSEN_SET="
            << set_index
            << "\n";

        std::cout
            << "V2D_CANDIDATE_COUNT="
            << field.candidate_values.size()
            << "\n";

        std::cout
            << "V2D_COMMON_PREFIX_TOKENS="
            << metrics.common_prefix_tokens
            << "\n";

        std::cout
            << "V2D_AMBIGUOUS_PROBE_COUNT="
            << metrics.ambiguous_probe_count
            << "\n";

        std::cout
            << "V2D_AMBIGUITY_DEPTH_COUNT="
            << metrics.ambiguity_depth_count
            << "\n";

        std::cout
            << "V2D_MAX_PROBE_SUFFIX_TOKENS="
            << metrics.max_probe_suffix_tokens
            << "\n";

        std::cout
            << "V2D_EXECUTED_SUFFIX_TOKENS="
            << field.executed_suffix_tokens
            << "\n";

        std::cout
            << "V2D_CANDIDATE_PROBABILITY_SUM="
            << std::setprecision(10)
            << probability_sum
            << "\n";

        std::cout
            << "V2D_WINNER_INDEX="
            << field.winner_index
            << "\n";

        std::cout
            << "V2D_SELECTED_VALUE="
            << field.selected_value
            << "\n";

        std::cout
            << "V2D_WINNER_TOKEN="
            << field.winner_token
            << "\n";

        std::cout
            << "V2D_WHOLE_PATH_TOKENIZATION=PASS\n";

        std::cout
            << "V2D_MULTI_TOKEN_PATHS=PASS\n";

        std::cout
            << "V2D_TWO_AMBIGUITY_DEPTHS=PASS\n";

        std::cout
            << "V2D_SHARED_INTERNAL_TRIE_PREFIX=PASS\n";

        std::cout
            << "V2D_DETERMINISTIC_UNARY_TAIL=PASS\n";

        // executed_suffix_tokens is compared above against the sum of
        // ambiguity-probe suffixes reconstructed independently from the
        // returned complete token paths. Any unary-tail scoring event would
        // make that equality fail.
        std::cout
            << "V2D_DETERMINISTIC_TAIL_SCORE_EVENTS=0\n";

        std::cout
            << "V2D_SEMANTIC_WINNER_MAPPING=PASS\n";

        std::cout
            << "V2D_REPEAT_PROBABILITY_MATCH=YES\n";

        std::cout
            << "V2D_RETAINED_FRONTIER_RESTORATION=PASS\n";

        std::cout
            << "V2D_RETAINED_SEQUENCE_LANES=1\n";

        std::cout
            << "V2D_MTP_USED=NO\n";

        std::cout
            << "V2D_VISION_USED=NO\n";

        std::cout
            << "V2D_NORMAL_DECODE_USED=NO\n";

        std::cout
            << "V2D_TRIE=PASS\n";

        return 0;
    }

    std::cerr
        << "FAIL: no candidate set produced the required D1 trie topology\n";

    return 1;
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr ||
        *artifact == '\0') {

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
