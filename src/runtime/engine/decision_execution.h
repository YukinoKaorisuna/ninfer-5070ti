#pragma once

#include <ninfer/types.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Internal backend lowering of semantic decisions.
//
// This type is deliberately not part of the public semantic API.

// Private V2-D1 lowering representation.
//
// A probe is one semantic ambiguity point. suffix_tokens is the complete
// deterministic traversal from the retained decision frontier to that
// ambiguity point. candidate_tokens are the legal outgoing token edges.
// descendant_candidate_indices maps each outgoing edge to the semantic leaves
// reachable beneath that edge.
struct DecisionTrieProbe {
    std::vector<TokenId> suffix_tokens;
    std::vector<TokenId> candidate_tokens;
    std::vector<std::vector<std::uint32_t>> descendant_candidate_indices;
};

struct DecisionTriePlan {
    std::vector<std::vector<TokenId>> candidate_token_paths;
    std::vector<DecisionTrieProbe> probes;
};

// Ordinary depth-1 callers remain source-compatible through the
// DecisionFieldSpec base. A trie plan is present only for a multi-token
// compiled semantic finite choice.
struct DecisionExecutionVariant : DecisionFieldSpec {
    std::optional<DecisionTriePlan> trie_plan;

    DecisionExecutionVariant() = default;

    DecisionExecutionVariant(const DecisionFieldSpec& field)
        : DecisionFieldSpec(field) {}

    DecisionExecutionVariant(DecisionFieldSpec&& field)
        : DecisionFieldSpec(std::move(field)) {}
};

struct DecisionExecutionNode {
    std::vector<DecisionExecutionVariant> variants;
    std::optional<std::size_t> parent_result_index;
};

struct DecisionExecutionProgram {
    std::vector<DecisionExecutionNode> nodes;

    [[nodiscard]] bool empty() const noexcept {
        return nodes.empty();
    }

    [[nodiscard]] std::size_t field_count() const noexcept {
        return nodes.size();
    }
};

} // namespace ninfer::runtime
