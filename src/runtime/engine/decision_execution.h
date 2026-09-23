#pragma once

#include <ninfer/types.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer::runtime {

// Internal backend lowering of semantic decisions.
//
// This type is deliberately not part of the public semantic API.
struct DecisionExecutionNode {
    std::vector<DecisionFieldSpec> variants;
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
