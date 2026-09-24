#include "ninfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ninfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ninfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ninfer::Engine>);
static_assert(!std::is_copy_constructible_v<ninfer::Engine>);

int main() {
    const ninfer::EngineOptions options;

    ninfer::DecisionFieldResult field;
    field.routing_probabilities = {0.6F, 0.4F};

    if (field.routing_probabilities.size() != 2) {
        return 1;
    }

    return options.enable_vision ? 1 : 0;
}
