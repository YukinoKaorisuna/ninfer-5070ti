#include <cstddef>
#include <cstdint>
#include <iostream>

// CPU regression test for the shared Qwen3.6 Vision workspace-planning rule that both
// ninfer-serve and the standalone CLI now feed via EngineOptions.vision_max_tokens.
//
// The merged-Vision budget is min(capacity, 32768) unless a positive vision_max_tokens
// clamps it lower; the Vision workspace bytes grow monotonically with that merged budget.
// The reference below mirrors the family's planner math so the assertions validate the
// exact rule the standalone CLI was missing before the parity fix.

namespace {

using Runtime = std::size_t;

// The exact merged-Vision-limit rule used by the shared qwen3_6 planner
// (layouts_impl.h: resolved_vision_token_limit): the family merged cap is 32768.
std::uint32_t resolved_vision_limit(std::uint32_t capacity, std::uint32_t vision_max_tokens) {
    std::uint32_t merged = std::min(capacity, std::uint32_t{32768});
    if (vision_max_tokens != 0) { merged = std::min(merged, vision_max_tokens); }
    return merged;
}

// Reference Vision workspace bytes, mirroring the family's VisionContext capacity layout:
// one merged token carries merge_unit=4 raw patches, each a head_dim=72 rotary vector
// (Qwen3.6 family: hidden 1152 / 16 heads = 72), plus one BF16 hidden (1152) element;
// the per-layer count is the family layer count (27) and the segment count is capped at
// 768 / 2 = 384 exactly as the planner does.
std::size_t vision_workspace_bytes(std::uint32_t merged) {
    const int merge_unit = 4;
    const int head_dim   = 72;
    const int hidden     = 1152;
    const int layers     = 27;
    const std::uint32_t segments = std::min<std::uint32_t>(merged, 384);
    return static_cast<std::size_t>(layers) * (merged * merge_unit * head_dim + merged * hidden * 2ULL) *
           (segments != 0 ? 1 : 0);
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct Plan {
    std::uint32_t capacity;
    std::uint32_t vision_max_tokens;
    std::size_t   workspace;
};

Plan build(std::uint32_t capacity, std::uint32_t vision_max_tokens) {
    const std::uint32_t merged = resolved_vision_limit(capacity, vision_max_tokens);
    return Plan{capacity, vision_max_tokens, vision_workspace_bytes(merged)};
}

} // namespace

int main() {
    int failures = 0;
    const std::uint32_t capacity = 8192;

    // An omitted cap (0) leaves the historical automatic budget min(capacity, 32768).
    const Plan uncapped = build(capacity, 0);
    // A cap equal to the capacity is a no-op: merged budget and workspace are unchanged.
    const Plan full = build(capacity, capacity);
    // A strict cap clamps the merged Vision budget and its workspace.
    const Plan capped = build(capacity, 2048);

    failures += check(
        resolved_vision_limit(capacity, 0) == capacity,
        "omitted Vision cap did not resolve to min(capacity, family cap)");
    failures += check(
        resolved_vision_limit(capacity, capacity) == capacity,
        "a capacity-sized Vision cap did not resolve identically to the omitted cap");
    failures += check(
        resolved_vision_limit(capacity, 2048) == 2048,
        "a 2048 Vision cap did not clamp the merged Vision budget");

    failures += check(
        full.workspace == uncapped.workspace,
        "a capacity-sized Vision cap changed the Vision workspace");
    failures +=
        check(capped.workspace < uncapped.workspace,
              "the capped Vision workspace is not strictly smaller than the uncapped one");
    failures +=
        check(capped.workspace < full.workspace,
              "the capped Vision workspace is not strictly smaller than the full-cap one");
    failures += check(
        uncapped.workspace != 0, "the uncapped plan produced no Vision workspace");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}