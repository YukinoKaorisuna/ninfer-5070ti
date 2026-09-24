#include "core/device.h"
#include "runtime/contract/types.h"
#include "targets/registry.h"

#include <ninfer/targets/qwen3_6/frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;

bool probability_vector_valid(
    const std::vector<float>& values,
    const char* label) {

    if (values.empty()) {
        std::cerr << "FAIL " << label << ": empty probability vector\n";
        return false;
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const float p = values[i];

        if (!std::isfinite(p) || p < 0.0F || p > 1.0F) {
            std::cerr
                << "FAIL " << label
                << ": invalid probability at " << i
                << " value=" << p
                << "\n";
            return false;
        }

        sum += static_cast<double>(p);
    }

    const double error = std::fabs(sum - 1.0);

    std::cout
        << label
        << "_SUM=" << std::setprecision(10)
        << sum
        << "\n";

    if (error > 1.0e-5) {
        std::cerr
            << "FAIL " << label
            << ": probability sum error=" << error
            << "\n";
        return false;
    }

    return true;
}

bool approximately_equal(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    float tolerance,
    float* max_error) {

    if (lhs.size() != rhs.size()) {
        return false;
    }

    float observed = 0.0F;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        observed =
            std::max(
                observed,
                std::fabs(lhs[i] - rhs[i]));
    }

    if (max_error != nullptr) {
        *max_error = observed;
    }

    return observed <= tolerance;
}

void print_probe(
    const char* name,
    const ninfer::targets::qwen3_6::DecisionProbeResult& result) {

    std::cout
        << name << "_FRONTIER="
        << result.frontier << "\n";

    std::cout
        << name << "_SUFFIX_TOKENS="
        << result.suffix_tokens << "\n";

    std::cout
        << name << "_WINNER_INDEX="
        << result.winner_index << "\n";

    std::cout
        << name << "_WINNER_TOKEN="
        << result.winner_token << "\n";

    std::cout
        << name << "_PROBABILITIES=";

    for (std::size_t i = 0;
         i < result.probabilities.size();
         ++i) {

        if (i != 0) {
            std::cout << ",";
        }

        std::cout
            << std::setprecision(9)
            << result.probabilities[i];
    }

    std::cout << "\n";

    std::cout
        << name << "_CAPTURE_MS="
        << std::fixed << std::setprecision(3)
        << result.capture_seconds * 1000.0
        << "\n";

    std::cout
        << name << "_SUFFIX_MS="
        << result.suffix_seconds * 1000.0
        << "\n";

    std::cout
        << name << "_SCORE_MS="
        << result.score_seconds * 1000.0
        << "\n";

    std::cout
        << name << "_RESTORE_MS="
        << result.restore_seconds * 1000.0
        << "\n";
}

int exercise_qwen38(const char* artifact) {
    ninfer::EngineOptions options;

    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency      = 1;
    options.prefill_chunk        = 896;
    options.kv_cache             =
        ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend  =
        ninfer::SpeculativeBackend::None;
    options.enable_vision        = false;
    options.use_cuda_graph       = false;

    ninfer::DeviceContext device(options.device);

    auto constructed =
        ninfer::targets::construct_target(
            options,
            device);

    auto* target =
        std::get_if<
            std::unique_ptr<
                ninfer::targets::Qwen3_6_27BInstance>>(
                    &constructed.active);

    if (target == nullptr || !*target) {
        std::cerr
            << "FAIL: artifact did not construct "
               "the Qwen3.6/3.8 27B target\n";
        return 1;
    }

    auto& instance = **target;
    auto& program  = *instance.program;

    std::cout
        << "MODEL_ID="
        << constructed.load.model_id
        << "\n";

    std::cout
        << "TARGET="
        << constructed.load.target
        << "\n";

    std::cout
        << "WEIGHTS_ID="
        << constructed.load.weights_id
        << "\n";

    // A small deterministic raw-token trunk using IDs already exercised
    // elsewhere by this target's real integration fixtures.
    //
    // The first sampled token from prefill is deliberately NOT part of the
    // decision frontier. The retained execution frontier represents the
    // completely executed prompt; decision suffixes branch from that point.
    // Frontier 63 deliberately places the two-token B probe across the
    // 64-token paged-KV boundary. A consumes position 63 only; B additionally
    // materializes position 64 in a second page. Repeating A after B proves
    // page entitlement/materialization/trim restoration as well as recurrent
    // state restoration.
    std::vector<TokenId> trunk(63, 198);

    auto prepared =
        instance.loaded->frontend.prepare_tokens(
            trunk,
            true);

    ninfer::runtime::ResolvedExecutionOptions execution;

    execution.sampling.temperature       = 0.0F;
    execution.sampling.top_k             = 0;
    execution.sampling.top_p             = 1.0F;
    execution.sampling.min_p             = 0.0F;
    execution.sampling.presence_penalty  = 0.0F;
    execution.sampling.frequency_penalty = 0.0F;
    execution.sampling.seed              = 0;

    execution.requested_output_tokens = 1;
    execution.allow_prefix_reuse      = false;

    auto base =
        program.plan_request_base(
            prepared,
            execution);

    auto plan =
        program.plan_request_for_lane(
            0,
            prepared,
            base);

    const auto summary = plan.summary();

    std::cout
        << "TRUNK_TOKENS="
        << trunk.size()
        << "\n";

    std::cout
        << "TRANSIENT_BYTES="
        << summary.transient_bytes
        << "\n";

    bool transient_active = false;

    try {
        ninfer::runtime::TransientRegion transient;

        if (summary.transient_bytes != 0) {
            instance.request_memory.activate(
                summary.transient_bytes,
                summary.transient_alignment);

            transient =
                instance.request_memory.region();

            transient_active = true;
        }

        auto step =
            program.start_prefill_lane(
                0,
                std::move(prepared),
                std::move(plan),
                transient);

        while (!step.complete) {
            step =
                program.advance_prefill_lane(0);
        }

        if (step.round.tokens.size() != 1) {
            std::cerr
                << "FAIL: prefill did not produce "
                   "exactly one licensed token\n";

            program.abort_lane(0);

            if (transient_active) {
                instance.request_memory.deactivate();
            }

            return 1;
        }

        // Terminal resolution makes lane 0 retained while preserving the
        // fully executed prompt frontier for later reuse.
        program.resolve_prefill_lane(
            0,
            true);

        if (transient_active) {
            instance.request_memory.deactivate();
            transient_active = false;
        }

    } catch (...) {
        program.abort_lane(0);

        if (transient_active) {
            instance.request_memory.deactivate();
        }

        throw;
    }

    if (!program.has_retained_lane(0)) {
        std::cerr
            << "FAIL: source prompt did not become retained\n";
        return 1;
    }

    std::cout << "RETAINED_AFTER_PREFILL=YES\n";

    // Candidate token IDs are ordinary licensed tokens.
    // M1-B is testing state reversibility here, not language semantics.
    const std::vector<TokenId> candidates{
        198,
        846,
        5834
    };

    const std::vector<TokenId> suffix_a{
        198
    };

    const std::vector<TokenId> suffix_b{
        5834,
        198
    };

    const auto a1 =
        program.decision_probe_lane(
            0,
            suffix_a,
            candidates);

    if (!program.has_retained_lane(0)) {
        std::cerr
            << "FAIL: lane lost retention after A1\n";
        return 1;
    }

    const auto b =
        program.decision_probe_lane(
            0,
            suffix_b,
            candidates);

    if (!program.has_retained_lane(0)) {
        std::cerr
            << "FAIL: lane lost retention after B\n";
        return 1;
    }

    const auto a2 =
        program.decision_probe_lane(
            0,
            suffix_a,
            candidates);

    if (!program.has_retained_lane(0)) {
        std::cerr
            << "FAIL: lane lost retention after A2\n";
        return 1;
    }

    print_probe("A1", a1);
    print_probe("B",  b);
    print_probe("A2", a2);

    if (!probability_vector_valid(
            a1.probabilities,
            "A1")) {
        return 1;
    }

    if (!probability_vector_valid(
            b.probabilities,
            "B")) {
        return 1;
    }

    if (!probability_vector_valid(
            a2.probabilities,
            "A2")) {
        return 1;
    }

    if (a1.frontier != a2.frontier ||
        a1.frontier != b.frontier) {

        std::cerr
            << "FAIL: decision frontier changed "
               "across probes\n";
        return 1;
    }

    if (a1.winner_index != a2.winner_index ||
        a1.winner_token != a2.winner_token) {

        std::cerr
            << "FAIL: repeated A probe changed winner\n";
        return 1;
    }

    float max_error = 0.0F;

    if (!approximately_equal(
            a1.probabilities,
            a2.probabilities,
            1.0e-6F,
            &max_error)) {

        std::cerr
            << "FAIL: repeated A probabilities changed; "
            << "max_error=" << max_error
            << "\n";

        return 1;
    }

    std::cout
        << "A_REPEAT_MAX_ABS_ERROR="
        << std::setprecision(10)
        << max_error
        << "\n";

    std::cout
        << "A_REPEAT_WINNER_MATCH=YES\n";

    std::cout
        << "A_REPEAT_PROBABILITY_MATCH=YES\n";

    std::cout
        << "FRONTIER_STABLE=YES\n";

    std::cout
        << "RETAINED_AFTER_ALL_PROBES="
        << (program.has_retained_lane(0)
                ? "YES"
                : "NO")
        << "\n";

    program.evict_retained_lane(0);

    std::cout
        << "M1B_REVERSIBILITY=PASS\n";

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
            << "skip: "
               "NINFER_QWEN3_8_27B_DECISION_WEIGHTS "
               "is not set\n";

        return 77;
    }

    try {
        return exercise_qwen38(artifact);

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
