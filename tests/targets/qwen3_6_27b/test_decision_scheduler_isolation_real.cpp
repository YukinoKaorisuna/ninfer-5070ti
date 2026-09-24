#include <ninfer/engine.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

constexpr std::uint32_t kLongOutputTokens = 1900;

ninfer::EngineOptions make_options(const char* artifact) {
    ninfer::EngineOptions options;

    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 2;
    options.max_pending_requests = 8;
    options.prefill_chunk = 896;
    options.kv_cache =
        ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend =
        ninfer::SpeculativeBackend::None;
    options.enable_vision = false;
    options.use_cuda_graph = false;

    return options;
}

std::vector<ninfer::TokenId> trunk(
    ninfer::TokenId token) {

    return std::vector<ninfer::TokenId>(
        63,
        token);
}

ninfer::RequestOptions long_request() {
    ninfer::RequestOptions options;

    options.execution.requested_output_tokens =
        kLongOutputTokens;

    options.execution.allow_prefix_reuse = false;

    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.top_k = 0;
    options.execution.sampling.top_p = 1.0F;
    options.execution.sampling.min_p = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.sampling.seed = 0;

    options.stop.include_model_defaults = false;
    options.output.raw = true;

    return options;
}

ninfer::RequestOptions short_request() {
    ninfer::RequestOptions options =
        long_request();

    options.execution.requested_output_tokens = 8;

    return options;
}

ninfer::DecisionFieldSpec decision_field() {
    ninfer::DecisionFieldSpec field;

    field.name = "scheduler-route";

    field.suffix_tokens = {
        198,
    };

    field.candidate_tokens = {
        198,
        846,
        5834,
    };

    return field;
}

template <class Predicate>
bool wait_until(
    ninfer::Engine& engine,
    Predicate predicate,
    std::chrono::milliseconds timeout) {

    const auto deadline =
        std::chrono::steady_clock::now() +
        timeout;

    while (std::chrono::steady_clock::now() <
           deadline) {

        const auto stats =
            engine.runtime_stats();

        if (predicate(stats)) {
            return true;
        }

        std::this_thread::sleep_for(1ms);
    }

    return false;
}

void print_stats(
    const char* label,
    const ninfer::RuntimeStats& stats) {

    std::cout
        << label
        << "_RUNNING="
        << stats.running_requests
        << "\n";

    std::cout
        << label
        << "_WAITING="
        << stats.waiting_requests
        << "\n";

    std::cout
        << label
        << "_DECODE_ROUNDS="
        << stats.decode_rounds
        << "\n";

    std::cout
        << label
        << "_COMMITTED_DECODE_TOKENS="
        << stats.committed_decode_tokens
        << "\n";
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr ||
        *artifact == '\0') {

        return 77;
    }

    try {
        ninfer::Engine engine(
            make_options(artifact));

        // Fill both execution lanes with independent ordinary requests.
        auto ordinary_a =
            engine.submit(
                engine.prepare_tokens(
                    trunk(198),
                    false),
                long_request());

        auto ordinary_b =
            engine.submit(
                engine.prepare_tokens(
                    trunk(846),
                    false),
                long_request());

        // Submit the decision immediately after A and B. Waiting for both
        // lanes to become active before submission made the original fixture
        // timing-sensitive because the 2-wide decode was fast enough for both
        // requests to finish before the queue-state polling began.
        auto decision =
            engine.submit_decision(
                engine.prepare_tokens(
                    trunk(5834),
                    true),
                {decision_field()});

        // Observe the actual full-engine queue state: both execution lanes are
        // occupied while the decision remains pending.
        const bool decision_queued =
            wait_until(
                engine,
                [](const ninfer::RuntimeStats& stats) {
                    return
                        stats.running_requests == 2 &&
                        stats.waiting_requests >= 1;
                },
                30s);

        if (!decision_queued) {
            print_stats(
                "QUEUE_TIMEOUT",
                engine.runtime_stats());

            throw std::runtime_error(
                "did not observe two occupied lanes with queued decision");
        }

        print_stats(
            "DECISION_QUEUED",
            engine.runtime_stats());

        // Abandon A. GenerationHandle destruction requests cancellation and
        // must release exactly one lane without disturbing B.
        ordinary_a = ninfer::GenerationHandle{};

        const ninfer::DecisionResult result =
            decision.wait();

        if (result.fields.size() != 1 ||
            result.fields[0].routing_probabilities.size() != 3) {

            throw std::runtime_error(
                "queued decision returned invalid result");
        }

        const auto after_decision =
            engine.runtime_stats();

        print_stats(
            "AFTER_DECISION",
            after_decision);

        // B should still be active: the decision must have progressed by using
        // the lane released by A rather than draining all unrelated work.
        if (after_decision.running_requests != 1 ||
            after_decision.waiting_requests != 0) {

            throw std::runtime_error(
                "decision did not isolate progress from unrelated active lane");
        }

        std::cout
            << "TWO_LANE_OCCUPANCY_OBSERVED=YES\n"
            << "DECISION_WAITED_BEHIND_OCCUPIED_LANES=YES\n"
            << "ABANDONED_REQUEST_RELEASED_LANE=YES\n"
            << "DECISION_PROGRESS_WITH_OTHER_LANE_ACTIVE=YES\n"
            << "SCHEDULER_ISOLATION=PASS\n";

        // Abandon the remaining long generation.
        ordinary_b = ninfer::GenerationHandle{};

        const bool drained =
            wait_until(
                engine,
                [](const ninfer::RuntimeStats& stats) {
                    return
                        stats.running_requests == 0 &&
                        stats.waiting_requests == 0;
                },
                30s);

        if (!drained) {
            print_stats(
                "DRAIN_TIMEOUT",
                engine.runtime_stats());

            throw std::runtime_error(
                "scheduler did not drain abandoned requests");
        }

        print_stats(
            "AFTER_ABANDON_DRAIN",
            engine.runtime_stats());

        std::cout
            << "ABANDONED_REQUESTS_DRAINED=PASS\n";

        // Engine must remain healthy after concurrent cancellation/decision.
        ninfer::GenerationResult recovery =
            engine.generate(
                engine.prepare_tokens(
                    trunk(198),
                    false),
                short_request());

        if (recovery.generated_token_ids.size() != 8) {
            throw std::runtime_error(
                "post-isolation recovery generation failed");
        }

        const bool final_idle =
            wait_until(
                engine,
                [](const ninfer::RuntimeStats& stats) {
                    return
                        stats.running_requests == 0 &&
                        stats.waiting_requests == 0;
                },
                10s);

        if (!final_idle) {
            throw std::runtime_error(
                "Engine not idle after recovery request");
        }

        print_stats(
            "FINAL",
            engine.runtime_stats());

        std::cout
            << "ENGINE_USABLE_AFTER_CONCURRENT_ABANDON=PASS\n"
            << "FINAL_RUNNING_REQUESTS=0\n"
            << "FINAL_WAITING_REQUESTS=0\n"
            << "MAX_CONCURRENCY_TESTED=2\n"
            << "R3B2A_SCHEDULER_ISOLATION=PASS\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
