#include <ninfer/engine.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::DecisionFieldSpec;
using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::GenerationResult;
using ninfer::PrefixReusePath;
using ninfer::RequestOptions;
using ninfer::SpeculativeBackend;
using ninfer::TokenId;

constexpr std::uint32_t kContinuationTokens = 8;

EngineOptions make_options(const char* artifact) {
    EngineOptions options;

    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk = 896;
    options.kv_cache =
        ninfer::KvCacheStorage::Int4Group64;
    options.speculative.backend =
        SpeculativeBackend::None;
    options.enable_vision = false;
    options.use_cuda_graph = false;

    return options;
}

RequestOptions greedy_request() {
    RequestOptions request;

    request.execution.requested_output_tokens =
        kContinuationTokens;

    request.execution.allow_prefix_reuse = true;

    request.execution.sampling.temperature = 0.0F;
    request.execution.sampling.top_k = 0;
    request.execution.sampling.top_p = 1.0F;
    request.execution.sampling.min_p = 0.0F;
    request.execution.sampling.presence_penalty = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.sampling.seed = 0;

    // We need a fixed observation length rather than model-default EOS.
    request.stop.include_model_defaults = false;
    request.output.raw = true;

    return request;
}

std::vector<TokenId> make_trunk() {
    return std::vector<TokenId>(63, 198);
}

void print_tokens(
    const char* label,
    const std::vector<TokenId>& tokens) {

    std::cout << label << "=";

    for (std::size_t i = 0;
         i < tokens.size();
         ++i) {

        if (i != 0) {
            std::cout << ",";
        }

        std::cout << tokens[i];
    }

    std::cout << "\n";
}

GenerationResult ordinary_continuation(
    Engine& engine,
    const char* label) {

    auto prompt =
        engine.prepare_tokens(
            make_trunk(),
            true);

    GenerationResult result =
        engine.generate(
            std::move(prompt),
            greedy_request());

    print_tokens(
        label,
        result.generated_token_ids);

    std::cout
        << label
        << "_COUNT="
        << result.generated_token_ids.size()
        << "\n";

    std::cout
        << label
        << "_REUSED_PROMPT_TOKENS="
        << result.reused_prompt_tokens
        << "\n";

    if (result.generated_token_ids.size() !=
        kContinuationTokens) {

        throw std::runtime_error(
            std::string(label) +
            " did not produce the requested fixed continuation");
    }

    return result;
}

DecisionFieldSpec valid_field() {
    DecisionFieldSpec field;

    field.name = "restore-cross-page";

    // Frontier 63 + two suffix tokens crosses the 64-token page boundary.
    field.suffix_tokens = {
        5834,
        198,
    };

    field.candidate_tokens = {
        198,
        846,
        5834,
    };

    return field;
}

bool same_continuation(
    const GenerationResult& lhs,
    const GenerationResult& rhs) {

    return lhs.generated_token_ids ==
        rhs.generated_token_ids;
}

void require_idle(
    const Engine& engine,
    const char* label) {

    const auto stats =
        engine.runtime_stats();

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

    if (stats.running_requests != 0 ||
        stats.waiting_requests != 0) {

        throw std::runtime_error(
            std::string(label) +
            " left scheduler state active");
    }
}

} // namespace

int main() {
    const char* artifact =
        std::getenv(
            "NINFER_QWEN3_8_27B_DECISION_WEIGHTS");

    if (artifact == nullptr ||
        *artifact == '\0') {

        std::cout
            << "skip: model artifact env is not set\n";

        return 77;
    }

    try {
        GenerationResult control;

        // ====================================================
        // CONTROL: fresh Engine, ordinary deterministic path.
        // ====================================================

        {
            Engine engine(
                make_options(artifact));

            const auto capacity =
                engine.decision_capacity();

            std::cout
                << "REAL_SCORER_WORKSPACE_BYTES="
                << capacity
                       .scorer_workspace_capacity_bytes
                << "\n";

            std::cout
                << "REAL_SCORER_WORKSPACE_MAX_K="
                << capacity
                       .scorer_workspace_max_candidates
                << "\n";

            if (!capacity.executable ||
                capacity.scorer_workspace_max_candidates < 3) {

                throw std::runtime_error(
                    "real Engine reports no usable decision capacity");
            }

            control =
                ordinary_continuation(
                    engine,
                    "CONTROL_TOKENS");
        }

        // ====================================================
        // SUCCESS:
        // decision -> restored retained frontier -> ordinary
        // generation must reuse that state and match control.
        // ====================================================

        {
            Engine engine(
                make_options(artifact));

            auto prompt =
                engine.prepare_tokens(
                    make_trunk(),
                    true);

            const auto decision =
                engine.decide(
                    std::move(prompt),
                    {valid_field()});

            if (decision.fields.size() != 1 ||
                decision.fields[0].frontier != 63 ||
                decision.fields[0].suffix_tokens != 2) {

                throw std::runtime_error(
                    "successful decision fixture did not execute expected cross-page probe");
            }

            const auto before_generation =
                engine.runtime_stats();

            if (before_generation.committed_decode_tokens != 0 ||
                before_generation.decode_rounds != 0 ||
                before_generation.running_requests != 0 ||
                before_generation.waiting_requests != 0) {

                throw std::runtime_error(
                    "decision entered ordinary decode or remained scheduled");
            }

            const GenerationResult after =
                ordinary_continuation(
                    engine,
                    "POST_DECISION_TOKENS");

            if (after.reused_prompt_tokens == 0 ||
                after.prefix_reuse_path ==
                    PrefixReusePath::FullReset) {

                throw std::runtime_error(
                    "post-decision continuation did not reuse the restored retained frontier");
            }

            if (!same_continuation(
                    control,
                    after)) {

                throw std::runtime_error(
                    "post-decision ordinary continuation differs from clean control");
            }

            require_idle(
                engine,
                "SUCCESS_FINAL");

            std::cout
                << "RESTORED_PREFIX_ACTUALLY_REUSED=YES\n"
                << "SUCCESS_CONTROL_CONTINUATION_MATCH=YES\n"
                << "SUCCESS_DIFFERENTIAL_NONINTERFERENCE=PASS\n";
        }

        // ====================================================
        // PRE-ADMISSION VALIDATION FAILURE:
        // negative TokenId is rejected by Engine validation.
        // No prefill/lane mutation is permitted.
        // ====================================================

        {
            Engine engine(
                make_options(artifact));

            auto prompt =
                engine.prepare_tokens(
                    make_trunk(),
                    true);

            DecisionFieldSpec invalid =
                valid_field();

            invalid.name =
                "negative-preflight";

            invalid.candidate_tokens = {
                198,
                -1,
            };

            bool rejected = false;

            try {
                (void)engine.decide(
                    std::move(prompt),
                    {std::move(invalid)});

            } catch (const std::invalid_argument&) {
                rejected = true;
            }

            if (!rejected) {
                throw std::runtime_error(
                    "negative TokenId was not rejected");
            }

            const auto after_rejection =
                engine.runtime_stats();

            if (after_rejection.computed_prefill_tokens != 0 ||
                after_rejection.committed_decode_tokens != 0 ||
                after_rejection.running_requests != 0 ||
                after_rejection.waiting_requests != 0) {

                throw std::runtime_error(
                    "pre-admission validation failure mutated scheduler/runtime state");
            }

            const GenerationResult recovery =
                ordinary_continuation(
                    engine,
                    "PREVALIDATION_RECOVERY_TOKENS");

            if (recovery.reused_prompt_tokens != 0 ||
                recovery.prefix_reuse_path !=
                    PrefixReusePath::FullReset) {

                throw std::runtime_error(
                    "pre-admission validation unexpectedly left reusable target state");
            }

            if (!same_continuation(
                    control,
                    recovery)) {

                throw std::runtime_error(
                    "pre-admission validation failure changed ordinary continuation");
            }

            require_idle(
                engine,
                "PREVALIDATION_FINAL");

            std::cout
                << "NEGATIVE_TOKEN_REJECTION=PASS\n"
                << "PREVALIDATION_STATE_UNTOUCHED=PASS\n"
                << "PREVALIDATION_CONTROL_CONTINUATION_MATCH=YES\n";
        }

        // ====================================================
        // POST-ADMISSION TARGET FAILURE:
        //
        // Engine-level raw-token validation permits non-negative
        // TokenId values because the Engine is model-agnostic.
        // INT32_MAX therefore reaches the target after prefill.
        //
        // The Qwen target rejects it against token_domain and
        // run_decision_request() must abort the lane.
        // ====================================================

        {
            Engine engine(
                make_options(artifact));

            auto prompt =
                engine.prepare_tokens(
                    make_trunk(),
                    true);

            DecisionFieldSpec invalid =
                valid_field();

            invalid.name =
                "target-domain-failure";

            invalid.candidate_tokens = {
                198,
                std::numeric_limits<TokenId>::max(),
            };

            bool rejected = false;
            std::string failure_text;

            try {
                (void)engine.decide(
                    std::move(prompt),
                    {std::move(invalid)});

            } catch (const std::exception& error) {
                rejected = true;
                failure_text = error.what();
            }

            if (!rejected) {
                throw std::runtime_error(
                    "out-of-domain target TokenId was not rejected");
            }

            std::cout
                << "TARGET_FAILURE_TEXT="
                << failure_text
                << "\n";

            const auto after_failure =
                engine.runtime_stats();

            if (after_failure.computed_prefill_tokens == 0) {
                throw std::runtime_error(
                    "target-domain fixture did not reach post-admission execution");
            }

            if (after_failure.committed_decode_tokens != 0 ||
                after_failure.running_requests != 0 ||
                after_failure.waiting_requests != 0) {

                throw std::runtime_error(
                    "failed decision left decode or scheduler state active");
            }

            const GenerationResult recovery =
                ordinary_continuation(
                    engine,
                    "TARGET_FAILURE_RECOVERY_TOKENS");

            // Failed decision lane must have been aborted, not retained.
            if (recovery.reused_prompt_tokens != 0 ||
                recovery.prefix_reuse_path !=
                    PrefixReusePath::FullReset) {

                throw std::runtime_error(
                    "failed decision lane was reused after abort contract");
            }

            if (!same_continuation(
                    control,
                    recovery)) {

                throw std::runtime_error(
                    "Engine continuation changed after failed decision lane abort");
            }

            require_idle(
                engine,
                "TARGET_FAILURE_FINAL");

            std::cout
                << "OUT_OF_DOMAIN_TARGET_TOKEN_REJECTION=PASS\n"
                << "POST_ADMISSION_FAILURE_REACHED=YES\n"
                << "FAILED_DECISION_LANE_ABORTED=PASS\n"
                << "FAILED_LANE_NOT_REUSED=PASS\n"
                << "ENGINE_USABLE_AFTER_DECISION_FAILURE=PASS\n"
                << "FAILURE_RECOVERY_CONTROL_CONTINUATION_MATCH=YES\n";
        }

        std::cout
            << "DIFFERENTIAL_CONTINUATION_TOKENS="
            << kContinuationTokens
            << "\n"
            << "R3B1_RESTORATION_NONINTERFERENCE=PASS\n"
            << "R3B1_VALIDATION_NONINTERFERENCE=PASS\n"
            << "R3B1_FAILURE_LANE_ISOLATION=PASS\n"
            << "R3B1_REAL_QWEN=PASS\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "FAIL EXCEPTION: "
            << error.what()
            << "\n";

        return 1;
    }
}
