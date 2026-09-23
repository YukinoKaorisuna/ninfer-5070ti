#include "decision_execution.h"
#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/registry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <cmath>

namespace ninfer {
namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class DecisionHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual DecisionResult wait(const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        DecisionResult wait(const CancellationView& cancellation) override {
            return submission_.wait(cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive),
                                                     std::move(submission))) {}

    DecisionResult wait(const CancellationView& cancellation) {
        return state_->wait(cancellation);
    }

private:
    std::unique_ptr<Concept> state_;
};

DecisionHandle::DecisionHandle() noexcept                          = default;
DecisionHandle::~DecisionHandle()                                  = default;
DecisionHandle::DecisionHandle(DecisionHandle&&) noexcept          = default;
DecisionHandle& DecisionHandle::operator=(DecisionHandle&&) noexcept = default;

DecisionHandle::DecisionHandle(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DecisionHandle::operator bool() const noexcept {
    return impl_ != nullptr;
}

DecisionResult DecisionHandle::wait(const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("DecisionHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(cancellation);
}

class Engine::Impl {
public:
    using Executor9  = runtime::ConcurrentExecutor<targets::Qwen3_5_9BInstance>;
    using Executor27 = runtime::ConcurrentExecutor<targets::Qwen3_6_27BInstance>;
    using Executor35 = runtime::ConcurrentExecutor<targets::Qwen3_6_35BA3BInstance>;
    using Executor =
        std::variant<std::monostate, std::unique_ptr<Executor9>, std::unique_ptr<Executor27>,
                     std::unique_ptr<Executor35>>;

    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        auto constructed  = targets::construct_target(options, device);
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        executor          = std::visit(
            [&](auto& target_ptr) -> Executor {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                if constexpr (std::is_same_v<Instance, targets::Qwen3_5_9BInstance>) {
                    return std::make_unique<Executor9>(*target_ptr, device, options);
                } else if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                    return std::make_unique<Executor27>(*target_ptr, device, options);
                } else {
                    return std::make_unique<Executor35>(*target_ptr, device, options);
                }
            },
            active);
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        executor.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Executor executor;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;

            GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& executor) -> GenerationHandle {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                auto submission = executor->submit(std::move(prompt.impl_->value), prompt_summary,
                                                   prepare_seconds, std::move(resolved_options),
                                                   pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->executor);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    return submit(std::move(prompt), std::move(options)).wait(sink, cancellation);
}

SemanticValue::SemanticValue(Storage value)
    : value_(std::move(value)) {}

SemanticValue
SemanticValue::boolean(bool value) noexcept {
    return SemanticValue(
        Storage(std::in_place_type<bool>, value));
}

SemanticValue
SemanticValue::integer(std::int64_t value) noexcept {
    return SemanticValue(
        Storage(std::in_place_type<std::int64_t>, value));
}

SemanticValue
SemanticValue::number(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "semantic number must be finite");
    }

    // Canonicalize negative zero so equality and future hashing have one
    // stable representation.
    if (value == 0.0) {
        value = 0.0;
    }

    return SemanticValue(
        Storage(std::in_place_type<double>, value));
}

SemanticValue
SemanticValue::string(std::string value) {
    return SemanticValue(
        Storage(std::in_place_type<std::string>,
                std::move(value)));
}

SemanticValueKind
SemanticValue::kind() const noexcept {
    switch (value_.index()) {
    case 0:
        return SemanticValueKind::Boolean;
    case 1:
        return SemanticValueKind::Integer;
    case 2:
        return SemanticValueKind::Number;
    case 3:
        return SemanticValueKind::String;
    default:
        std::terminate();
    }
}

bool
SemanticValue::boolean_value() const {
    return std::get<bool>(value_);
}

std::int64_t
SemanticValue::integer_value() const {
    return std::get<std::int64_t>(value_);
}

double
SemanticValue::number_value() const {
    return std::get<double>(value_);
}

const std::string&
SemanticValue::string_value() const {
    return std::get<std::string>(value_);
}

bool
operator==(const SemanticValue& lhs,
           const SemanticValue& rhs) noexcept {
    return lhs.value_ == rhs.value_;
}

class StructuredDecisionSchema::Impl {
public:
    struct Node {
        SemanticNodeId id;
        FiniteChoice choice;
    };

    struct Dependency {
        SemanticNodeId parent;
        SemanticNodeId child;
    };

    std::vector<Node> nodes;
    std::vector<Dependency> dependencies;
};

StructuredDecisionSchema::StructuredDecisionSchema()
    : impl_(std::make_unique<Impl>()) {}

StructuredDecisionSchema::~StructuredDecisionSchema() = default;

StructuredDecisionSchema::StructuredDecisionSchema(
    const StructuredDecisionSchema& other)
    : impl_(other.impl_
                ? std::make_unique<Impl>(*other.impl_)
                : nullptr) {}

StructuredDecisionSchema&
StructuredDecisionSchema::operator=(
    const StructuredDecisionSchema& other) {

    if (this != &other) {
        impl_ = other.impl_
                    ? std::make_unique<Impl>(*other.impl_)
                    : nullptr;
    }

    return *this;
}

StructuredDecisionSchema::StructuredDecisionSchema(
    StructuredDecisionSchema&&) noexcept = default;

StructuredDecisionSchema&
StructuredDecisionSchema::operator=(
    StructuredDecisionSchema&&) noexcept = default;

SemanticNodeId
StructuredDecisionSchema::add_finite_choice(
    FiniteChoice choice) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "StructuredDecisionSchema is moved from");
    }

    if (choice.choices.size() < 2) {
        throw std::invalid_argument(
            "finite choice requires at least two semantic values");
    }

    for (std::size_t i = 0;
         i < choice.choices.size();
         ++i) {

        for (std::size_t j = 0;
             j < i;
             ++j) {

            if (choice.choices[i] ==
                choice.choices[j]) {

                throw std::invalid_argument(
                    "finite choice semantic values must be unique");
            }
        }
    }

    if (impl_->nodes.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {

        throw std::overflow_error(
            "semantic node ID space exhausted");
    }

    const SemanticNodeId id{
        static_cast<std::uint32_t>(
            impl_->nodes.size() + 1)
    };

    impl_->nodes.push_back(
        Impl::Node{
            id,
            std::move(choice)
        });

    return id;
}


void
StructuredDecisionSchema::add_dependency(
    SemanticNodeId parent,
    SemanticNodeId child) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "StructuredDecisionSchema is moved from");
    }

    const auto node_exists =
        [&](SemanticNodeId id) {

        return
            id.valid() &&
            id.value <= impl_->nodes.size() &&
            impl_->nodes[id.value - 1].id == id;
    };

    if (!node_exists(parent) ||
        !node_exists(child)) {

        throw std::invalid_argument(
            "semantic dependency references an unknown node");
    }

    if (parent == child) {
        throw std::invalid_argument(
            "semantic dependency cannot reference the same node");
    }

    for (const Impl::Dependency& dependency :
         impl_->dependencies) {

        if (dependency.parent == parent &&
            dependency.child == child) {

            throw std::invalid_argument(
                "semantic dependency already exists");
        }
    }

    std::vector<SemanticNodeId> pending{
        child
    };

    std::vector<std::uint8_t> visited(
        impl_->nodes.size() + 1,
        0);

    while (!pending.empty()) {
        const SemanticNodeId current =
            pending.back();

        pending.pop_back();

        if (current == parent) {
            throw std::invalid_argument(
                "semantic dependency would create a cycle");
        }

        if (visited[current.value] != 0) {
            continue;
        }

        visited[current.value] = 1;

        for (const Impl::Dependency& dependency :
             impl_->dependencies) {

            if (dependency.parent == current) {
                pending.push_back(
                    dependency.child);
            }
        }
    }

    impl_->dependencies.push_back(
        Impl::Dependency{
            parent,
            child
        });
}

bool
StructuredDecisionSchema::empty() const noexcept {
    return impl_ == nullptr ||
           impl_->nodes.empty();
}

std::size_t
StructuredDecisionSchema::node_count() const noexcept {
    return impl_ != nullptr
               ? impl_->nodes.size()
               : 0;
}

class DecisionModelPresentation::Impl {
public:
    struct Entry {
        SemanticNodeId node;
        FiniteChoicePresentation presentation;
    };

    struct DependencyEntry {
        SemanticNodeId parent;
        SemanticNodeId child;
        DependencyConditioningPresentation presentation;
    };

    std::vector<Entry> entries;
    std::vector<DependencyEntry> dependency_entries;
};

DecisionModelPresentation::DecisionModelPresentation()
    : impl_(std::make_unique<Impl>()) {}

DecisionModelPresentation::~DecisionModelPresentation() = default;

DecisionModelPresentation::DecisionModelPresentation(
    const DecisionModelPresentation& other)
    : impl_(other.impl_
                ? std::make_unique<Impl>(*other.impl_)
                : nullptr) {}

DecisionModelPresentation&
DecisionModelPresentation::operator=(
    const DecisionModelPresentation& other) {

    if (this != &other) {
        impl_ = other.impl_
                    ? std::make_unique<Impl>(*other.impl_)
                    : nullptr;
    }

    return *this;
}

DecisionModelPresentation::DecisionModelPresentation(
    DecisionModelPresentation&&) noexcept = default;

DecisionModelPresentation&
DecisionModelPresentation::operator=(
    DecisionModelPresentation&&) noexcept = default;

void
DecisionModelPresentation::set_finite_choice(
    SemanticNodeId node,
    FiniteChoicePresentation presentation) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "DecisionModelPresentation is moved from");
    }

    if (!node.valid()) {
        throw std::invalid_argument(
            "finite-choice presentation requires a valid semantic node ID");
    }

    for (Impl::Entry& entry : impl_->entries) {
        if (entry.node == node) {
            entry.presentation =
                std::move(presentation);
            return;
        }
    }

    impl_->entries.push_back(
        Impl::Entry{
            node,
            std::move(presentation)
        });
}

void
DecisionModelPresentation::set_dependency_conditioning(
    SemanticNodeId parent,
    SemanticNodeId child,
    DependencyConditioningPresentation presentation) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "DecisionModelPresentation is moved from");
    }

    if (!parent.valid() ||
        !child.valid() ||
        parent == child) {

        throw std::invalid_argument(
            "dependency conditioning requires distinct valid semantic node IDs");
    }

    for (Impl::DependencyEntry& entry :
         impl_->dependency_entries) {

        if (entry.parent == parent &&
            entry.child == child) {

            entry.presentation =
                std::move(presentation);

            return;
        }
    }

    impl_->dependency_entries.push_back(
        Impl::DependencyEntry{
            parent,
            child,
            std::move(presentation)
        });
}


namespace {

struct LegacyDecisionDefinition {
    StructuredDecisionSchema schema;
    DecisionModelPresentation presentation;
};

LegacyDecisionDefinition
lower_legacy_decision_fields(
    std::vector<DecisionFieldInput> fields) {

    LegacyDecisionDefinition definition;

    for (DecisionFieldInput& input : fields) {
        if (input.suffix.empty()) {
            throw std::invalid_argument(
                "decision field suffix must not be empty");
        }

        FiniteChoice choice;
        choice.label = input.name;

        FiniteChoicePresentation model;

        model.continuation_prefix =
            input.suffix;

        switch (input.type) {
        case DecisionFieldType::Boolean:
            if (!input.values.empty()) {
                throw std::invalid_argument(
                    "boolean decision fields must not provide enum values");
            }

            choice.choices.push_back(
                SemanticValue::boolean(false));

            choice.choices.push_back(
                SemanticValue::boolean(true));

            model.candidate_texts = {
                "false",
                "true",
            };
            break;

        case DecisionFieldType::Enum:
            if (input.values.size() < 2) {
                throw std::invalid_argument(
                    "enum decision field requires at least two values");
            }

            choice.choices.reserve(
                input.values.size());

            for (const std::string& value :
                 input.values) {

                choice.choices.push_back(
                    SemanticValue::string(value));
            }

            model.candidate_texts =
                input.values;
            break;
        }

        const SemanticNodeId node =
            definition.schema.add_finite_choice(
                std::move(choice));

        definition.presentation.set_finite_choice(
            node,
            std::move(model));
    }

    return definition;
}

bool
is_canonical_boolean_choice(
    const FiniteChoice& choice) {

    if (choice.choices.size() != 2) {
        return false;
    }

    const SemanticValue& first =
        choice.choices[0];

    const SemanticValue& second =
        choice.choices[1];

    return
        first.kind() ==
            SemanticValueKind::Boolean &&
        second.kind() ==
            SemanticValueKind::Boolean &&
        !first.boolean_value() &&
        second.boolean_value();
}


struct DecisionProgramProjection {
    std::uint64_t service_work = 0;
    std::uint64_t max_frontier_extension = 0;
};

void
validate_decision_field_variant(
    const DecisionFieldSpec& field) {

    if (field.name.empty()) {
        throw std::invalid_argument(
            "decision field name must not be empty");
    }

    if (field.suffix_tokens.empty()) {
        throw std::invalid_argument(
            "decision field suffix must not be empty");
    }

    for (const TokenId token :
         field.suffix_tokens) {

        if (token < 0) {
            throw std::invalid_argument(
                "decision suffix token must be non-negative");
        }
    }

    if (field.candidate_tokens.size() < 2 ||
        field.candidate_tokens.size() > 16) {

        throw std::invalid_argument(
            "decision field requires 2..16 candidate tokens");
    }

    for (std::size_t i = 0;
         i < field.candidate_tokens.size();
         ++i) {

        if (field.candidate_tokens[i] < 0) {
            throw std::invalid_argument(
                "decision candidate token must be non-negative");
        }

        for (std::size_t j = 0;
             j < i;
             ++j) {

            if (field.candidate_tokens[i] ==
                field.candidate_tokens[j]) {

                throw std::invalid_argument(
                    "decision candidate tokens must be unique");
            }
        }
    }

    if (!field.candidate_values.empty() &&
        field.candidate_values.size() !=
            field.candidate_tokens.size()) {

        throw std::invalid_argument(
            "decision candidate value/token metadata size mismatch");
    }
}

DecisionProgramProjection
validate_and_project_decision_program(
    const runtime::DecisionExecutionProgram& program) {

    if (program.nodes.empty() ||
        program.nodes.size() > 8) {

        throw std::invalid_argument(
            "decision request requires 1..8 execution nodes");
    }

    DecisionProgramProjection projection;

    for (std::size_t node_index = 0;
         node_index < program.nodes.size();
         ++node_index) {

        const runtime::DecisionExecutionNode& node =
            program.nodes[node_index];

        if (node.variants.empty()) {
            throw std::invalid_argument(
                "decision execution node has no variants");
        }

        if (node.parent_result_index) {
            if (*node.parent_result_index >=
                node_index) {

                throw std::invalid_argument(
                    "decision dependency parent must precede its child in the execution program");
            }

            const runtime::DecisionExecutionNode& parent =
                program.nodes[
                    *node.parent_result_index];

            if (parent.variants.size() != 1) {
                throw std::invalid_argument(
                    "V2-B parent execution node must have exactly one root variant");
            }

            const std::size_t parent_choices =
                parent.variants.front()
                    .candidate_tokens.size();

            if (node.variants.size() !=
                parent_choices) {

                throw std::invalid_argument(
                    "decision child variant count must match its parent candidate count");
            }

        } else if (node.variants.size() != 1) {
            throw std::invalid_argument(
                "independent decision node must contain exactly one execution variant");
        }

        const DecisionFieldSpec& first =
            node.variants.front();

        for (std::size_t prior = 0;
             prior < node_index;
             ++prior) {

            if (program.nodes[prior]
                    .variants.front().name ==
                first.name) {

                throw std::invalid_argument(
                    "decision field names must be unique");
            }
        }

        std::size_t max_suffix = 0;

        for (const DecisionFieldSpec& variant :
             node.variants) {

            validate_decision_field_variant(
                variant);

            if (variant.name != first.name ||
                variant.type != first.type ||
                variant.candidate_values !=
                    first.candidate_values ||
                variant.candidate_tokens.size() !=
                    first.candidate_tokens.size()) {

                throw std::invalid_argument(
                    "decision execution variants disagree on field metadata");
            }

            max_suffix =
                std::max(
                    max_suffix,
                    variant.suffix_tokens.size());
        }

        projection.service_work +=
            static_cast<std::uint64_t>(
                max_suffix);

        projection.max_frontier_extension =
            std::max(
                projection.max_frontier_extension,
                static_cast<std::uint64_t>(
                    max_suffix));
    }

    return projection;
}

runtime::DecisionExecutionProgram
make_independent_decision_program(
    std::vector<DecisionFieldSpec> fields) {

    runtime::DecisionExecutionProgram program;

    program.nodes.reserve(
        fields.size());

    for (DecisionFieldSpec& field :
         fields) {

        runtime::DecisionExecutionNode node;

        node.variants.push_back(
            std::move(field));

        program.nodes.push_back(
            std::move(node));
    }

    return program;
}

} // namespace

class CompiledDecisionPlan::Impl {
public:
    Impl(std::weak_ptr<const void> engine_identity,
         runtime::DecisionExecutionProgram compiled_program)
        : owner_engine(std::move(engine_identity)),
          program(std::move(compiled_program)) {}

    // Weak shared-ownership identity prevents the plan from retaining the
    // Engine/model while avoiding raw-address identity reuse.
    std::weak_ptr<const void> owner_engine;
    runtime::DecisionExecutionProgram program;
};

CompiledDecisionPlan::CompiledDecisionPlan() noexcept = default;
CompiledDecisionPlan::~CompiledDecisionPlan() = default;

CompiledDecisionPlan::CompiledDecisionPlan(
    const CompiledDecisionPlan&) noexcept = default;

CompiledDecisionPlan&
CompiledDecisionPlan::operator=(
    const CompiledDecisionPlan&) noexcept = default;

CompiledDecisionPlan::CompiledDecisionPlan(
    CompiledDecisionPlan&&) noexcept = default;

CompiledDecisionPlan&
CompiledDecisionPlan::operator=(
    CompiledDecisionPlan&&) noexcept = default;

CompiledDecisionPlan::CompiledDecisionPlan(
    std::shared_ptr<const Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CompiledDecisionPlan::operator bool() const noexcept {
    return impl_ != nullptr;
}

bool CompiledDecisionPlan::empty() const noexcept {
    return impl_ == nullptr ||
           impl_->program.empty();
}

std::size_t CompiledDecisionPlan::field_count() const noexcept {
    return impl_ != nullptr
               ? impl_->program.field_count()
               : 0;
}

CompiledDecisionPlan
Engine::compile_decision_plan(
    const StructuredDecisionSchema& schema,
    const DecisionModelPresentation& presentation) const {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "Engine is moved from");
    }

    if (schema.impl_ == nullptr) {
        throw std::invalid_argument(
            "StructuredDecisionSchema is moved from");
    }

    if (presentation.impl_ == nullptr) {
        throw std::invalid_argument(
            "DecisionModelPresentation is moved from");
    }

    const auto& nodes =
        schema.impl_->nodes;

    const auto& dependencies =
        schema.impl_->dependencies;

    if (nodes.empty() ||
        nodes.size() > 8) {

        throw std::invalid_argument(
            "decision backend currently requires 1..8 finite-choice nodes");
    }

    if (presentation.impl_->entries.size() !=
        nodes.size()) {

        throw std::invalid_argument(
            "decision presentation must cover exactly the semantic nodes");
    }

    const auto tokenize =
        [&](std::string_view value) {

        return std::visit(
            [&](const auto& target_ptr)
                -> std::vector<TokenId> {

                if (target_ptr == nullptr) {
                    throw std::logic_error(
                        "Engine target is not active");
                }

                return target_ptr->loaded->frontend
                    .tokenize_decision_text(value);
            },
            impl_->active);
    };

    const auto find_node =
        [&](SemanticNodeId id)
            -> const StructuredDecisionSchema::Impl::Node* {

        for (const auto& node :
             nodes) {

            if (node.id == id) {
                return &node;
            }
        }

        return nullptr;
    };

    const auto find_presentation =
        [&](SemanticNodeId id)
            -> const DecisionModelPresentation::Impl::Entry* {

        for (const auto& entry :
             presentation.impl_->entries) {

            if (entry.node == id) {
                return &entry;
            }
        }

        return nullptr;
    };

    const auto compile_field =
        [&](const FiniteChoice& choice,
            std::string_view continuation_prefix,
            const std::vector<std::string>& candidate_texts)
            -> DecisionFieldSpec {

        if (choice.label.empty()) {
            throw std::invalid_argument(
                "decision backend requires a non-empty finite-choice label");
        }

        if (choice.choices.size() < 2 ||
            choice.choices.size() > 16) {

            throw std::invalid_argument(
                "decision backend currently supports 2..16 choices per node");
        }

        if (candidate_texts.size() !=
            choice.choices.size()) {

            throw std::invalid_argument(
                "finite-choice presentation candidate count does not match semantic domain");
        }

        for (std::size_t i = 0;
             i < candidate_texts.size();
             ++i) {

            for (std::size_t j = 0;
                 j < i;
                 ++j) {

                if (candidate_texts[i] ==
                    candidate_texts[j]) {

                    throw std::invalid_argument(
                        "finite-choice presentation texts must be unique");
                }
            }
        }

        DecisionFieldSpec raw;

        raw.name = choice.label;

        raw.type =
            is_canonical_boolean_choice(choice)
                ? DecisionFieldType::Boolean
                : DecisionFieldType::Enum;

        std::vector<std::vector<TokenId>> paths;

        paths.reserve(
            candidate_texts.size());

        for (const std::string& candidate_text :
             candidate_texts) {

            std::string path_text;

            path_text.reserve(
                continuation_prefix.size() +
                candidate_text.size());

            path_text.append(
                continuation_prefix.data(),
                continuation_prefix.size());

            path_text.append(
                candidate_text);

            std::vector<TokenId> path_tokens =
                tokenize(path_text);

            if (path_tokens.empty()) {
                throw std::invalid_argument(
                    "decision presentation path tokenized to no tokens");
            }

            paths.push_back(
                std::move(path_tokens));
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

        if (common == 0) {
            throw std::invalid_argument(
                "decision presentation paths have no shared token prefix");
        }

        raw.suffix_tokens.assign(
            paths.front().begin(),
            paths.front().begin() +
                static_cast<std::ptrdiff_t>(
                    common));

        raw.candidate_values.reserve(
            candidate_texts.size());

        raw.candidate_tokens.reserve(
            candidate_texts.size());

        for (std::size_t choice_index = 0;
             choice_index <
                 candidate_texts.size();
             ++choice_index) {

            const std::vector<TokenId>& path_tokens =
                paths[choice_index];

            if (path_tokens.size() !=
                common + 1) {

                throw std::invalid_argument(
                    "decision finite choice requires a multi-token branch after whole-path tokenization in the current backend: " +
                    candidate_texts[choice_index]);
            }

            const TokenId token =
                path_tokens[common];

            if (std::find(
                    raw.candidate_tokens.begin(),
                    raw.candidate_tokens.end(),
                    token) !=
                raw.candidate_tokens.end()) {

                throw std::invalid_argument(
                    "decision presentation choices must produce distinct one-token branches");
            }

            raw.candidate_values.push_back(
                candidate_texts[
                    choice_index]);

            raw.candidate_tokens.push_back(
                token);
        }

        return raw;
    };

    for (std::size_t i = 0;
         i < nodes.size();
         ++i) {

        if (nodes[i].choice.label.empty()) {
            throw std::invalid_argument(
                "decision backend requires a non-empty finite-choice label");
        }

        for (std::size_t j = 0;
             j < i;
             ++j) {

            if (nodes[i].choice.label ==
                nodes[j].choice.label) {

                throw std::invalid_argument(
                    "decision backend requires unique finite-choice labels");
            }
        }

        if (find_presentation(
                nodes[i].id) == nullptr) {

            throw std::invalid_argument(
                "decision presentation is missing a semantic node");
        }
    }

    runtime::DecisionExecutionProgram program;

    if (dependencies.empty()) {
        if (!presentation.impl_->
                 dependency_entries.empty()) {

            throw std::invalid_argument(
                "dependency conditioning was supplied for a schema with no semantic dependencies");
        }

        program.nodes.reserve(
            nodes.size());

        for (const auto& node :
             nodes) {

            const auto* model =
                find_presentation(
                    node.id);

            runtime::DecisionExecutionNode
                execution_node;

            execution_node.variants.push_back(
                compile_field(
                    node.choice,
                    model->presentation
                        .continuation_prefix,
                    model->presentation
                        .candidate_texts));

            program.nodes.push_back(
                std::move(
                    execution_node));
        }

    } else {
        if (nodes.size() != 2 ||
            dependencies.size() != 1) {

            throw std::invalid_argument(
                "V2-B backend currently supports exactly one root-to-child dependency across two finite-choice nodes");
        }

        if (presentation.impl_->
                dependency_entries.size() != 1) {

            throw std::invalid_argument(
                "V2-B dependency requires exactly one conditioning presentation");
        }

        const auto& dependency =
            dependencies.front();

        const auto* root =
            find_node(
                dependency.parent);

        const auto* child =
            find_node(
                dependency.child);

        if (root == nullptr ||
            child == nullptr) {

            throw std::logic_error(
                "compiled semantic dependency references an unknown node");
        }

        const auto* root_model =
            find_presentation(
                root->id);

        const auto* child_model =
            find_presentation(
                child->id);

        const DecisionModelPresentation::Impl::
            DependencyEntry* conditioning =
                nullptr;

        for (const auto& entry :
             presentation.impl_->
                 dependency_entries) {

            if (entry.parent ==
                    dependency.parent &&
                entry.child ==
                    dependency.child) {

                conditioning = &entry;
                break;
            }
        }

        if (conditioning == nullptr) {
            throw std::invalid_argument(
                "decision presentation is missing dependency conditioning");
        }

        if (conditioning->presentation
                .selected_choice_texts.size() !=
            root->choice.choices.size()) {

            throw std::invalid_argument(
                "dependency conditioning count must match the parent semantic choice count");
        }

        runtime::DecisionExecutionNode
            root_execution;

        root_execution.variants.push_back(
            compile_field(
                root->choice,
                root_model->presentation
                    .continuation_prefix,
                root_model->presentation
                    .candidate_texts));

        program.nodes.push_back(
            std::move(root_execution));

        runtime::DecisionExecutionNode
            child_execution;

        child_execution.parent_result_index =
            std::size_t{0};

        child_execution.variants.reserve(
            root->choice.choices.size());

        for (std::size_t parent_choice = 0;
             parent_choice <
                 root->choice.choices.size();
             ++parent_choice) {

            const std::string&
                selected_conditioning =
                    conditioning->presentation
                        .selected_choice_texts[
                            parent_choice];

            std::string continuation;

            continuation.reserve(
                selected_conditioning.size() +
                child_model->presentation
                    .continuation_prefix.size());

            continuation.append(
                selected_conditioning);

            continuation.append(
                child_model->presentation
                    .continuation_prefix);

            child_execution.variants.push_back(
                compile_field(
                    child->choice,
                    continuation,
                    child_model->presentation
                        .candidate_texts));
        }

        program.nodes.push_back(
            std::move(child_execution));
    }

    (void)validate_and_project_decision_program(
        program);

    return CompiledDecisionPlan(
        std::make_shared<
            const CompiledDecisionPlan::Impl>(
                std::weak_ptr<const void>(
                    std::shared_ptr<const void>(
                        impl_)),
                std::move(program)));
}

DecisionHandle
Engine::submit_decision(
    PreparedPrompt prompt,
    const CompiledDecisionPlan& plan,
    std::chrono::steady_clock::time_point pending_deadline) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "Engine is moved from");
    }

    if (prompt.impl_ == nullptr) {
        throw std::invalid_argument(
            "PreparedPrompt is empty");
    }

    if (plan.impl_ == nullptr ||
        plan.impl_->program.empty()) {

        throw std::invalid_argument(
            "CompiledDecisionPlan is empty");
    }

    const std::shared_ptr<const void> plan_owner =
        plan.impl_->owner_engine.lock();

    const std::shared_ptr<const void> current_owner =
        impl_;

    const bool same_engine_identity =
        plan_owner &&
        !plan_owner.owner_before(current_owner) &&
        !current_owner.owner_before(plan_owner);

    if (!same_engine_identity) {
        throw std::invalid_argument(
            "CompiledDecisionPlan belongs to a different or expired Engine instance");
    }

    const DecisionProgramProjection projection =
        validate_and_project_decision_program(
            plan.impl_->program);

    constexpr std::uint64_t
        kDecisionPrefillOutputWork = 1;

    if (projection.service_work == 0 ||
        projection.service_work >
            static_cast<std::uint64_t>(
                std::numeric_limits<
                    std::uint32_t>::max()) -
                kDecisionPrefillOutputWork) {

        throw std::invalid_argument(
            "decision projected work is outside supported bounds");
    }

    const std::uint64_t scheduler_work =
        projection.service_work +
        kDecisionPrefillOutputWork;

    const PromptSummary prompt_summary =
        prompt.impl_->summary;

    if (prompt_summary.prompt_tokens >
            impl_->options.max_context ||
        projection.max_frontier_extension >
            static_cast<std::uint64_t>(
                impl_->options.max_context -
                prompt_summary.prompt_tokens)) {

        throw RequestError(
            RequestErrorKind::
                ContextLengthExceeded,
            context_capacity_error(
                static_cast<std::uint32_t>(
                    prompt_summary.prompt_tokens +
                    std::min<std::uint64_t>(
                        projection.max_frontier_extension,
                        std::numeric_limits<
                            std::uint32_t>::max())),
                impl_->options.max_context));
    }

    RequestOptions planning;

    planning.execution.requested_output_tokens =
        static_cast<std::uint32_t>(
            scheduler_work);

    planning.execution.allow_prefix_reuse =
        true;

    runtime::ResolvedRequestOptions resolved =
        resolve_request_options(
            impl_->sampling_defaults,
            prompt.impl_->sampling_mode,
            std::move(planning));

    const double prepare_seconds =
        prompt.impl_->prepare.seconds;

    return std::visit(
        [&](auto& executor)
            -> DecisionHandle {

            using Executor =
                std::remove_cvref_t<
                    decltype(executor)>;

            if constexpr (
                std::is_same_v<
                    Executor,
                    std::monostate>) {

                throw std::logic_error(
                    "concurrent Engine executor is unavailable");

            } else {
                auto submission =
                    executor->submit_decision(
                        std::move(
                            prompt.impl_->value),
                        prompt_summary,
                        prepare_seconds,
                        std::move(resolved),
                        plan.impl_->program,
                        pending_deadline);

                return DecisionHandle(
                    std::make_unique<
                        DecisionHandle::Impl>(
                            impl_,
                            std::move(
                                submission)));
            }
        },
        impl_->executor);
}

DecisionResult
Engine::decide(
    PreparedPrompt prompt,
    const CompiledDecisionPlan& plan,
    const CancellationView& cancellation) {

    return submit_decision(
               std::move(prompt),
               plan)
        .wait(cancellation);
}

DecisionHandle
Engine::submit_decision(
    PreparedPrompt prompt,
    std::vector<DecisionFieldInput> fields,
    std::chrono::steady_clock::time_point pending_deadline) {

    LegacyDecisionDefinition definition =
        lower_legacy_decision_fields(
            std::move(fields));

    CompiledDecisionPlan plan =
        compile_decision_plan(
            definition.schema,
            definition.presentation);

    return submit_decision(
        std::move(prompt),
        plan,
        pending_deadline);
}

DecisionResult
Engine::decide(
    PreparedPrompt prompt,
    std::vector<DecisionFieldInput> fields,
    const CancellationView& cancellation) {

    LegacyDecisionDefinition definition =
        lower_legacy_decision_fields(
            std::move(fields));

    CompiledDecisionPlan plan =
        compile_decision_plan(
            definition.schema,
            definition.presentation);

    return decide(
        std::move(prompt),
        plan,
        cancellation);
}

DecisionHandle
Engine::submit_decision(PreparedPrompt prompt, std::vector<DecisionFieldSpec> fields,
                        std::chrono::steady_clock::time_point pending_deadline) {

    if (impl_ == nullptr) {
        throw std::logic_error(
            "Engine is moved from");
    }

    if (prompt.impl_ == nullptr) {
        throw std::invalid_argument(
            "PreparedPrompt is empty");
    }

    runtime::DecisionExecutionProgram program =
        make_independent_decision_program(
            std::move(fields));

    const DecisionProgramProjection projection =
        validate_and_project_decision_program(
            program);

    constexpr std::uint64_t
        kDecisionPrefillOutputWork = 1;

    if (projection.service_work == 0 ||
        projection.service_work >
            static_cast<std::uint64_t>(
                std::numeric_limits<
                    std::uint32_t>::max()) -
                kDecisionPrefillOutputWork) {

        throw std::invalid_argument(
            "decision projected work is outside supported bounds");
    }

    const std::uint64_t scheduler_work =
        projection.service_work +
        kDecisionPrefillOutputWork;

    const PromptSummary prompt_summary =
        prompt.impl_->summary;

    if (prompt_summary.prompt_tokens >
            impl_->options.max_context ||
        projection.max_frontier_extension >
            static_cast<std::uint64_t>(
                impl_->options.max_context -
                prompt_summary.prompt_tokens)) {

        throw RequestError(
            RequestErrorKind::
                ContextLengthExceeded,
            context_capacity_error(
                static_cast<std::uint32_t>(
                    prompt_summary.prompt_tokens +
                    std::min<std::uint64_t>(
                        projection.max_frontier_extension,
                        std::numeric_limits<
                            std::uint32_t>::max())),
                impl_->options.max_context));
    }

    RequestOptions planning;

    planning.execution.requested_output_tokens =
        static_cast<std::uint32_t>(
            scheduler_work);

    planning.execution.allow_prefix_reuse =
        true;

    runtime::ResolvedRequestOptions resolved =
        resolve_request_options(
            impl_->sampling_defaults,
            prompt.impl_->sampling_mode,
            std::move(planning));

    const double prepare_seconds =
        prompt.impl_->prepare.seconds;

    return std::visit(
        [&](auto& executor)
            -> DecisionHandle {

            using Executor =
                std::remove_cvref_t<
                    decltype(executor)>;

            if constexpr (
                std::is_same_v<
                    Executor,
                    std::monostate>) {

                throw std::logic_error(
                    "concurrent Engine executor is unavailable");

            } else {
                auto submission =
                    executor->submit_decision(
                        std::move(
                            prompt.impl_->value),
                        prompt_summary,
                        prepare_seconds,
                        std::move(resolved),
                        std::move(program),
                        pending_deadline);

                return DecisionHandle(
                    std::make_unique<
                        DecisionHandle::Impl>(
                            impl_,
                            std::move(
                                submission)));
            }
        },
        impl_->executor);
}

DecisionResult
Engine::decide(PreparedPrompt prompt, std::vector<DecisionFieldSpec> fields,
               const CancellationView& cancellation) {
    return submit_decision(std::move(prompt), std::move(fields)).wait(cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> MemorySummary {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->memory_summary();
            }
        },
        impl_->executor);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> RuntimeStats {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->runtime_stats();
            }
        },
        impl_->executor);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (!std::is_same_v<Executor, std::monostate>) {
                executor->reset_memory_peaks();
            }
        },
        impl_->executor);
}

} // namespace ninfer
