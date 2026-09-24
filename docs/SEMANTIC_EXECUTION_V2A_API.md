# V2-A Semantic Execution C++ API

> **Status:** historical V2-A design record; implemented. Later V2-B/C/D1
> milestones supersede this document's statements about independent-only
> graphs and one-token backend restrictions. See
> [Constrained Decisions: current API and usage](CONSTRAINED_DECISIONS_USAGE.md)
> for the current contract.

Status: Accepted for implementation

## Scope

V2-A introduces the semantic API and generic finite-choice compiler boundary.

It does not change GPU/runtime execution.

The execution proof remains:

    semantic FiniteChoice
            +
    model presentation
            |
            v
    compile_decision_plan()
            |
            v
    existing DecisionFieldSpec[]
            |
            v
    existing executor / frontier / constrained scorer

V2-A does not add dependency execution, execution waves, tries,
restricted-row projection, FreeGeneration, MTP integration or Vision changes.

## SemanticValue

SemanticValue is a public value class.

Its storage representation is private.

The initial semantic kinds are:

    Boolean
    Integer
    Number
    String

The public interface should be conceptually equivalent to:

    enum class SemanticValueKind : std::uint8_t {
        Boolean,
        Integer,
        Number,
        String,
    };

    class SemanticValue final {
    public:
        static SemanticValue boolean(bool) noexcept;
        static SemanticValue integer(std::int64_t) noexcept;
        static SemanticValue number(double);
        static SemanticValue string(std::string);

        SemanticValueKind kind() const noexcept;

        bool boolean_value() const;
        std::int64_t integer_value() const;
        double number_value() const;
        const std::string& string_value() const;

        friend bool operator==(
            const SemanticValue&,
            const SemanticValue&) noexcept;

    private:
        using Storage =
            std::variant<
                bool,
                std::int64_t,
                double,
                std::string>;

        explicit SemanticValue(Storage value);

        Storage value_;
    };

The variant is private implementation and is not the caller-facing API.

Factories are preferred over overloaded converting constructors so bool,
integer, floating-point and string construction is explicit and unambiguous.

Semantic identity is type-sensitive:

    Integer(1) != Number(1.0)
    Boolean(true) != String("true")

Number values must be finite.

NaN and positive/negative infinity are invalid semantic values.

Negative zero is canonicalized to positive zero so equality and future hashing
remain deterministic.

SemanticValue deliberately provides no JSON conversion and no generic
model-text conversion.

## SemanticNodeId

Semantic node identity is strongly typed:

    struct SemanticNodeId {
        std::uint32_t value = 0;

        constexpr bool valid() const noexcept {
            return value != 0;
        }

        friend constexpr bool operator==(
            SemanticNodeId,
            SemanticNodeId) noexcept = default;
    };

Zero is invalid.

StructuredDecisionSchema assigns monotonically increasing IDs beginning at 1.

Callers do not choose numeric node IDs.

A strong type is used to prevent accidental interchange with token IDs, field
indices, sequence indices and other integer domains.

## FiniteChoice

The initial semantic node payload is:

    struct FiniteChoice {
        std::string label;
        std::vector<SemanticValue> choices;
    };

SemanticNodeId is node identity.

`label` is caller/diagnostic metadata. It is not identity and must not be
interpreted as a JSON property name, tool argument name or output-rendering
instruction.

Labels do not define execution ordering.

FiniteChoice contains no:

- token IDs;
- model suffixes;
- tokenizer paths;
- JSON syntax;
- protocol syntax;
- scorer geometry;
- candidate-count backend limit;
- tree/greedy selection;
- MTP controls;
- CUDA/runtime settings.

A FiniteChoice requires at least two semantic choices.

A one-value domain is conceptually a future KnownValue rather than a choice.

Duplicate SemanticValue entries are invalid.

No semantic candidate-count ceiling is imposed by FiniteChoice.

Current backend limits are checked during compilation.

## StructuredDecisionSchema

StructuredDecisionSchema becomes an opaque evolvable semantic graph builder.

Its public interface should be conceptually equivalent to:

    class StructuredDecisionSchema {
    public:
        StructuredDecisionSchema();
        ~StructuredDecisionSchema();

        StructuredDecisionSchema(
            const StructuredDecisionSchema&);
        StructuredDecisionSchema&
        operator=(const StructuredDecisionSchema&);

        StructuredDecisionSchema(
            StructuredDecisionSchema&&) noexcept;
        StructuredDecisionSchema&
        operator=(StructuredDecisionSchema&&) noexcept;

        SemanticNodeId add_finite_choice(
            FiniteChoice choice);

        bool empty() const noexcept;
        std::size_t node_count() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;

        friend class Engine;
    };

The implementation uses the selected flat semantic arena/DAG architecture.

The internal payload representation may use std::variant.

The public storage representation does not expose that variant.

This allows future node classes and dependency metadata to be added without
making a public variant alternative list the schema contract.

Schema copies preserve node IDs.

## V2-A graph semantics

V2-A contains independent FiniteChoice nodes only.

V2-A does not expose:

- root nodes;
- dependency edges;
- conditional predicates;
- rendering order;
- model-conditioning order.

All V2-A nodes are semantically independent.

Dependency APIs are deferred until V2-B defines their exact semantics.

The opaque schema boundary allows those capabilities to be added without
redesigning public node storage.

## Model presentation

Semantic values and their model-facing representations are separate.

The V2-A finite-choice presentation is:

    struct FiniteChoicePresentation {
        std::string continuation_prefix;
        std::vector<std::string> candidate_texts;
    };

`candidate_texts[i]` is the model-facing representation corresponding to
`FiniteChoice::choices[i]`.

These strings are presentation metadata, not semantic values and not output
rendering.

The presentation container is evolvable:

    class DecisionModelPresentation {
    public:
        DecisionModelPresentation();
        ~DecisionModelPresentation();

        DecisionModelPresentation(
            const DecisionModelPresentation&);
        DecisionModelPresentation&
        operator=(const DecisionModelPresentation&);

        DecisionModelPresentation(
            DecisionModelPresentation&&) noexcept;
        DecisionModelPresentation&
        operator=(DecisionModelPresentation&&) noexcept;

        void set_finite_choice(
            SemanticNodeId node,
            FiniteChoicePresentation presentation);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;

        friend class Engine;
    };

Compilation validates that every executable V2-A node has exactly one finite
presentation and that candidate_texts cardinality equals the semantic domain.

Presentation metadata contains no token IDs.

## Whole-path tokenization

For each finite candidate, compilation tokenizes:

    continuation_prefix + candidate_text

as one complete continuation.

It then derives the common token prefix and divergent candidate path.

Separately tokenizing prefix and candidate text is not permitted.

Current one-token divergence is a backend restriction only.

Future multi-token finite choices may lower to a trie without changing
FiniteChoice semantics.

## Compilation API

The generic V2-A compiler entry point becomes:

    CompiledDecisionPlan compile_decision_plan(
        const StructuredDecisionSchema& schema,
        const DecisionModelPresentation& presentation) const;

Compilation does not consume either object.

One semantic schema may therefore be compiled repeatedly or paired with
different presentation strategies.

CompiledDecisionPlan remains immutable, opaque and Engine-instance-bound.

## Legacy compatibility

DecisionFieldInput remains supported.

The legacy typed path must lower through the generic compiler.

Boolean lowering:

    semantic choices:
        Boolean(false)
        Boolean(true)

    presentation:
        continuation_prefix = existing suffix
        candidate_texts =
            "false"
            "true"

Enum lowering:

    semantic choices:
        String(value[0])
        String(value[1])
        ...

    presentation:
        continuation_prefix = existing suffix
        candidate_texts = existing enum strings

The pipeline is therefore:

    DecisionFieldInput
            |
            v
    FiniteChoice + DecisionModelPresentation
            |
            v
    generic compiler
            |
            v
    DecisionFieldSpec[]

There must not be a second typed tokenizer/compiler implementation.

The raw DecisionFieldSpec API remains available for diagnostics, parity tests
and lower-level callers.

## Backend compatibility validation

Semantic validation must not inherit backend qualification ceilings.

The existing backend may still reject at compile time:

- more than the currently supported field count;
- more than the currently supported candidate count;
- tokenization that does not yield the current one-token divergent branch;
- colliding candidate token paths;
- an executable suffix unsupported by the current runtime.

Those errors describe current compiled-backend compatibility.

They are not FiniteChoice semantic rules.

## Result API

V2-A deliberately retains DecisionResult.

The semantic result API is deferred.

V2-A proves the new semantic/presentation compiler boundary while leaving
execution and result consumption unchanged.

A later result milestone will introduce node-attributed typed values without
coupling that work to the schema/compiler transition.

The compiled plan may retain whatever private semantic-node mapping is useful
for that later result transition.

## Public variant decision

A raw public:

    std::variant<...>

is rejected for semantic nodes and for the semantic schema.

Reasons:

- NInfer currently uses variants internally rather than as its public semantic
  API;
- adding future node alternatives would become a caller-visible source type
  change;
- callers would become coupled to implementation payload alternatives;
- an opaque graph allows dependencies and new node classes to evolve cleanly.

SemanticValue may use a private std::variant because it is a compact value type
and does not justify per-value heap allocation.

## ABI decision

The repository recon found no NInfer SOVERSION/API-version contract or explicit
public binary-ABI policy that justifies heap-allocating every SemanticValue or
otherwise compromising the value model purely for stable object size.

Complex evolvable graph/plan objects nevertheless use opaque implementation
boundaries because this is a good API design independently of binary ABI.

## Performance preservation

The V2-A public API preserves all currently planned execution optimizations.

Reusable semantic/presentation prefix compilation remains possible.

Independent nodes can later form execution waves.

Wave width remains runtime-selected.

Current one-token scoring remains a valid backend.

Multi-token FiniteChoice can lower to token tries.

Finite domains remain explicit for future restricted-row LM-head projection.

FreeGeneration may later choose ordinary/MTP generation.

Grammar may later become a FreeGeneration backend.

Vision remains input context rather than schema structure.

Attention KV and recurrent/GDN memory remain runtime/backend concerns.

JSON, tool-call, SQL, source-code and other rendering remain adapter concerns.

## V2-A implementation acceptance

The first code milestone is accepted only if:

    new semantic API
        +
    separate model presentation
        +
    legacy bool/enum lowering
        ->
    one generic compiler
        ->
    identical existing DecisionFieldSpec execution

and existing runtime behavior remains unchanged.

Required regression properties include:

- current bool/enum selected values remain unchanged;
- candidate token metadata remains unchanged;
- probability distributions remain valid;
- compiled plans remain reusable across prompt frontiers;
- foreign/expired Engine plan rejection remains intact;
- ordinary decode remains unused;
- production runtime/CUDA code need not change.
