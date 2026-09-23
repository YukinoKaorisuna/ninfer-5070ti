# Constrained Decisions

## Purpose

Add a finite-choice decision execution path to NInfer.

Unlike normal autoregressive JSON generation, a constrained decision scores only
explicitly allowed values and assembles the typed result outside the model.

The initial target is Qwen3.8-27B on RTX 5080 16 GB.

## Design principles

1. Do not allocate persistent model state per candidate.
2. Reuse the existing sequence, KV, prefix-checkpoint, and GDN replay machinery.
3. Candidates are output-logit choices, not independent model sequences.
4. M1 supports one-token labels only.
5. MTP and Vision are excluded from M1.
6. Existing generation behavior must remain unchanged.

## M1 architecture

A request consists of:

- one prepared/shared context;
- one or more fields;
- each field has a short suffix/question;
- each field has 2-16 allowed values;
- allowed values are represented internally by one-token labels.

Execution:

1. Prefill the shared context once.
2. Retain that frontier as the trunk state.
3. For each field:
   - restore/copy trunk state into one scratch sequence;
   - append the field suffix;
   - run the target model to next-token logits;
   - gather logits for the allowed token IDs;
   - apply softmax over only those candidates;
   - return winner and probabilities.
4. Restore/reuse the same scratch sequence for the next field.

Candidate count therefore does not multiply KV or linear-attention state.

## M1-A: constrained token scorer

Input:

- BF16 logits `[physical_vocab, B]`
- I32 candidate token IDs `[K, B]`

Output:

- FP32 probabilities `[K, B]`
- I32 winner indices `[B]`

Initial domain:

- `B` in `[1, 8]`
- `K` in `[2, 16]`
- every candidate token ID must be inside the valid token domain.

Semantics:

For each batch column `b`:

    score[k] = float(logits[candidate_ids[k,b], b])

    probability[k] =
        exp(score[k] - max(score)) /
        sum_j exp(score[j] - max(score))

Winner is the lowest candidate index with maximum probability.

This is a constrained distribution over the supplied choices. It must not be
described as calibrated probability of correctness.

## M1-B: runtime prototype

Expose an internal target/runtime path capable of:

- preserving a shared trunk frontier;
- restoring that frontier into one scratch lane;
- running a field suffix;
- producing next-token logits without committing sampled generation;
- calling the M1-A constrained scorer.

No public server route yet.

## M1-C: product API

Add typed decision structures to the Engine API and `/v1/decision`.

Initial field types:

- boolean
- enum

Initial restrictions:

- 1 context
- 1-8 fields
- 2-16 choices per field
- single-token internal labels
- no MTP
- no Vision

Response should include:

- selected value
- candidate probabilities
- prompt/context token counts
- scored fields
- timing
- prefix reuse information

## Benchmark

Compare constrained decision latency with normal JSON generation for equivalent
boolean/enum decisions.

Record:

- cold prefill latency
- warm/prefix-reused latency
- per-field scoring latency
- total decision latency
- VRAM delta
- output correctness
- candidate probability distribution

## Later work

M2:
- multi-token candidate tries
- batched contexts
- numeric grids
- restricted-row LM-head evaluation

M3:
- calibration evaluation
- OpenClaw routing integration
- policy thresholds / abstention

## M1-B decision frontier implementation

Runtime reconnaissance established that M1-B does not require a second sequence
lane.

A retained sequence already owns the complete attention KV prefix while its
linear-attention state occupies the stable state slot associated with that lane.

M1-B therefore uses a reversible same-lane frontier:

1. Capture the current linear/GDN state to a decision-specific pinned-host
   snapshot.
2. Re-bind the retained sequence's existing KV allocation.
3. Temporarily enlarge KV page entitlement when a suffix crosses a page boundary.
4. Execute explicit suffix tokens without sampling.
5. Score the final next-token logits with `constrained_choice`.
6. Restore the linear/GDN state from the host snapshot.
7. Trim KV back to the original frontier.
8. Restore the original page entitlement and unbind the retained allocation.

No attention-prefix KV is copied.

The ordinary decode path is reused but the decision traversal stops before
sampling and does not publish continuation hidden into the retained sequence.

The existing rewrite-checkpoint host allocation is deliberately not reused.
Decision execution has a separate lazy pinned-host snapshot so ordinary prefix
checkpoint policy remains unaffected.


## M1-C2: typed bool/enum tokenization

M1-C2 adds a product-facing typed layer above the raw-token M1-C1
Engine/executor path.

Supported field types:

- boolean
- enum

Boolean fields use the canonical caller-visible values:

- `false`
- `true`

Enum fields carry 2-16 caller-visible string values.

The Qwen frontend tokenizes each complete `suffix + candidate` path before
submission to the executor. It computes the longest common token prefix across
all candidate paths and uses that shared prefix as the executable field suffix.
In M1 every candidate path must then diverge by exactly one valid token, and
distinct values must produce distinct candidate token IDs.

Joint path tokenization is required because BPE tokenization can change at the
boundary between the suffix and candidate text. Independently tokenizing those
strings is not generally equivalent to tokenizing the actual continuation.
Multi-token divergences are deferred to the M2 trie path.

Suffixes may contain multiple tokens.

Typed requests are translated into the existing `DecisionFieldSpec` raw-token
representation, so M1-C2 does not change scheduler admission, retained-frontier
execution, constrained scoring, or state restoration.

The result preserves:

- candidate string values
- candidate token IDs
- restricted-choice probabilities
- winner index
- winner token
- selected string value

Multi-token candidate values remain M2 work.



## Harness-agnostic architecture

Harness neutrality is a core architectural invariant of constrained decisions.

The NInfer structured-decision core must not depend on a particular agent
harness, protocol, API dialect, or tool-call representation.

Possible callers include OpenClaw, Pi, Hermes-style harnesses, DeepSeek
Harness, MCP-based systems, OpenAI-compatible clients, Anthropic-compatible
clients, custom agent runtimes, and direct NInfer API callers.

Those systems are adapters into the generic NInfer decision contract. They are
not part of the decision execution model itself.

Required layering:

    harness / protocol / application
                    |
                    v
           adapter / translator
                    |
                    v
       StructuredDecisionSchema
                    |
                    v
       compile_decision_plan()
                    |
                    v
        CompiledDecisionPlan
                    |
                    v
       decision execution runtime
                    |
                    v
       structured decision result

The structured-decision core must remain independent of OpenClaw-specific,
Pi-specific, Hermes-specific, and DeepSeek Harness-specific concepts, as well
as OpenAI function-call syntax, Anthropic tool_use syntax, MCP transport
details, JSON punctuation, protocol framing, and any one server endpoint.

Protocol and harness adapters translate their native representations into
StructuredDecisionSchema and translate generic results back into whatever
representation the caller requires.

For tool calling, JSON is an output representation rather than the core
execution abstraction. Deterministic braces, commas, quotes, property names,
and other known syntax should eventually be assembled outside the model.

The same compiled-decision machinery must remain usable for non-tool decisions
such as agent routing, specialist selection, retry / stop / escalate,
approvals, policy decisions, validation, workflow branching, bounded
configuration choices, and ordinary boolean and enum decisions.

Adding support for a new harness must require an adapter, not changes to the
constrained-decision GPU/runtime architecture.

## CompiledDecisionPlan architecture

The structured decision API is separated into three layers.

### StructuredDecisionSchema

`StructuredDecisionSchema` is model agnostic. It represents caller/schema
semantics such as boolean and enum fields without exposing tokenizer state or
token IDs.

### CompiledDecisionPlan

`Engine::compile_decision_plan()` resolves a structured schema against the
Engine's active target and tokenizer.

The compiled plan is immutable and reusable across requests. It stores the
model-resolved finite execution metadata required by the current constrained
runtime.

Compilation performs operations such as:

- canonical boolean expansion;
- enum validation;
- joint suffix-plus-choice tokenization;
- longest-common-token-prefix extraction;
- candidate branch resolution;
- current backend compatibility validation.

Compilation does not execute the model.

### DecisionFieldSpec

`DecisionFieldSpec` remains the low-level execution primitive consumed by the
current executor/runtime. It is not the long-term schema abstraction.

This separation allows future schema nodes such as optional fields, numeric
grids, tries and free-generation leaves to compile to different execution
forms without exposing those details to callers.

Tool-call protocol parsing remains outside the Engine. OpenAI, Anthropic,
Responses and other server adapters translate their tool schemas into the
generic structured schema before compilation.

The current constrained-choice candidate ceiling is an execution-backend
qualification limit and is deliberately not part of the long-term structured
schema abstraction.


## Semantic execution architecture

The long-term objective is broader than faster JSON generation or tool calling.

The core principle is:

> Keep semantic constraints. Eliminate representational generation.

The model should spend inference compute only where genuine semantic uncertainty
exists. Structure already known to the application should not be regenerated
token by token merely because a protocol chooses to represent that structure as
JSON, XML, SQL, source code, a tool call, or another textual format.

### Three classes of execution

A structured result contains three fundamentally different kinds of work.

1. **Deterministic structure**

   Structure already known from the semantic contract must not require a model
   choice.

   Examples include protocol framing, property names, fixed literals,
   punctuation, known function names after a prior selection, static query
   structure, and other representation-only material.

2. **Finite semantic uncertainty**

   A value with a known finite domain should be evaluated as a constrained
   choice rather than generated autoregressively.

   Examples include booleans, enums, routing targets, bounded categorical
   values, approval decisions, policy outcomes, tool selection, bounded
   integers, and numeric grids.

3. **Open-ended semantic uncertainty**

   Content whose domain is not practically finite remains ordinary language
   generation and may use the normal decode path, MTP, or another generation
   backend.

   Examples include free-form explanations, long text, unrestricted code
   bodies, and open-ended argument values.

These execution classes may coexist in one request.

### Semantic graph, not serialization graph

The core abstraction must describe semantic computation rather than JSON or any
other wire representation.

Protocol and harness adapters may translate:

- OpenAI-compatible tool schemas;
- Anthropic tool-use schemas;
- MCP tools;
- OpenClaw;
- Pi;
- Hermes-style harnesses;
- DeepSeek Harness;
- workflow systems;
- extraction/classification applications;
- SQL/query builders;
- code-generation systems;
- vision applications;
- custom callers;

into one generic semantic execution representation.

The core must not require changes when a new harness or serialization format is
added.

JSON, tool-call syntax, SQL, source code and other formats are renderings of the
semantic result. They are not the execution model.

### FiniteChoice as the fundamental bounded operation

Boolean and enum remain useful product-facing and compatibility types, but they
are not the fundamental long-term execution primitive.

Conceptually, both lower to a finite semantic choice:

    Boolean:
        choices = [false, true]

    Enum:
        choices = [value0, value1, ...]

Future bounded integer and numeric-grid decisions may use the same semantic
operation even when their execution backend differs.

Backend qualification limits such as candidate count, batch width, one-token
branches, vocabulary geometry or kernel dimensions must not leak into this
semantic abstraction.

### Graph representation

The semantic representation should support stable node identities and explicit
dependencies.

The preferred direction is a flat node graph / arena with typed payloads rather
than recursive ownership being the execution contract.

Conceptually:

    SemanticExecutionGraph
        nodes[]
        root / roots

    SemanticNode
        typed payload
        dependencies / children

Typed payload representation may use `std::variant` or an equivalent tagged
representation. Exact public C++ structures are deliberately deferred until the
V2 implementation contract is reviewed.

Stable node identities enable:

- dependency validation;
- topological planning;
- execution-wave construction;
- conditional paths;
- DAG reuse;
- compiled-plan lowering;
- serialization or hashing later;
- reusable subplans;
- result-to-node attribution.

### Dependency analysis and parallel waves

Independent decisions should not remain permanently serialized merely because
the first runtime prototype executes fields sequentially.

A future compiler should identify independent nodes and schedule them in the
same execution wave where the target/runtime permits it.

For example:

    Wave 0:
        classify
        urgent?
        sentiment

    Wave 1:
        selected tool depends on classify
        escalation path depends on urgent?

    Wave 2:
        argument schema depends on selected tool

    Wave 3:
        optional free-text leaf uses ordinary/MTP generation

The exact GPU/runtime implementation of a wave is backend-specific and is not
part of the semantic schema.

This generalizes the independent-field parallel-scoring approach demonstrated
by parallel finite-decision systems while still supporting dependency chains.

### Published result and model conditioning are distinct

A semantic node may affect the application-visible result, the model's working
context, both, or neither directly.

These concepts must not be conflated.

For example, deterministic protocol syntax may need to be represented in model
conditioning so that a later decision is scored in the correct textual/model
context, while that syntax should not appear as an independent semantic value
in the application result.

Conversely, an application-provided value may contribute directly to the
semantic result without requiring the model to choose it.

The compiler/runtime therefore needs a conceptual separation between:

    semantic result effect

and:

    model-conditioning effect

This does not imply that low-level execution flags belong in the public schema.
The compiler should derive execution effects from semantic nodes and adapter
metadata wherever possible.

### Whole-path tokenization remains authoritative

Whenever semantic execution requires model-conditioning text, compilation must
preserve actual tokenizer behavior across boundaries.

The M1-C2 rule remains authoritative:

    tokenize the complete semantic continuation path

rather than assuming separately tokenized fragments can be concatenated.

BPE and related tokenizers can change segmentation across textual boundaries.

Future deterministic conditioning, finite tries, dependent branches and
free-generation transitions must preserve this property.

### Node-specific execution strategies

The compiled plan may select a different execution mechanism for different
semantic nodes.

Conceptually:

    deterministic node
        -> assemble result and/or advance model conditioning

    finite choice
        -> constrained scorer

    multi-token finite choice
        -> token trie / finite-path scorer

    open-ended leaf
        -> ordinary decode / MTP

    application-known value
        -> no model decision

The semantic graph does not prescribe CUDA kernels, KV strategy, batching
geometry, MTP details or target-specific implementations.

### Full-vocabulary projection is not an architectural requirement

The current constrained-decision path still obtains full-vocabulary logits
before selecting a small number of candidates.

That is a current implementation limitation, not part of the semantic
contract.

A future restricted-row LM-head path may project only the required output rows
for finite decisions.

For a small finite domain this changes the conceptual work from:

    hidden -> complete vocabulary -> retain K logits

toward:

    hidden -> K required output rows

without changing the semantic graph.

### Prefill and reusable semantic prefixes

Stable application instructions, schema descriptions, tool catalogues and
other reusable semantic context may eventually be compiled into reusable
prefixes.

A request may therefore separate:

    stable application/schema context
        -> cache/prefill once

    request-specific context
        -> normal request prefill

    semantic execution
        -> deterministic / finite / free execution nodes

The semantic architecture must not prevent such prefix reuse.

### Vision and multimodal input

Vision is an input-context concern rather than a separate structured-output
architecture.

A semantic execution graph may operate over context produced from:

- text;
- images;
- video;
- retrieved documents;
- tool/application state;
- multimodal combinations.

The same finite, deterministic and free-generation nodes apply after the
multimodal context has been established.

### MTP and structured decisions are complementary

Semantic execution does not replace MTP.

MTP should be used where the node genuinely requires open-ended generation.

Finite choices should not be forced through MTP merely because they occur
inside a response that also contains free text.

A mixed plan may therefore contain:

    finite node
        -> constrained scorer

    deterministic node
        -> deterministic conditioning / assembly

    free-generation node
        -> MTP

The compiler selects the appropriate execution strategy per node.

### CompiledDecisionPlan remains opaque

`CompiledDecisionPlan` remains the boundary between model-agnostic semantics
and model-resolved execution.

It may eventually contain:

- resolved tokenizer paths;
- dependency graph;
- execution waves;
- finite-choice metadata;
- deterministic conditioning paths;
- trie nodes;
- result mappings;
- free-generation transitions;
- target/backend compatibility information.

These details must not be exposed as the public semantic schema.

For the current milestone, compiled plans remain bound to the Engine instance
that created them. Model/tokenizer fingerprints and cross-Engine compiled-plan
caching are deferred optimizations.

### V2 staged implementation

The next implementation series should proceed in deliberately separated steps.

#### V2-A: semantic graph

Introduce the model-agnostic semantic node graph and preserve legacy bool/enum
APIs through compatibility lowering.

V2-A must not require changes to:

- CUDA constrained-choice kernels;
- decision frontier mechanics;
- executor execution semantics;
- ordinary generation;
- MTP;
- Vision.

Initial graph compilation may lower finite nodes back into the existing
`DecisionFieldSpec` execution representation.

#### V2-B: dependency-aware execution

Introduce an ephemeral working frontier and sequential execution nodes so a
later node can be conditioned on an earlier selected value.

Preserve the caller's retained frontier.

#### V2-C: parallel execution waves

Identify independent finite nodes and evaluate them together when the target
runtime can do so efficiently.

Dependency edges determine later waves.

#### V2-D: deterministic conditioning and result assembly

Allow known structure to advance model context where required without asking
the model to choose deterministic representation tokens.

Keep application-visible semantic output separate from model-conditioning
syntax.

#### V2-E: multi-token finite choices

Generalize finite choices to token tries / finite paths.

#### V2-F: restricted-row LM head

Avoid complete-vocabulary output projection when only a small set of output
rows is required by a finite node.

#### V2-G: open-ended generation leaves

Allow a semantic graph to hand control to ordinary generation or MTP for
genuinely open-ended values, then return to structured execution where
supported.

#### V2-H: adapters and compiled-plan caching

Add protocol/harness adapters and later introduce cache identities based on
semantic schema, model/tokenizer identity and compiler version.

### Performance principles

Major architecture decisions must be evaluated across:

- prompt/prefill work;
- prefix reuse;
- decision/scoring work;
- ordinary decode;
- MTP/speculative execution;
- Vision/multimodal input;
- KV and recurrent-state memory;
- temporary workspace;
- LM-head bandwidth;
- CPU/GPU synchronization;
- batching and concurrency;
- dependent versus independent decisions;
- server/harness integration.

A design should not be accepted solely because its C++ representation is
convenient.

Where relevant, decisions should be compared against approaches used by other
inference engines and structured-decision systems.

### Long-term success criterion

The goal is not merely valid structured text.

The goal is to minimize model computation spent reproducing information the
runtime already knows.

The ideal execution path asks the model only the questions whose answers are
actually uncertain.
