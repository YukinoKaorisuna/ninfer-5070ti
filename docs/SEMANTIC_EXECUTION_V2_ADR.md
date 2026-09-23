# ADR: Semantic Execution V2 Intermediate Representation

Status: Accepted for V2-A design

## Context

The constrained-decision work began with finite boolean and enum decisions, but
the long-term opportunity is broader than JSON or tool calling.

The governing principle is:

> Keep semantic constraints. Eliminate representational generation.

Model computation should be spent on semantic uncertainty rather than
reproducing structure the runtime already knows.

The architecture must remain useful for:

- classification and extraction;
- routing and workflow decisions;
- approvals and policy decisions;
- tool selection and tool arguments;
- bounded numeric decisions;
- SQL/query construction;
- code and DSL construction;
- multimodal and vision-derived decisions;
- mixed structured and free-form responses;
- custom application runtimes and agent harnesses.

No protocol or serialization format is privileged by the core design.

## External reference systems

The V2 decision was informed by several distinct structured-inference
architectures.

### RLCD / Parallel Constrained Decoding

The Qwen RLCD implementation demonstrates:

- one shared semantic/context prefill;
- field-level KV broadcasting;
- batched independent finite decisions;
- candidate-logit slicing;
- multi-token collision handling;
- programmatic JSON assembly;
- zero autoregressive JSON output tokens for the finite path.

This validates the value of removing serialization generation and batching
independent semantic decisions.

The current PyTorch implementation still obtains a complete model logits tensor
and indexes candidate token IDs afterward. Therefore its candidate slicing is
not evidence that a restricted-row LM head has already eliminated full
vocabulary projection.

### thecodacus llama.cpp parallel-decision

The parallel-decision branch demonstrates:

- finite boolean, enum, bounded integer and numeric-grid values;
- one cached/static prefix;
- multiple independent field branches;
- shared/forked KV state;
- whole-path tokenization;
- multi-token candidate tries;
- tree scoring of divergence nodes;
- greedy trie traversal for larger candidate sets;
- code-side JSON result assembly;
- batching multiple contexts and branches subject to sequence/batch capacity.

This validates both finite-choice generalisation and execution strategies that
depend on candidate path topology.

Its public schema/compiler remains JSON-oriented, and candidate scoring obtains
normal logits before selecting required token entries. Those details are useful
reference implementations but are not adopted as core NInfer abstractions.

### XGrammar / vLLM / SGLang

Grammar-based systems demonstrate a different strength:

- arbitrary JSON/regex/EBNF-style output constraints;
- grammar compilation;
- stateful matchers;
- vocabulary masks;
- rollback and matcher forking;
- speculative-decode integration;
- deterministic jump-forward regions;
- mixed free-form and structured output.

These capabilities remain valuable as a possible backend for genuinely
open-ended constrained generation.

They are not selected as NInfer's primary semantic IR because a grammar
describes the legal serialized token language rather than the semantic
uncertainty of the application.

A grammar therefore cannot, by itself, preserve all of the semantic knowledge
needed to choose cheaper execution such as direct finite scoring or
representation-free result assembly.

## Decision

NInfer V2 will use a:

> flat value-centric semantic DAG, with model presentation separated from
> semantic meaning, lowered into an opaque cost-optimised compiled execution
> plan.

This supersedes the simpler description "flat node arena plus variant payload"
as the complete architectural decision.

The arena remains useful, but separation of concerns is the essential property.

## Layer 1: semantic DAG

The semantic graph describes application meaning.

It must answer questions such as:

- what values are uncertain;
- what values are already known;
- what finite domains are legal;
- which computations depend on earlier semantic results;
- which values require open-ended generation.

It must not encode:

- token IDs;
- BPE boundaries;
- JSON punctuation;
- tool-call delimiters;
- OpenAI/Anthropic/MCP protocol syntax;
- KV page details;
- CUDA geometry;
- MTP implementation;
- scorer candidate ceilings;
- backend-specific batching limits.

### Stable node identity

Nodes have stable IDs within a graph.

A flat arena is preferred because stable IDs support:

- dependency edges;
- DAG reuse;
- validation;
- topological ordering;
- execution-wave construction;
- node-attributed results;
- compiled-plan lowering;
- future hashing/serialization.

Recursive ownership is not the execution contract.

### SemanticValue

Finite and known values are represented as typed semantic values rather than
serialized text.

The initial value domain should be capable of representing at least:

- boolean;
- signed integer;
- floating-point number;
- string.

The exact public C++ type is deferred to the V2-A API review.

Adding a value kind must be justified by semantic need rather than by a
particular wire format.

### Fundamental node classes

The long-term semantic model distinguishes at least these classes.

#### KnownValue

A value already known to the application.

No model decision is required.

#### FiniteChoice

A semantic value chosen from a finite legal domain.

Boolean, enum, bounded integer and numeric grid are specialisations or adapter
conveniences around this concept rather than fundamental GPU/runtime classes.

#### FreeGeneration

A genuinely open-ended semantic value.

This class may use ordinary autoregressive generation, MTP/speculative
generation or a constrained grammar backend.

FreeGeneration is intentionally deferred beyond V2-A.

### Boolean and enum compatibility

Existing Boolean and Enum product APIs remain supported.

They lower into generic FiniteChoice semantics.

For example:

    Boolean
        -> FiniteChoice [false, true]

    Enum ["local", "remote", "human"]
        -> FiniteChoice ["local", "remote", "human"]

Current backend restrictions such as one-token branches or K <= 16 are
compilation/backend constraints, not FiniteChoice semantics.

## Ordering is not a semantic Sequence node

V2 does not initially define Sequence as a fundamental semantic node.

Three distinct concepts must not be conflated:

1. semantic dependency;
2. model-conditioning order;
3. output rendering order.

For example, JSON property order is a rendering concern.

A tool argument depending on the selected tool is a semantic dependency.

A deterministic text fragment that the model must see before a later decision
is a model-conditioning concern.

These may coincide in some adapters but are not the same abstraction.

## Layer 2: model presentation

Semantic meaning and model representation are explicitly separate.

A FiniteChoice may contain semantic values:

    false
    true

while a particular model continuation may require representations such as:

    "false"
    "true"

or:

    " false"
    " true"

or tokenisation whose common path includes surrounding syntax.

Likewise a semantic tool name may be represented differently by different model
chat templates.

The model-presentation layer is responsible for supplying or deriving:

- semantic cues/descriptions;
- finite-choice textual representations;
- deterministic conditioning material;
- target/chat-template-specific representation where necessary.

It still contains no token IDs.

Tokenizer resolution belongs to compilation.

### Whole-path tokenization

Compilation must tokenize actual complete continuation paths.

Separately tokenized fragments must not be assumed composable because BPE and
other tokenizers can change segmentation at boundaries.

The existing M1-C2 whole-path plus longest-common-token-prefix rule remains an
architectural invariant.

## Output rendering is separate again

Application-visible semantic results are not serialized output strings.

The runtime should be able to return node-attributed typed values and associated
decision telemetry.

Adapters may render those results as:

- JSON;
- OpenAI-compatible tool calls;
- Anthropic tool use;
- MCP arguments;
- SQL;
- source code;
- configuration;
- native application objects;
- workflow transitions;
- another representation.

Known punctuation and protocol framing must not require model generation merely
because an adapter ultimately emits text.

## Layer 3: CompiledDecisionPlan

CompiledDecisionPlan remains opaque.

It is the boundary at which semantic meaning plus model presentation becomes a
model/tokenizer/backend-resolved execution program.

A future compiled plan may contain:

- tokenizer-resolved continuation paths;
- finite-choice token tries;
- dependency topology;
- topological execution waves;
- deterministic conditioning operations;
- node/result mappings;
- memory requirements;
- branch-state requirements;
- backend capability decisions;
- free-generation transitions;
- grammar fallback state;
- restricted-row LM-head plans;
- prefix/cache metadata.

These are not public semantic-schema concerns.

For now, compiled plans remain bound to the Engine instance that compiled them.

## Dependency graph

Semantic dependency is represented independently from rendering order.

If node B needs the selected semantic value of node A before B can be evaluated,
the compiled plan must schedule B after A.

Independent nodes are eligible for the same execution wave.

Conditional activation and branch predicates are deferred beyond V2-A; their
future representation must build on semantic node identities rather than
serialized-token positions.

## Parallel execution waves

The compiler should eventually construct topological waves such as:

    Wave 0:
        A
        B
        C

    Wave 1:
        D depends on A
        E depends on B

    Wave 2:
        F depends on D and E

Nodes in a wave are semantically independent.

Semantic independence does not require the runtime to execute every node in the
wave simultaneously.

## Runtime cost model

Execution width and strategy are runtime/compiler decisions.

For an independent wave the backend may choose:

- serial reversible probing;
- full parallel execution;
- bounded-width microbatching;
- trie node batching;
- another target-specific strategy.

The choice may depend on:

- free VRAM;
- KV layout;
- recurrent/GDN state size;
- suffix/path lengths;
- request concurrency;
- batch/ubatch limits;
- temporary workspace;
- target architecture;
- current service load.

This is especially important on memory-constrained accelerators.

Parallelism must never require naive full candidate-state duplication as a
semantic contract.

## Qwen hybrid/recurrent-state implication

NInfer's current Qwen decision path preserves attention KV and reversible
linear/GDN state without persistent state per candidate.

V2 parallelisation must preserve that memory advantage.

Independent semantic nodes may be eligible for parallel execution while still
being microbatched or evaluated serially if duplicating recurrent state would
violate the memory plan.

The semantic graph expresses independence.

The compiled/runtime plan chooses affordable execution width.

## Node-specific execution strategies

Different semantic nodes may use different execution mechanisms.

Conceptually:

    KnownValue
        -> no model evaluation

    one-token FiniteChoice
        -> constrained candidate scorer

    multi-token FiniteChoice
        -> finite token trie

    independent FiniteChoice wave
        -> parallel or microbatched scoring

    deterministic conditioning
        -> advance model state without semantic choice

    FreeGeneration
        -> ordinary decode / MTP

    FreeGeneration with arbitrary format constraint
        -> ordinary/MTP decode plus grammar backend

No single mechanism is required to handle every node.

## Grammar is a fallback/backend, not the semantic IR

Grammar/FSM systems remain important for output whose legal domain is too broad
or dynamic to enumerate usefully.

Examples may include:

- arbitrary strings subject to syntax;
- recursively nested free structures;
- regex-constrained open text;
- mixed free text and structured tags.

A future FreeGeneration node may compile to a grammar matcher.

Finite semantic choices should not be degraded into token-by-token grammar
decoding when their legal domain is already explicitly known.

## Deterministic output versus deterministic conditioning

Known output must not automatically imply zero model work.

A deterministic representation fragment can have two independent effects:

1. it may be assembled directly into the external representation;
2. later model decisions may need the corresponding continuation in their
   model context.

The first requires no model execution.

The second may require model-state advancement.

The architecture therefore distinguishes semantic/result effects from
model-conditioning effects.

Jump-forward techniques in grammar systems support the same general insight:
deterministic output tokens need not be chosen by the model, but model state may
still need to become consistent with the deterministic continuation.

## Prefill architecture

The compiler/runtime should eventually distinguish:

    stable application/schema/tool context
        -> reusable prefix

    request-specific evidence/context
        -> per-request prefill

    node-specific model presentation
        -> semantic execution

This enables repeated requests to amortize stable catalogue/instruction
prefill.

A semantic catalogue is useful only when its prefill cost is lower than the
work it removes or can be reused sufficiently.

Therefore catalogue construction and prefix reuse are compiler/runtime
optimisations rather than mandatory semantic-graph structure.

## Full-vocabulary LM head

A FiniteChoice semantically requires scores only for its finite legal domain.

Current NInfer, RLCD and the inspected llama.cpp parallel-decision
implementation all still obtain normal model logits before selecting candidate
entries.

That must not become an architectural assumption.

For Qwen3.8 with physical vocabulary 248,320 and hidden width 5,120, a complete
output matrix contains approximately 1.27 billion weights.

At an idealised four bits per weight, that corresponds to roughly 606 MiB of
weight payload for the full matrix, before quantisation metadata and other
implementation effects.

A two-choice decision semantically requires only two output rows.

A future restricted-row LM-head backend is therefore a high-value optimisation
enabled by preserving FiniteChoice semantics in the IR.

Actual realised speedup must be benchmarked; raw row-count ratios are not a
latency prediction.

## MTP / speculative generation

MTP is complementary to semantic execution.

FiniteChoice nodes normally bypass MTP.

FreeGeneration nodes may use MTP.

Grammar-constrained FreeGeneration may require grammar state for draft
positions, verification and rollback.

Keeping finite nodes outside that mechanism reduces unnecessary coupling
between finite decisions and speculative grammar state.

V2 must not make MTP state part of the public semantic schema.

## Vision and multimodal input

Vision does not require a different semantic IR.

Images, video, text, retrieved data and application state contribute evidence
to the model context.

The semantic graph describes the output computation after that context exists.

The same KnownValue, FiniteChoice and FreeGeneration concepts apply.

Vision-specific encoder/cache/workspace behaviour remains a target/runtime
concern.

## Result probabilities

Restricted-choice probabilities describe the model's normalized distribution
over the supplied finite alternatives at the evaluated semantic node.

They are not automatically calibrated probabilities of real-world correctness.

Calibration is an evaluation layer and remains separate from execution.

## Rejected alternatives

### Grammar/FSM as the core IR

Rejected as the primary representation.

Benefit:

- very broad serialized-language coverage.

Reason for rejection:

- represents legal output token language rather than semantic uncertainty;
- keeps ordinary decode in the critical path;
- loses direct finite-domain optimisation opportunities;
- couples structured output to per-token state/masks;
- complicates speculative/MTP integration.

Grammar remains a possible backend for FreeGeneration.

### Recursive semantic tree as the execution contract

Rejected.

Benefit:

- natural authoring syntax;
- closely resembles JSON Schema and many user-facing schemas.

Reason for rejection:

- nesting does not equal dependency;
- shared subgraphs/DAGs become awkward;
- topological wave planning is less natural;
- ownership and reusable nodes become more complex.

Adapters may still expose convenient recursive builders.

### Flat graph with presentation embedded in semantic nodes

Rejected as the final abstraction.

Benefit:

- stable IDs and easy execution lowering.

Reason for rejection:

- semantic values become contaminated by model strings, protocol syntax or
  output representation;
- makes multi-model/model-template compilation harder;
- risks freezing JSON/tool assumptions into the core.

The selected design retains the flat graph but separates presentation.

## V2-A implementation scope

V2-A deliberately proves the new abstraction without changing runtime
execution.

V2-A should introduce enough generic representation for existing independent
finite decisions to lower through the semantic IR.

The proof path is:

    legacy Boolean / Enum
            |
            v
    generic FiniteChoice semantics
            |
            + model-presentation metadata
            |
            v
    compile_decision_plan()
            |
            v
    existing DecisionFieldSpec[]
            |
            v
    existing executor / frontier / scorer

V2-A must not require changes to:

- constrained-choice CUDA kernels;
- reversible decision frontier mechanics;
- scheduler execution semantics;
- MTP;
- Vision;
- ordinary generation;
- production HTTP protocols.

## V2-A API design gates

Before C++ mutation, the exact API review must resolve:

1. the precise SemanticValue representation;
2. SemanticNodeId type and invalid-ID semantics;
3. FiniteChoice option representation;
4. where semantic descriptions/cues live;
5. how candidate model representations are associated with semantic values;
6. whether graph roots are explicit or inferred;
7. whether unconditional dependency edges enter V2-A or remain deferred;
8. how node-attributed results coexist with the legacy DecisionResult fields;
9. validation rules for duplicate IDs, duplicate values and unreachable nodes;
10. compatibility lowering from DecisionFieldInput;
11. whether the initial variant contains only implemented node classes or
    reserves future classes;
12. source/ABI consequences of expanding the public variant later.

No production API structure should be frozen until these gates are reviewed.

## Performance acceptance framework

Every major semantic-execution design change must be evaluated across:

- prefill work;
- reusable-prefix opportunity;
- decode rounds eliminated;
- finite scoring cost;
- LM-head bandwidth;
- MTP/speculative interaction;
- Vision/multimodal interaction;
- attention KV memory;
- recurrent/GDN state memory;
- temporary workspace;
- batch/microbatch width;
- request concurrency;
- CPU/GPU synchronization;
- tokenizer correctness;
- dependent versus independent nodes;
- protocol/harness portability.

A design is not accepted merely because its C++ representation is convenient.

## References reviewed

- Qwen-2.5-1B-RLCD / Parallel Constrained Decoding:
  https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD
- RLCD PyTorch engine:
  https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_torch.py
- thecodacus llama.cpp parallel-decision branch:
  https://github.com/thecodacus/llama.cpp/tree/parallel-decision
- parallel-decision engine:
  https://github.com/thecodacus/llama.cpp/tree/parallel-decision/tools/parallel-decision
- XGrammar engine integration:
  https://github.com/mlc-ai/xgrammar/blob/main/docs/using_xgrammar/engine_integration.md
- XGrammar structural tags:
  https://github.com/mlc-ai/xgrammar/blob/main/docs/structural_tag/structural_tag.md
- vLLM structured output:
  https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md
- SGLang structured output:
  https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/structured_outputs.mdx

## Consequence

V2-A may now proceed to exact public C++ API design.

No runtime implementation should begin until the V2-A API gates above have
been reconciled against the existing public NInfer API and tests.
