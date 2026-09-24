# Constrained Decisions: current API and usage

This document describes the implemented constrained-decision surface through
V2-D1. The milestone design documents remain useful architectural records, but
this file is the current developer/operator contract.

## What is implemented

NInfer can score explicit finite semantic choices from one retained prompt
frontier without committing ordinary autoregressive output.

Implemented paths:

- convenience Boolean/Enum `DecisionFieldInput`;
- reusable `StructuredDecisionSchema` + `DecisionModelPresentation`;
- immutable `CompiledDecisionPlan` bound to the Engine that compiled it;
- one-root direct fan-out semantic dependencies;
- V2-C shared-frontier optimization for compatible depth-1 siblings;
- V2-D1 multi-token finite choices using private token tries;
- synchronous `Engine::decide()` and asynchronous
  `Engine::submit_decision(...).wait()`.

The result distribution is a constrained softmax over legal choices. It is not
a calibrated probability that a choice is factually correct.

## Runtime requirement: ordinary backend

Current finite-decision execution requires:

```cpp
options.speculative.backend = ninfer::SpeculativeBackend::None;
```

MTP/DFlash engines do not allocate the ordinary decision scratch/state used by
the reversible probe path. Compilation/execution therefore rejects an Engine
configured with a speculative backend.

This is an execution-mode limitation, not a semantic-schema rule. D1 does not
silently disable MTP or mutate an Engine's configured backend.

The normal production RTX 5080 profile in this fork uses MTP-3. A caller that
wants the constrained-decision C++ API today must use an ordinary-backend Engine
(or a separate process configured without speculation).

## Protocol status

The current constrained-decision implementation adds a C++ Engine API. It does
**not** add an HTTP
`/v1/decision` endpoint and does not change the OpenAI/Anthropic protocol
adapters.

Existing `ninfer-serve` routes therefore do not automatically expose
`StructuredDecisionSchema`.

## Convenience Boolean/Enum API

Use `DecisionFieldInput` when the caller-visible enum strings are also the
model-facing candidate strings and no semantic dependencies are needed.

```cpp
ninfer::EngineOptions options;
options.artifact_path = "/path/to/model.ninfer";
options.speculative.backend = ninfer::SpeculativeBackend::None;

ninfer::Engine engine(options);

ninfer::PromptInput input;
// Populate input.messages / input.options for the request.
ninfer::PreparedPrompt prompt =
    engine.prepare(std::move(input));

ninfer::DecisionFieldInput route;
route.name = "route";
route.type = ninfer::DecisionFieldType::Enum;
route.suffix = " route: ";
route.values = {
    "local alpine ridge north",
    "local alpine valley south",
    "remote coastal ridge east",
    "remote coastal valley west",
};

ninfer::DecisionResult result =
    engine.decide(std::move(prompt), {std::move(route)});
```

Boolean fields omit `values`; the convenience layer supplies canonical
`false` and `true` semantic/model-facing candidates.

The convenience path uses the same whole-path compiler as the semantic API.
Enum values are not restricted to one token: a field may lower to a V2-D1 trie
when its complete candidate continuations require multiple tokens.

## Compiled semantic API

Use the semantic API when semantic values and model-facing presentation must be
separate, when a plan will be reused, or when dependencies are required.

```cpp
ninfer::StructuredDecisionSchema schema;
ninfer::DecisionModelPresentation presentation;

ninfer::FiniteChoice route;
route.label = "route";
route.choices = {
    ninfer::SemanticValue::string("local"),
    ninfer::SemanticValue::string("remote"),
};

const ninfer::SemanticNodeId route_id =
    schema.add_finite_choice(std::move(route));

presentation.set_finite_choice(
    route_id,
    ninfer::FiniteChoicePresentation{
        " route: ",
        {"local alpine ridge", "remote coastal ridge"},
    });

ninfer::CompiledDecisionPlan plan =
    engine.compile_decision_plan(schema, presentation);

ninfer::DecisionResult result =
    engine.decide(std::move(prompt), plan);
```

A compiled plan is tied to the exact `Engine` instance that created it. It may
be copied cheaply and reused for requests on that Engine, but it is not
portable to another Engine instance even when the model artifact is identical.

## Dependencies

The semantic graph and model conditioning are separate.

```cpp
schema.add_dependency(parent_id, child_id);

presentation.set_dependency_conditioning(
    parent_id,
    child_id,
    ninfer::DependencyConditioningPresentation{
        {
            " selected parent outcome=false; ",
            " selected parent outcome=true; ",
        },
    });
```

`selected_choice_texts[i]` is deterministic model-facing conditioning used
when parent semantic choice index `i` wins. It is not generated output and it
does not have to equal the parent's scoring candidate text.

Current compiled topology is intentionally narrower than the semantic graph:

- exactly one semantic root when dependencies are present;
- every other node is a direct child of that root;
- one parent per child;
- no deep dependency chain or multiple-parent child yet.

V2-D1 trie variants can be selected as dependency variants; the same retained
frontier is restored between probes.

## Multi-token choices and whole-path tokenization

For every semantic candidate the compiler tokenizes:

```text
continuation_prefix + candidate_text
```

as one complete continuation. Prefix and candidate text must not be tokenized
independently because BPE composition can change at the boundary.

The compiler factors the longest common token prefix. D1 currently requires
that common executable prefix to be non-empty.

After factoring:

- a depth-1 field exposes one token per choice in `candidate_tokens`;
- a multi-token field lowers to a private trie and exposes complete paths in
  `candidate_token_paths`.

Exact-prefix semantic candidates are rejected. For example, if one complete
token path is `[x,y]` and another is `[x,y,z]`, D1 does not invent a
synthetic END choice.

Distinct candidate strings that normalize/tokenize to the same complete token
path are also rejected.

## Reading DecisionFieldResult

With dependencies, `DecisionResult::fields` is emitted in compiled execution
order (the root precedes its children), which need not equal schema insertion
order. The current compatibility result does not expose `SemanticNodeId`, so
use the required-unique field `name` as the practical lookup key rather than
assuming a stable positional index.

For every field:

- `candidate_values` are caller-visible semantic values;
- `probabilities` align by index with `candidate_values`;
- `winner_index` is the authoritative selected index;
- `selected_value` is the corresponding caller-visible value.

For a depth-1 result:

- `candidate_tokens` contains the legal next-token IDs;
- `candidate_token_paths` is empty;
- `winner_token` identifies the winning token.

For a V2-D1 trie result:

- `candidate_tokens` is empty;
- `candidate_token_paths` contains the complete model-facing continuation path
  for each semantic candidate;
- `winner_token == -1`; use `winner_index` / `selected_value`.

`frontier` is the retained prompt frontier restored by the decision
transaction.

`suffix_tokens` is the logical deterministic extension used for capacity
projection. For a trie it is the maximum ambiguity-probe suffix length.

`executed_suffix_tokens` is actual attributed deterministic traversal. It can
be smaller than `suffix_tokens` for a V2-C shared-prefix sibling, or larger
than `suffix_tokens` for D1 tries because independent ambiguity probes replay
full suffixes.

## Asynchronous execution

The async API uses the same execution program:

```cpp
ninfer::DecisionHandle handle =
    engine.submit_decision(std::move(prompt), plan);

ninfer::DecisionResult result = handle.wait();
```

Destroying an unconsumed handle cancels/abandons the request consistently with
the Engine submission model.

## Current backend limits

The semantic IR is intentionally broader than the currently qualified backend.
Compilation/execution currently imposes:

- 1..8 finite-choice nodes per plan;
- at least 2 choices per finite node; the former K=16 product/backend
  ceiling has been removed. Candidate counts are bounded only by representable
  tensor/result indices, unique model token branching, and available runtime
  workspace/resources;
- Boolean choices must be canonical `false,true`, or all choices must be
  Strings for the current `DecisionResult` backend;
- Integer/Number `SemanticValue` kinds are representable in the semantic IR
  but are not currently executable finite-choice result types;
- finite-decision execution requires `SpeculativeBackend::None`;
- dependent plans support one root with direct one-parent children;
- D1 requires a non-empty common executable token prefix;
- exact-prefix and duplicate complete token paths are rejected;
- V2-D1 probe execution is correctness-first and may occupy the executor while
  its ambiguity probes replay; probe sharing/interleaving is D2 optimization
  work, not part of D1.

`FiniteChoice.label` is not semantic identity; `SemanticNodeId` is. The
current compatibility `DecisionResult` does not yet expose node IDs, so the
compiled backend requires labels to be unique and `DecisionFieldResult::name`
is the practical lookup key for now.

## Qualification status

The shared Qwen3.6-family runtime contains the implementation. Current
real-artifact qualification is specifically Qwen3.8-27B on RTX 5080.

The V2-D1 real qualification test covers:

- four multi-token candidates in the multi-depth topology case;
- a 26-way finite choice whose scored ambiguity degree is greater than 16;
- three ambiguity probes across three ambiguity depths;
- shared internal trie prefixes and a deterministic unary tail;
- exact-prefix and distinct-text duplicate-token-path rejection;
- semantic probability normalization and winner mapping;
- dependent trie selection;
- asynchronous `DecisionHandle` execution;
- typed multi-token convenience lowering;
- exact retained-frontier restoration;
- zero committed ordinary decode.

Other family targets share the implementation but are not claimed here as
equally real-artifact qualified.

The D1 execution strategy intentionally replays ambiguity probes independently.
Shared trie traversal is V2-D2 optimization work.

## Related architecture documents

- [Constrained Decisions](CONSTRAINED_DECISIONS.md)
- [Semantic Execution V2 ADR](SEMANTIC_EXECUTION_V2_ADR.md)
- [V2-A Semantic Execution API](SEMANTIC_EXECUTION_V2A_API.md)
- [V2-B Semantic Dependencies](SEMANTIC_EXECUTION_V2B_DEPENDENCIES.md)
- [V2-C Dependency Waves](SEMANTIC_EXECUTION_V2C_WAVES.md)
- [V2-D Multi-token Trie](SEMANTIC_EXECUTION_V2D_TRIE.md)
