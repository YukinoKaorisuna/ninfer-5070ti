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
