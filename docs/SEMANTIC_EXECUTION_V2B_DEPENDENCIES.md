# V2-B Semantic Dependency Execution

> **Status:** historical V2-B design record; implemented and extended by V2-C
> fan-out/shared waves and V2-D1 multi-token trie variants. The current compiled
> backend supports one semantic root with direct one-parent children. See
> [Constrained Decisions: current API and usage](CONSTRAINED_DECISIONS_USAGE.md).

Status: Accepted for implementation

## Scope

V2-B introduces the first semantic dependency execution path.

The narrow proof is:

    root FiniteChoice
            |
            v
    select semantic winner
            |
            v
    select parent-conditioned child plan
            |
            v
    deterministic conditioning replay
            |
            v
    dependent FiniteChoice

V2-B preserves the governing principle:

> Keep semantic constraints. Eliminate representational generation.

No selected semantic representation is sampled or emitted merely to establish
model state for a dependent decision.

## Key runtime decision

V2-B does not introduce a persistent semantic working frontier.

The current Qwen constrained-decision probe already provides a reversible
transaction against one retained prompt frontier:

- capture recurrent/GDN state;
- bind the retained KV allocation;
- execute explicit suffix tokens without sampling;
- score constrained candidate logits;
- restore recurrent/GDN state;
- trim KV back to the retained frontier;
- restore entitlement/bindings.

The ordinary forward path used by that probe deliberately performs no sampling
and does not publish continuation hidden or mutate the retained output ledger.

V2-B therefore reuses the existing probe transaction.

After a parent winner is known, the executor selects a compiled child variant
whose deterministic suffix contains the chosen parent's downstream
conditioning plus the child's common presentation prefix.

The child probe replays that deterministic path from the retained prompt
frontier, scores the child candidates and restores the frontier.

This is called:

    replay-from-retained-frontier dependency execution

It is not ordinary autoregressive generation.

## Why V2-B does not commit a persistent working frontier

Making a probed continuation permanently authoritative would require defining
and updating at least:

- execution_frontier;
- ledger_frontier;
- output ledger;
- prefix identity;
- text KV validity;
- recurrent/GDN current-state authority;
- tail hidden validity;
- KV page entitlement;
- cancellation restoration;
- checkpoint ownership.

That is unnecessary for the first dependency proof.

Replay from the existing retained frontier keeps the current transactional
correctness boundary and avoids introducing persistent candidate state.

A later optimization may materialize a selected semantic frontier when its
benefit is measurable.

Such materialization is an execution-plan optimization and does not alter
semantic dependency meaning.

## Semantic dependency

Semantic dependency remains part of StructuredDecisionSchema.

Conceptually:

    schema.add_dependency(parent, child);

means:

    child semantically depends on the selected value of parent.

Rendering order is not implied.

Model-conditioning text is not implied.

V2-B implementation initially supports only the narrow backend shape needed for
one-parent dependency execution.

Multiple-parent semantics and arbitrary dependency graphs remain deferred
backend capabilities.

## Model-conditioning dependency

Model conditioning remains separate from semantic dependency.

The presentation layer may associate an edge with a model-facing
representation of each selected parent alternative.

Conceptually:

    struct DependencyConditioningPresentation {
        std::vector<std::string> selected_choice_texts;
    };

    presentation.set_dependency_conditioning(
        parent,
        child,
        conditioning);

`selected_choice_texts[i]` is the deterministic model-facing text used when the
parent's semantic choice index is `i`.

It is not:

- the semantic value itself;
- required to equal the parent's scoring candidate text;
- output rendering;
- a token sequence.

An empty conditioning string is semantically valid and means that the
dependency affects execution ordering but contributes no parent-derived model
text at this edge.

For V2-B a conditioning source must be the declared semantic parent of the
child.

## Scoring presentation and downstream conditioning presentation

The representation used to score a parent choice and the representation used
to condition descendants are explicitly different concepts.

For example:

    semantic value:
        "search"

    parent scoring representation:
        "search"

    child-conditioning representation:
        " selected tool: search"

No implicit conversion from scoring presentation to descendant conditioning is
performed.

Adapters may intentionally supply the same text for both when appropriate.

## Dependency whole-path tokenization

Tokenizer correctness extends across the complete dependent continuation.

For parent choice `p` and child candidate `c`, compilation constructs:

    conditioning[p]
    + child.continuation_prefix
    + child.candidate_text[c]

and tokenizes that complete string as one path.

For each parent alternative, the compiler computes the longest common token
prefix across all child candidate paths.

That shared prefix becomes the deterministic suffix executed by the child
probe.

The remaining divergent token is scored by the current one-token finite-choice
backend.

Separately tokenizing:

    conditioning[p]

and:

    child.continuation_prefix + child.candidate_text[c]

is not permitted.

Current one-token divergence remains a backend restriction rather than a
semantic FiniteChoice restriction.

## Compiled dependency representation

CompiledDecisionPlan remains opaque.

Conceptually a dependent finite node contains:

    node identity
    semantic parent identity
    child execution variants indexed by parent winner

A root finite node has one compiled field.

A dependent child with N parent alternatives has N compiled field variants.

Each variant is an ordinary backend DecisionFieldSpec representing the correct
whole-path tokenization for that parent result.

These variants are compiler/runtime details and are not exposed in the semantic
schema.

## V2-B execution

The initial execution algorithm is:

    probe root from retained prompt frontier
    restore retained prompt frontier
    record root winner index

    select child variant[root winner index]

    probe selected child variant from retained prompt frontier
    restore retained prompt frontier
    record child winner index

No selected parent candidate token is sampled.

No parent representation is published as generated output.

No second sequence lane is required.

No permanent per-candidate KV or recurrent/GDN state is retained.

## Scheduler accounting

A dependent node has multiple compiled variants but only one variant executes.

Scheduler service-work accounting must therefore reserve for the executed
worst-case path, not sum mutually exclusive variants.

For the narrow root-to-child case the projected decision work is conceptually:

    root suffix work
    +
    max(child variant suffix work)
    +
    existing decision prefill-output work

Backend accounting must remain conservative without treating every
parent-conditioned child variant as simultaneously executed work.

Admission remains one active lane in V2-B.

## Cancellation and failure

Each probe remains transactionally reversible.

Failure or cancellation after the root result but before child completion must
leave the retained sequence at the original prompt frontier.

V2-B does not create a second persistent state that requires independent
cleanup.

## Independent siblings

The architecture remains compatible with:

          A
         / \
        B   C

After A is selected, B and C may initially replay their own parent-conditioned
paths from the same retained frontier.

V2-B does not require concurrent sibling execution.

A later execution-wave optimizer may:

- factor their common selected-parent prefix;
- materialize one temporary selected frontier;
- batch sibling probes;
- microbatch them according to available memory.

Those optimizations must not require changes to semantic dependency APIs.

## Multiple parents and deep chains

V2-B does not implement multiple-parent conditioning.

It also does not require general deep dependency-chain execution.

For:

    C depends on A and B

the ordering and model representation of A and B must be explicit rather than
derived from arbitrary topological order.

That design remains deferred.

Deep dependency execution must also avoid exponential compiled-path expansion.

The V2-B root-to-child proof must not freeze an architecture that requires such
expansion later.

## Relation to deterministic state advancement

A generic future operation:

    advance deterministic token path

remains a useful execution concept.

V2-B does not need to expose or persist such an operation independently because
the current child probe already performs deterministic advancement followed by
finite scoring inside one reversible transaction.

A future optimizer may split that transaction into:

    deterministic advance
        +
    one or more dependent probes

when shared-prefix reuse makes that cheaper.

## Cross-engine implications

Parallel finite-decision systems that score independent fields from one shared
trunk validate the value of shared prefill and branch-specific suffixes.

Their independent-field assumption does not by itself solve semantic
dependencies.

Grammar engines demonstrate useful mechanisms such as:

- state rollback;
- state fork;
- deterministic jump-forward;
- speculative-state traversal.

V2-B borrows the concepts of reversible and deterministic state movement but
does not encode semantic dependency as a serialized grammar state machine.

Finite semantic nodes continue to bypass token-by-token grammar masking.

## MTP

V2-B does not change MTP.

FiniteChoice remains outside speculative generation.

Future FreeGeneration descendants may use MTP independently.

The replay-from-retained-frontier dependency mechanism must not place MTP state
in the public semantic schema.

## Vision

V2-B does not change Vision.

Multimodal evidence remains part of the prepared prompt context.

Semantic dependency execution begins from the already-established retained
prompt frontier.

No Vision-specific dependency node is introduced.

## Restricted-row LM head

Dependent FiniteChoice nodes retain explicit finite candidate domains.

Therefore the V2-B design preserves future restricted-row LM-head projection.

Dependency execution does not require falling back to full-vocabulary grammar
masking.

## V2-B implementation boundary

The first implementation should add:

- semantic dependency storage/validation;
- separate dependency-conditioning presentation;
- per-parent child variant compilation;
- whole dependent-path tokenization;
- winner-directed child-variant execution;
- conservative mutually-exclusive service-work accounting;
- host/compiler tests;
- one real root-to-child Qwen qualification.

It should reuse:

- existing constrained-choice CUDA;
- existing reversible decision frontier;
- existing ordinary_forward_batch;
- existing one-token scorer;
- existing retained sequence;
- existing DecisionResult for this milestone.

It must not require changes to:

- constrained-choice CUDA kernels;
- ordinary generation;
- MTP;
- Vision;
- production protocol adapters.

## V2-B acceptance

V2-B is accepted when a real Qwen test demonstrates:

    root FiniteChoice
        ->
    selected parent semantic value
        ->
    parent-specific deterministic conditioning replay
        ->
    dependent FiniteChoice

with:

- parent winner mapped correctly;
- correct child variant selected from the parent winner;
- child candidate tokenization matching whole dependent-path tokenization;
- no ordinary sampled generation;
- committed decode tokens remaining zero;
- retained frontier exactly restored;
- no extra sequence lane;
- existing V2-A independent-decision qualification still passing.
