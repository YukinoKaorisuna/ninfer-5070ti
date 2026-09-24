# V2-C Dependency Waves and Shared Conditioning

> **Status:** V2-C1/C2 implemented. Shared-frontier waves currently optimize
> compatible depth-1 sibling variants. V2-D1 trie-containing groups use the
> correctness-first sequential trie path; shared trie traversal is V2-D2 work.
> See [Constrained Decisions: current API and usage](CONSTRAINED_DECISIONS_USAGE.md).

Status: Accepted for implementation

## Objective

V2-C extends semantic dependency execution from one root-to-child edge to
one-parent fan-out and then optimizes repeated parent-conditioned traversal.

The semantic shape is:

        A
      / | \
     B  C  D

B, C and D each depend semantically on A.

Multiple-parent nodes remain deferred.

## Governing invariant

Semantic dependency does not imply:

- execution width;
- materialization strategy;
- rendering order;
- output serialization;
- persistent model state.

The compiler/runtime chooses the execution strategy.

## V2-C1: fan-out correctness

The existing internal DecisionExecutionProgram is sufficient.

Conceptually:

    node 0:
        root A

    node 1:
        child B
        parent_result_index = 0

    node 2:
        child C
        parent_result_index = 0

    node 3:
        child D
        parent_result_index = 0

Each dependent child carries one compiled variant per parent choice.

The executor resolves each child's variant using the already-computed parent
winner.

V2-C1 uses the existing replay-from-retained-frontier transaction.

Execution is initially serial:

    score A
    restore retained frontier

    replay selected-A conditioning + B path
    score B
    restore retained frontier

    replay selected-A conditioning + C path
    score C
    restore retained frontier

    replay selected-A conditioning + D path
    score D
    restore retained frontier

V2-C1 requires no Qwen target/runtime, CUDA, MTP or Vision changes.

## Whole-path tokenization

Every dependent child path remains tokenized as a complete continuation.

For parent alternative p and child candidate c:

    conditioning[p]
    + child.continuation_prefix
    + child.candidate_text[c]

is tokenized as one path.

V2-C shared-prefix factoring occurs only after these complete paths have been
tokenized.

Textual-prefix equality must never substitute for token-path equality.

## V2-C2: shared-conditioning optimization

After the parent winner is known, the compiler/runtime may inspect the selected
variants of all sibling nodes.

It may compute a token-level common deterministic prefix shared by those
selected sibling variants.

This prefix may include:

    parent conditioning
    +
    any additional child-shared token prefix

The optimizer is therefore not restricted to the textual dependency
conditioning string.

## Replay strategy

Replay remains the correctness baseline.

It requires no persistent temporary frontier and is preferred when:

- sibling count is small;
- shared conditioning is short;
- snapshot/materialization overhead dominates;
- transient memory is constrained.

## Materialized shared frontier

When profitable, the runtime may execute the selected siblings as:

    retained prompt frontier
            |
    deterministic shared-prefix advance
            |
    temporary selected frontier
         /     |     \
        B      C      D
            |
    restore retained prompt frontier

The temporary frontier is backend execution state only.

It is not:

- a semantic node;
- public API state;
- output rendering state;
- a second retained sequence.

## Nested state ownership

The current one-level decision probe is insufficient for shared-frontier
materialization because two reversible levels are required:

1. original retained frontier;
2. temporary selected/shared frontier.

The materialized strategy therefore requires distinct state ownership for:

    outer snapshot:
        original retained recurrent/GDN state

    inner snapshot:
        temporary shared-frontier recurrent/GDN state

Sibling probes restore to the inner temporary frontier.

Wave completion restores the outer retained frontier.

The ordinary rewrite checkpoint must not be repurposed for this mechanism.

## KV ownership

The materialized selected frontier may keep its temporary attention KV resident
while sibling nodes execute.

The design must not:

- allocate a second sequence lane;
- duplicate the retained attention prefix;
- publish temporary conditioning to the output ledger;
- append temporary conditioning to prefix identity;
- make the temporary frontier a retained user-visible sequence state.

Temporary KV growth is discarded when the wave transaction ends.

## Retained-frontier metadata

V2-C2 must not fake a temporary frontier by partially mutating the existing
retained SequenceState.

The retained invariants involving:

- execution_frontier;
- ledger_frontier;
- ledger;
- prefix_identity;
- text_kv_valid;
- tail_hidden_valid;

remain authoritative for the retained prompt frontier.

Temporary decision-frontier metadata must be represented separately inside the
target/runtime transaction.

## Cost model

Execution strategy remains private.

Conceptually:

    replay_work =
        sum(selected sibling suffix work)

    materialized_work =
        shared_prefix_work
        + sum(residual sibling suffix work)
        + nested snapshot/restore overhead

The runtime may also account for:

- available VRAM;
- temporary KV pages;
- recurrent/GDN snapshot bytes;
- sibling count;
- shared-prefix length;
- current request concurrency;
- workspace pressure.

Materialization is selected only when cheaper and memory-safe.

## Wave width

Semantic sibling count is not execution width.

The runtime may use:

- serial execution;
- bounded-width microbatching;
- full sibling batching;

without changing the semantic graph.

V2-C2 initially proves shared-frontier reuse with serial sibling probes.

GPU parallel sibling scoring is deferred until there is measured benefit and a
clean target interface.

## Scheduler accounting

V2-C1 conservatively reserves the sum of the worst-case selected variant work
for the root and all siblings.

For V2-C2 the planner may reduce projected work when a compiled shared prefix is
materialized once.

Context-capacity accounting remains based on the maximum simultaneously
materialized frontier extension rather than summing sequential sibling paths.

## Cancellation and failure

Cancellation may occur between sibling nodes.

Every V2-C strategy must restore the original retained frontier exactly on:

- success;
- cancellation;
- child failure;
- scoring failure;
- runtime exception.

No temporary selected frontier survives request completion.

## MTP, Vision and grammar

V2-C does not alter:

- MTP;
- Vision;
- grammar execution;
- constrained-choice CUDA;
- external protocol adapters.

Finite semantic waves remain independent of speculative/free-generation state.

## Acceptance: V2-C1

A real Qwen qualification must demonstrate:

    one root FiniteChoice
        ->
    at least two sibling dependent FiniteChoice nodes

with:

- one shared parent winner;
- correct winner-directed variant selection for every child;
- whole-path tokenization parity against independently compiled equivalents;
- semantic/presentation separation preserved;
- all sibling frontiers equal to the retained prompt frontier;
- committed normal decode tokens equal to zero;
- one sequence lane;
- no persistent candidate state;
- existing V2-A and V2-B regressions passing.

The qualification records total replayed dependent suffix work.

## Acceptance: V2-C2

Using the same semantic fixture, compare:

    replay baseline

against:

    shared-frontier materialization

and require:

- exactly identical semantic results;
- matching candidate probabilities within existing deterministic tolerance;
- lower executed deterministic conditioning work when common prefixes exist;
- exact final retained-frontier restoration;
- no additional retained sequence lane;
- no ordinary decode;
- production restoration after qualification.

Only after V2-C2 is qualified should wider sibling batching be considered.
