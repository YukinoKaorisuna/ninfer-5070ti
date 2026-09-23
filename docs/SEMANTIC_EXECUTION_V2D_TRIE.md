# V2-D Multi-token FiniteChoice Trie

Status: V2-D1 implemented and real-Qwen qualified; V2-D2 planned

## Objective

V2-D removes the former one-token divergent-branch restriction for
FiniteChoice while preserving the governing semantic-execution rule:

> Keep semantic constraints. Eliminate representational generation.

Finite semantic candidates remain explicitly enumerable semantic alternatives.

They do not become ordinary generation, grammar state, sampled JSON, or
sequence-likelihood ranking.

## Public semantic API

V2-D does not change:

- SemanticValue;
- FiniteChoice;
- StructuredDecisionSchema;
- FiniteChoicePresentation;
- dependency semantics.

Model-facing candidate strings remain presentation metadata.

Trie structure is a private compiled-backend representation.

## Current runtime requirements

V2-D1 finite-decision execution currently requires the ordinary target backend:

    EngineOptions::speculative.backend == SpeculativeBackend::None

The Engine rejects finite-decision compilation/execution on MTP/DFlash
configurations rather than silently changing the configured speculative mode.
MTP remains complementary for future/open-ended generation; it is not the
current finite-choice probe backend.

The current implementation adds the C++ Engine API only. It does not add an
HTTP
`/v1/decision` endpoint or change OpenAI/Anthropic protocol adapters.

The implementation is in the shared Qwen3.6-family runtime. Real-artifact D1
qualification in this branch is specifically Qwen3.8-27B on RTX 5080.

## Whole-path tokenization

For every candidate c:

    continuation_prefix + candidate_text[c]

is tokenized as one complete continuation.

Independent prefix/candidate tokenization is not authoritative because BPE
composition may differ at the boundary.

The compiler first obtains all complete token paths.

Only then may it factor common token structure.

## Global deterministic prefix

The longest common token prefix across all candidate paths is deterministic
model conditioning.

It is not a semantic choice.

The compiler factors it before constructing semantic branch probes.

For V2-D1 the backend continues to require a non-empty executable prefix.

## Exact-prefix rejection

V2-D1 rejects:

    path A = [x, y]
    path B = [x, y, z]

when one semantic candidate token path is an exact prefix of another.

At the terminal prefix there is no model token expressing:

    choose terminal semantic value

versus:

    continue longer semantic value

unless the presentation supplies an explicit differentiating token.

V2-D1 does not invent a synthetic semantic END token.

Presentation adapters may instead choose unambiguous model-facing text.

Identical complete token paths for different semantic candidates are also
rejected.

## Compiled trie semantics

After global-prefix factoring, the compiler constructs a token trie.

Trie edges are model tokens.

A node with one outgoing edge represents deterministic representation.

A node with two or more outgoing edges represents finite semantic uncertainty.

Only ambiguous nodes require model scoring.

Unary paths are folded into deterministic traversal toward the next ambiguous
node or terminal candidate.

## D1 lowered representation

The runtime does not need a pointer-heavy trie.

The temporary compiler trie lowers to an internal DecisionTriePlan containing:

    candidate_token_paths

and a list of ambiguous probes.

Each ambiguous probe contains:

    suffix_tokens
        full deterministic path from the retained decision frontier to this
        ambiguity point;

    candidate_tokens
        distinct outgoing edge tokens at this ambiguity point;

    descendant candidate mapping
        for each outgoing edge, the semantic candidate indices below it.

The probe list is backend-only.

With K semantic leaves, the number of ambiguous branch nodes is at most K - 1.

The current backend limit K <= 16 therefore implies at most 15 ambiguous D1
probes per finite-choice variant.

## Probability semantics

Each ambiguous trie node is scored with the existing restricted constrained
softmax over that node's outgoing legal tokens.

For semantic candidate c:

    P(c) =
        product of branch probabilities
        along ambiguous nodes on c's route

Deterministic one-edge traversal contributes factor 1.

Therefore deterministic token count does not directly penalize a semantic
candidate.

After accumulation, candidate probabilities may be normalized once to remove
floating-point accumulation drift.

The selected semantic candidate is the candidate with maximum final
probability.

Ties select the lowest semantic candidate index, preserving existing finite
decision tie behavior.

These probabilities describe the constrained finite-choice distribution.

They are not calibrated confidence or probability of correctness.

## Why not full sequence likelihood

V2-D deliberately does not score every deterministic token of every rendered
candidate and multiply ordinary LM token probabilities.

Doing so would make semantic probability depend on representational token
length and BPE accidents.

A longer deterministic label could be penalized despite containing no
additional semantic uncertainty.

That conflicts with semantic execution.

## V2-D1 execution strategy

D1 prioritizes correctness and architectural separation.

Each ambiguous trie probe executes independently from the retained decision
frontier using the existing reversible:

    decision_probe_lane(
        suffix_tokens,
        candidate_tokens)

transaction.

Therefore D1 initially performs:

    retained frontier
        -> deterministic path to ambiguous node
        -> restricted branch score
        -> restore retained frontier

for each ambiguous node.

No second retained sequence lane is created.

No candidate path is sampled.

No branch representation is committed to output.

## D2 optimization boundary

D2 may optimize D1 replay using the already-qualified V2-C2 machinery.

Possible optimizations include:

- materializing shared trie prefixes once;
- restoring child probes to temporary trie frontiers;
- grouping sibling ambiguity nodes;
- reducing repeated deterministic traversal.

D2 must preserve exactly the D1 semantic probabilities.

Trie materialization remains a runtime/backend decision.

## Internal execution representation

DecisionExecutionProgram remains private.

V2-D introduces an internal execution variant capable of representing:

    ordinary one-token DecisionFieldSpec

or:

    DecisionFieldSpec semantic metadata
    + DecisionTriePlan

Raw existing DecisionFieldSpec callers remain supported as one-token execution
variants.

Existing V2-A/B/C compiled plans remain valid depth-1 finite choices.

## Result metadata

Existing one-token fields preserve:

    candidate_tokens

exactly as today.

Multi-token trie results additionally expose:

    candidate_token_paths

where each path is the complete model-facing token path corresponding to the
semantic candidate.

For trie results:

    winner_index
    selected_value

remain authoritative semantic result fields.

A single winner_token cannot faithfully identify an arbitrary multi-token
semantic candidate.

The backend must not fabricate a misleading token identifier.

## Scheduler accounting

For a one-token variant:

    service_work =
        existing suffix work

For a D1 trie variant:

    service_work =
        sum(full suffix length of every ambiguous probe)

because all ambiguity probes execute to construct the complete semantic
candidate distribution.

For a semantic dependency node with mutually-exclusive parent variants, reserve:

    max(service_work of each parent-selected variant)

rather than summing mutually-exclusive variants.

Context-capacity projection uses:

    max suffix length of any single probe

because D1 restores to the retained frontier after each probe.

D2 may later reduce service-work projection when shared trie traversal is
materialized.

## Result construction

For K semantic candidates:

    probability[K] = 1

For each ambiguous probe:
    score its outgoing tokens;
    for each edge e:
        multiply every descendant semantic candidate by P(e)

Normalize accumulated candidate probabilities for floating-point drift.

Select argmax with lowest-index tie break.

Deterministic tails do not create scoring events.

## Interaction with V2-C

V2-D1 does not combine trie execution with V2-C2 shared-sibling waves.

If a selected execution variant is a trie, the executor uses the D1 trie path.

Existing V2-C2 optimization remains available to ordinary one-token sibling
variants.

V2-D2 may reuse the same temporary-frontier mechanism inside a trie.

## Cancellation

Cancellation is checked between ambiguous probes.

Every decision_probe_lane transaction independently restores the retained
frontier before returning.

Cancellation therefore leaves no persistent trie state.

## CUDA and target runtime

V2-D1 reuses:

- existing decision_probe_lane;
- existing ordinary_forward_batch deterministic traversal;
- existing constrained_choice CUDA operation;
- existing recurrent/GDN snapshot;
- existing KV entitlement/trim/restore behavior.

V2-D1 does not require changes to:

- constrained_choice CUDA kernels;
- MTP implementation;
- Vision implementation;
- ordinary sampling;
- production protocol adapters.

This means D1 does not mutate those subsystems; it does not mean finite-choice
probes can execute through an MTP/DFlash-configured Engine. They currently
require `SpeculativeBackend::None`.

## V2-D1 qualification

Real Qwen qualification must contain at least four semantic candidates with:

- multi-token candidate paths;
- at least two ambiguity depths;
- at least one deterministic unary tail;
- at least one shared internal trie prefix.

It must demonstrate:

- whole-path tokenization;
- exact-prefix candidate rejection;
- duplicate token-path rejection using byte-distinct presentation strings
  which normalize/tokenize to the same complete model path;
- candidate probability sum approximately 1;
- semantic winner maps correctly;
- deterministic tails create no semantic scoring events;
- one-token V2-A regression passes;
- V2-B dependency regression passes;
- a multi-token trie executes correctly as a dependency-selected variant;
- asynchronous DecisionHandle execution passes;
- the DecisionFieldInput convenience API can lower multi-token enum values to
  the same trie path;
- V2-C1/C2 regression passes;
- zero ordinary committed decode;
- exact retained-frontier restoration;
- one retained sequence lane;
- no MTP/Vision behavior change.

## D1 acceptance telemetry

Qualification should report:

    V2D_CANDIDATE_COUNT
    V2D_AMBIGUOUS_PROBE_COUNT
    V2D_MAX_PROBE_SUFFIX_TOKENS
    V2D_EXECUTED_SUFFIX_TOKENS
    V2D_CANDIDATE_PROBABILITY_SUM
    V2D_DETERMINISTIC_TAIL_SCORE_EVENTS=0
    V2D_NORMAL_DECODE_USED=NO
    V2D_TRIE=PASS

## Future restricted-row LM head

V2-D preserves explicit outgoing finite domains at every ambiguity node.

That is directly compatible with V2-E restricted-row LM-head projection.

V2-D must not replace finite branch domains with full-vocabulary grammar
masking.
