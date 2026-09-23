# constrained_choice

`constrained_choice` is the finite-choice scorer used by constrained semantic
decision execution.

It consumes the model's existing full-vocabulary BF16 logits, gathers only the
explicit legal token IDs, and computes a softmax over that finite domain.

Current contract:

- batch `B`: 1..8;
- candidate count `K`: 2..16;
- candidate IDs: I32 `[K,B]`;
- probabilities: FP32 `[K,B]`;
- winners: I32 `[B]`;
- equal maxima select the lowest candidate index.

The wrapper validates tensor shape/type/aliasing. Decision runtime callers also
validate candidate token domains and uniqueness before the CUDA operation is
launched.

The operation is used by both reversible single-field decision probes and
V2-C shared-frontier sibling waves. V2-D multi-token tries reuse the same op at
each ambiguity node; deterministic unary trie traversal is not scored.

The current implementation still consumes a full-vocabulary LM-head result.
Restricted-row LM-head projection is a separate future optimization.

Numerical coverage lives in `tests/ops/test_constrained_choice.cpp`; the
microbenchmark lives in `bench/ops/constrained_choice_bench.cu`.
