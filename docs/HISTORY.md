# Project History

This document captures the engineering journey from a broken long-context state to the validated RTX 5080 true-128K release.

## Starting point

The production tree already contained substantial NInfer work, including Qwen3.8 support, MTP/DFlash2 work and unrelated active development. The live working tree was intentionally left untouched throughout recovery.

All recovery work happened in isolated worktrees.

## Recovering the Q4 lineage

Git history showed a known-good Q4 sequence:

```text
3270a698 Use independent Q4 MMA for large-T attention input
5c51901e Honor output stride in Q4 RowSplit MMA
fb9f560a Qualify large-T Q4 attention input MMA
```

A clean reconstruction was made onto the current production base instead of replaying a broad historical checkpoint wholesale.

This became the clean Q4 candidate and later the 128K memory-fixed candidate.

## Batch 200 — short oracle on recovered Q4 path

The clean candidate passed the deterministic JSON oracle.

Key result:

```text
PREFILL=524.40 tok/s
DECODE=120.93 tok/s
RESULT=PASS
```

The prompt was only 106 tokens, so it did not yet prove the large-T route.

## Batch 201 — large-T long oracle

A 3,201-token prompt forced chunk widths of:

```text
896 / 896 / 896 / 509
```

The recovered path exercised both `T=896` and `T=509` and reached **2226.37 tok/s**, only about 0.52% below the historical 2238.05 tok/s result.

This proved the compute-path recovery.

## Batch 202 — long-context reservation failure

The next step attempted the historical 118,001-token workload with a large KV allocation.

The model failed before startup because the runtime reservation required roughly 166 MiB more than the card had available.

This was the turning point: the compute path was working, but the current artifact no longer had the historical memory profile.

## Git archaeology — finding the exact-128K profile

Historical documentation and source showed that the previous exact-128K model used:

```text
24 x Q4 value_z
7  x Q4 gate_value
```

This saved approximately **210.625 MiB** of GPU weight memory.

History also showed the runtime used:

- combined persistent/workspace backing,
- a workspace-backed full prefill hidden tensor,
- only one persistent hidden column,
- first-use prewarms.

These became the blueprint for the final recovery.

## Converter failure and CPU workaround

The GPU converter failed because PyTorch required a multi-GiB temporary allocation while the 16 GB GPU was already heavily committed.

Conversion was rerun on CPU and succeeded.

The resulting current-contract artifact was:

```text
bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Batch 210 — true 131072 capacity restored

The new mixed-Q4 artifact started successfully with:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
```

The deterministic oracle passed.

Most importantly, the planner reported **11.39 MiB slack**, matching the historical exact-128K documentation.

That was strong evidence the intended memory profile had been recovered.

## Batch 211 — mixed-Q4 long oracle

The 3,201-token oracle was rerun at full 131072 capacity.

Result:

```text
PREFILL=2236.99 tok/s
historical=2238.05 tok/s
difference=-0.047%
RESULT=PASS
```

This proved that the final mixed-Q4 artifact retained the recovered large-T performance.

## Batch 212 — final 118K acceptance

The final workload used an actual **118,001-token prompt** while the full **131,072-token KV capacity** remained allocated.

Result:

```text
PREFILL=1377.81 tok/s
DECODE=71.51 tok/s
RESULT=PASS
```

Historical comparison:

```text
historical B133 prefill=1377.66 tok/s
historical B133 decode=70.24 tok/s
```

The final result effectively reproduced the historical optimized performance while satisfying the stricter true-128K requirement.

## Batch 213 — freeze

The exact working source was tagged:

```text
qwen3.8-27b-rtx5080-128k-v1
```

pointing to:

```text
473dade56031852a7d96edef049d859da96a6df9
```

## Batch 214B — publish

The validated branch and annotated tag were pushed to:

```text
github.com/toddballinger/ninfer-5080
```

No force push was used and the upstream repositories were left untouched.

## Documentation phase

All public documentation is intentionally layered after the frozen validated commit. The release tag therefore continues to identify the exact source that produced the final benchmark.


## v1.3 rolling-tool checkpoint production validation

Long OpenClaw tool loops exposed a prefix-reuse pathology in which
`restore_turn_checkpoint` repeatedly restored the first assistant boundary while the prompt kept
growing. The OpenClaw request payload was verified to remain exact-prefix append-only; the stale
checkpoint was therefore traced to Qwen frontend checkpoint placement rather than client history
ordering.

A configurable checkpoint policy was backported to the v1.3 release line:

```text
--prefix-checkpoint-policy stable-turn|rolling-tool
```

`stable-turn` remains the general default. The validated OpenClaw production profile uses
`rolling-tool`.

Validated runtime commit:

```text
ceb32f7d002edab224a83a2e2609f45fca4f8919
```

Validated production server SHA256:

```text
3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
```

The production smoke test advanced the restored checkpoint:

```text
19023 -> 21146 -> 24664 -> 26641
```

with zero plateaus or regressions. This removed the previously observed fixed-checkpoint growth
pattern while preserving new-user-turn safety and the existing `preserve_thinking=true`
response-replay behavior.

## PR #3 — `9e163eee` semantic port

Upstream commit `9e163eee4b8acec21ab0ac765107b6a3f287b217` was integrated as RTX 5080-specific semantic port:

```text
4b62aca386a0a214049201ebeb2a422b0cb609ce
```

and merged through PR #3 as:

```text
33546d7d5be6d82eaac5e4a87a3f7e578f8a1a13
```

The port retained fork-specific Q4/Q4 routing, Q4 GDN `value_z` handling, A8 behavior, 4096-geometry routing and workspace-aware execution while adopting the upstream Q4/Q5 column-band routing mechanism.

The exact 118,001-token workload was run three times on the merge lineage. The mean was 1378.263 tok/s prefill and 71.467 tok/s decode, with 44.74% MTP acceptance and 2.31 tok/round. This was within normal run-to-run variation relative to the strongest v1.2 reference.

## PR #5 — rolling-tool reconciliation into the PR #3 lineage

The validated v1.3 rolling-tool feature had existed on the release branch but was missing from the later `main` lineage. PR #5 reconciled that feature without removing the PR #3 routing work.

The merge commit was:

```text
b44b1958c301ec6bf4d18973a97d7b42fa6733aa
```

That exact runtime tree was rebuilt and qualified with the historical 118,001-token workload at 131,072 context and 131,072 Q4 KV:

```text
PREFILL=1378.85 tok/s
DECODE=71.44 tok/s
MTP_ACCEPTANCE=44.74%
MTP_LENGTH=2.31 tok/round
RESULT=PASS_EQUIVALENT_WITHIN_NOISE
```

The source tree was restored cleanly after the temporary benchmark CLI patch, and the production v1.3 service was restored after the clean-GPU run.

## PR #4 — documentation-only descendant

PR #4 was merged after PR #5 as:

```text
c8439fbcb89a4daf74cf2692a9425930998c763f
```

The only change from `b44b1958` to `c8439fb` was documentation in `docs/BENCHMARKS.md`. For that reason, runtime performance qualification remains attributed to `b44b1958`; `c8439fb` is recorded as a documentation-only descendant rather than being described as if it were independently benchmarked.

## `a9a0d10a` — Q5 A16 LinearAdd semantic port

Upstream `a9a0d10a933713d3066110a3caa4663c482319da` retuned Q5 A16
LinearAdd at T=1 and for partial >512-column waves. A direct cherry-pick
was rejected because upstream also removed residual-GEMV infrastructure
still required by the fork's 4096-row paths and used route boundaries that
would have overwritten RTX 5080-specific C64 crossover tuning.

The mechanism was therefore ported manually as:

```text
00e8e47fa6001067257f7ae6594c2deeabaed590
```

The port keeps 4096-row T=1 on residual GEMV, makes the historical 4096
route policy explicit, retains the fork's C64 crossover bands, enables T=1
Split2 only for the two 5120-row shapes, and adds >512 narrow-tail
decomposition when the remainder is at most 192 columns.

RTX 5080 operator A/B testing showed 25-31% gains at T=1, 34-35% gains at
T=513, about 24% gains at T=1025 and useful gains at the observed T=621
remainder, while T=896 and other fallback widths remained effectively
unchanged.

The committed tree then passed the exact 118,001-token / 131,072-context /
131,072-Q4-KV workload at:

```text
PREFILL=1376.30 tok/s
DECODE=71.53 tok/s
MTP_ACCEPTANCE=44.74%
MTP_LENGTH=2.31 tok/round
RESULT=PASS_EQUIVALENT_WITHIN_NOISE
```

Relative to `b44b1958`, this is -0.185% prefill and +0.126% decode with
unchanged memory usage and MTP behavior.
