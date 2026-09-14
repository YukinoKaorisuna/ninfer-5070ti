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
