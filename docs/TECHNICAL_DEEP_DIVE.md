# Technical Deep Dive

## Objective

Run Qwen3.8-27B on a single RTX 5080 16 GB with all of the following at once:

- genuine 131,072-token max context,
- genuine 131,072-token KV capacity,
- Q4 KV cache,
- MTP-3 speculative decoding,
- relatively high-quality mixed quantization,
- strong long-prompt prefill,
- useful decode speed,
- deterministic correctness.

The project succeeded at an 118,001-token live prompt with **1377.81 tok/s prefill** and **71.51 tok/s decode**.

## 1. Recovering the large-T Q4 attention path

Historical known-good Q4 work showed that the safest routing was not one single implementation for all token widths.

Final behavior:

```text
Q4/Q4 T <= 256 -> legacy/native split-output route
Q4/Q4 T >= 257 -> independent Q4 RowSplit MMA projections
```

Three historical fixes mattered:

```text
3270a698  Use independent Q4 MMA for large-T attention input
5c51901e  Honor output stride in Q4 RowSplit MMA
fb9f560a  Qualify large-T Q4 attention input MMA
```

The threshold is a validated qualification boundary, not a claim that numerical behavior changes monotonically at 256/257.

A key lesson was that synthetic operator tests were insufficient. Small-width tensor-core paths could appear correct in isolated tests while the real model exposed numerical issues. Real-model deterministic oracles became authoritative.

## 2. Why the newer artifact stopped fitting

The current production artifact had drifted away from the historical exact-128K quantization profile.

The older working profile selectively quantized:

```text
24 x GDN value_z
7  x attention gate_value
```

to Q4 while keeping the rest of the relevant model at the higher Q5-class profile.

Historical GPU-weight saving:

```text
value_z      180.000 MiB
gate_value    30.625 MiB
----------------------
total        210.625 MiB
```

The all-Q5-ish artifact failed a large-context runtime reservation by roughly 166 MiB. Restoring the historical mixed-Q4 profile provided enough headroom to recover full 131072 capacity.

## 3. Runtime memory architecture

The historical exact-128K lineage also used runtime-memory optimizations that had been lost or altered.

### Combined backing allocation

Persistent runtime state and workspace were backed by one combined CUDA allocation instead of relying on a second near-capacity `cudaMalloc` late in startup.

This does not magically reduce the planner byte total, but it matters when operating only a few megabytes below physical VRAM capacity because allocation fragmentation/failure risk becomes significant.

### Prefill hidden workspace alias

The full prefill hidden tensor was moved into workspace. Only the final hidden column remained persistent.

This avoided carrying a full `[hidden, prefill_chunk]` persistent allocation throughout runtime.

### First-use prewarms

Critical Q4 paths were prewarmed before the card was almost completely committed, preventing surprise first-use allocations after startup.

## 4. The exact 128K memory result

Final validated startup at:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
```

reported:

```text
GPU weights          ~12.64 GiB
free after weights     2.56 GiB
runtime reservation    2.55 GiB
KV cache payload        2.26 GiB
free after startup     10.56 MiB
planned slack          11.39 MiB
```

The **11.39 MiB** planned slack exactly matched the historical exact-128K documentation found during Git archaeology. That was an important confirmation that the recovered implementation had converged on the intended memory profile rather than merely finding a new accidental fit.

## 5. Large-prompt behavior

The 3,201-token long oracle was intentionally chosen to exercise more than one prefill chunk:

```text
896 / 896 / 896 / 509
```

The runtime showed the expected large-T geometry, including `T=896` and `T=509`.

Result:

```text
prefill: 2236.99 tok/s
historical: 2238.05 tok/s
delta: -0.047%
```

That result effectively reproduced historical performance while using the new current-contract mixed-Q4 artifact at full 131072 capacity.

## 6. Final B133 workload

The final acceptance workload used:

```text
prompt tokens: 118001
max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4
MTP:           3 draft tokens
```

Result:

```text
prefill: 1377.81 tok/s
decode:    71.51 tok/s
MTP acceptance rate: 44.74%
MTP acceptance length: 2.31
```

Historical comparison:

```text
older baseline:      1235.03 tok/s
historical optimized 1371.10 tok/s
historical B133:     1377.66 tok/s
final:               1377.81 tok/s
```

The final prefill result is +11.56% over the older baseline and effectively identical to the historical B133 result.

## 7. Why 129024 appeared in earlier tests

`129024 = 144 x 896`, so it aligns perfectly with a 896-token prefill chunk and was useful for historical performance comparisons.

It is not binary 128K.

The final project acceptance target is explicitly:

```text
131072 max context
131072 KV capacity
```

This distinction is important when comparing public long-context claims.

## 8. Artifact quality

The final artifact is deliberately mixed rather than globally reduced to Q2/Q3.

It keeps the majority of the main model at the higher Q5-class profile and selectively spends Q4 only where it buys the memory needed for 128K.

The `.ninfer` file size is 16,461,267,456 bytes (~15.33 GiB). A naive artifact-size / parameter-count calculation gives an effective storage figure around the mid-4-bit range, but this should not be confused with a pure GGUF-style neural-weight BPW because the artifact also contains scales, metadata, non-quantized tensors and auxiliary model/runtime data.

For quality comparisons, the actual tensor-level quantization profile is more meaningful than treating the entire file as one uniform BPW number.
