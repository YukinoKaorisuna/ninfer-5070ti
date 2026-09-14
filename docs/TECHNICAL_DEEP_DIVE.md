# Technical Deep Dive

## Objective

Run Qwen3.8-27B on a single RTX 5080 16 GB with all of the following at once:

- genuine 131,072-token max context,
- genuine 131,072-token KV capacity,
- Q4 KV cache,
- MTP-3 speculative decoding,
- mixed Q3/Q4/Q5 groupwise weights,
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

The current production artifact had drifted away from the historical exact-128K memory profile.

The exact-128K recovery restored these specific Q4 placements:

```text
24 x GDN value_z
7  x attention gate_value
```

Historical GPU-weight saving from those placements:

```text
value_z      180.000 MiB
gate_value    30.625 MiB
----------------------
total        210.625 MiB
```

The heavier comparison artifact failed a large-context runtime reservation by roughly 166 MiB. Restoring the historical 128K profile provided enough headroom to recover full 131072 capacity.

Important correction: this does **not** mean the rest of the model is predominantly Q5. The final text core is a mixed Q3/Q4/Q5 groupwise model; see the BPW section below.

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

That result effectively reproduced historical performance while using the current-contract 128K artifact at full 131072 capacity.

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

## 8. GGUF-comparable BPW and artifact quality

The final text core is a mixed **Q3/Q4/Q5 groupwise** profile.

Parameter-weighted distribution:

```text
Q3G64_F16S   42.42%   3.25 bpw encoded
Q4G64_F16S   45.92%   4.25 bpw encoded
Q5G64_F16S   11.57%   5.25 bpw encoded
BF16 / FP32  ~0.10%   small norms / miscellaneous tensors
```

Main text-model accounting:

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953

quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

Therefore, for public comparisons against GGUF claims, the most useful description is:

> **~3.95 BPW effective main-model quantization**

The full `.ninfer` artifact is 16,461,267,456 bytes (~15.33 GiB), but dividing the whole container by a headline model parameter count is not a valid GGUF-style BPW comparison because the artifact includes auxiliary/non-main-model content in addition to the quantized text weights.

The 24 Q4 `value_z` and 7 Q4 `gate_value` placements remain important to the final 128K memory profile, but they are only part of the overall mixed Q3/Q4/Q5 quantization layout.
