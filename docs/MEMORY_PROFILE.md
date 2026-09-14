# Memory Profile

The final result is a near-capacity fit on a 16 GB RTX 5080, so the memory profile is part of the design rather than an incidental detail.

## Selective mixed-Q4 profile

The historical exact-128K lineage used selective Q4 quantization for the tensors that bought the most useful headroom while allowing the rest of the main model to remain at the higher Q5-class profile.

Validated mixed-Q4 counts:

```text
24 x GDN value_z
7  x attention gate_value
```

Historical weight-memory savings:

| Tensor group | Saving |
|---|---:|
| `value_z` | 180.000 MiB |
| `gate_value` | 30.625 MiB |
| **Total** | **210.625 MiB** |

This was decisive. The all-Q5-ish current artifact was short by roughly 165.96 MiB during the earlier large-context runtime reservation attempt.

## Final artifact

```text
file bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The file is approximately 15.33 GiB on disk. File-size-derived BPW should be treated cautiously because the artifact includes more than only quantized neural-network weights.

## Runtime memory at true 128K

Final validated configuration:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
```

Observed planner/runtime values:

| Item | Value |
|---|---:|
| GPU weights | ~12.64 GiB |
| Free after weights | 2.56 GiB |
| Runtime reservation | 2.55 GiB |
| KV cache payload | 2.26 GiB |
| Free after startup | 10.56 MiB |
| Planned slack | **11.39 MiB** |

## Why combined backing matters

The known-good runtime used one combined CUDA backing allocation for persistent data and workspace. This does not reduce the planner's required byte count, but it avoids relying on a second very large device allocation after VRAM is already heavily committed.

With only around 11 MiB of planned slack, allocation layout and timing become material to reliability.

## Prefill hidden workspace alias

The full prefill hidden tensor is workspace-backed. Only one final hidden column needs to remain persistent.

At a prefill chunk of 896, retaining an unnecessary full hidden-width persistent buffer would consume several additional MiB—large enough to matter when the final margin is only about 11 MiB.

## First-use prewarms

Critical Q4 paths are prewarmed before the card is fully committed. This avoids first-use CUDA allocations becoming unexpected late-startup failures.

## Practical lesson

At ordinary VRAM utilization, tens of MiB often do not matter. At this utilization level they are the difference between:

```text
131072 / 131072 starts successfully
```

and

```text
runtime reservation fails before inference begins
```
