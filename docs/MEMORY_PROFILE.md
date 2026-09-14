# Memory Profile

The final result is a near-capacity fit on a 16 GB RTX 5080, so the memory profile is part of the design rather than an incidental detail.

## Mixed Q3/Q4/Q5 profile

The final text-model core is a mixed **Q3/Q4/Q5 groupwise** profile, not a predominantly-Q5 model.

Approximate parameter-weighted distribution:

| Format | Share of main-model parameters | Encoded storage |
|---|---:|---:|
| Q3G64_F16S | **42.42%** | 3.25 bpw |
| Q4G64_F16S | **45.92%** | 4.25 bpw |
| Q5G64_F16S | **11.57%** | 5.25 bpw |
| BF16 / FP32 | ~0.10% | small norms / miscellaneous tensors |

Main-model accounting:

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953

quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

For comparison with GGUF-style BPW claims, the appropriate public shorthand is **~3.95 BPW effective main-model quantization**.

## Exact-128K selective Q4 placements

Within that broader mixed profile, the exact-128K recovery restored these specific Q4 tensor placements:

```text
24 x GDN value_z
7  x attention gate_value
```

Historical weight-memory savings from those placements:

| Tensor group | Saving |
|---|---:|
| `value_z` | 180.000 MiB |
| `gate_value` | 30.625 MiB |
| **Total** | **210.625 MiB** |

This was decisive. The heavier comparison artifact was short by roughly 165.96 MiB during the earlier large-context runtime reservation attempt.

## Final artifact

```text
file bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The file is approximately 15.33 GiB on disk. Dividing the entire `.ninfer` container size by a headline model parameter count is **not** a GGUF-comparable BPW calculation because the artifact contains auxiliary/non-main-model content in addition to the text-core weight storage.

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

## Measured clean-start requirement

The true-128K profile should be treated as requiring an effectively clean RTX 5080 at process start.

A successful `ninfer-serve` launch with the full `131072 / 131072`, Q4 KV, MTP-3 and chunk-896 configuration began from this NVIDIA-SMI state:

```text
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB
```

After the ~12.64 GiB weights were resident and all required prewarms had run, the measured free memory was:

```text
free before runtime reservation = 2624.56 MiB
runtime reservation             = 2613.17 MiB
planned slack                   = 11.39 MiB
free after startup              = 10.56 MiB
```

The server then successfully reached the listening state without reducing context, KV capacity, MTP draft width or prefill chunk.

This means the practical memory requirement is not simply “an RTX 5080 with 16 GB.” The process needs essentially the entire usable framebuffer at startup. A competing CUDA process or other GPU allocation can consume the ~11 MiB margin and cause the runtime-reservation check to fail.

Recommended preflight check:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

For comparable reproduction, record this pre-launch GPU memory state alongside the benchmark configuration.

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
