# NInfer RTX 5080 — Qwen3.8-27B at true 128K on 16 GB

This repository preserves and documents a validated NInfer configuration for running **Qwen3.8-27B** on a single **NVIDIA GeForce RTX 5080 16 GB** with a genuine **131,072-token context/KV capacity**.

## Headline result

Final validated workload:

- GPU: **RTX 5080 16 GB**
- Model: **Qwen3.8-27B**
- Main-model effective precision: **~3.95 BPW** (GGUF-comparable weighted storage figure)
- Max context: **131,072**
- KV capacity: **131,072**
- KV dtype: **Q4**
- Speculation: **MTP-3**
- Prompt length: **118,001 tokens**
- Prefill: **1,377.81 tok/s**
- Decode: **71.51 tok/s**
- GPU weights: **~12.64 GiB**
- KV payload: **2.26 GiB**
- Planned slack: **11.39 MiB**

This was not a reduced-KV or short-context benchmark. The full 131,072-token KV capacity remained allocated during the 118,001-token test.

## Canonical validated release

The exact tested source is frozen at:

```text
commit: 473dade56031852a7d96edef049d859da96a6df9
tag:    qwen3.8-27b-rtx5080-128k-v1
```

Model artifact:

```text
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
bytes:  16461267456
```

Validated binary:

```text
SHA256: b38e987a16f7cdda3c5eee81b0ac821f9f2aad86f117b745401a9ef3b5c43012
```

The release tag remains on the exact source that produced the final benchmark. Documentation lives on later commits/branches so the tested state stays immutable.

## Quantization profile and BPW

The final text-model core is a mixed **Q3/Q4/Q5 groupwise** profile, not a predominantly-Q5 model.

Approximate parameter-weighted distribution of the main text model:

| Format | Share of main-model parameters | Encoded storage |
|---|---:|---:|
| Q3G64_F16S | **42.42%** | 3.25 bpw |
| Q4G64_F16S | **45.92%** | 4.25 bpw |
| Q5G64_F16S | **11.57%** | 5.25 bpw |
| BF16 / FP32 | ~0.10% | small norms/other tensors |

Main text-model accounting:

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953

quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

For public GGUF-style comparisons, **~3.95 BPW effective main-model quantization** is the appropriate headline figure. The full `.ninfer` artifact is larger because it also contains auxiliary/non-main-model data and should not be divided by the headline parameter count to infer quantization quality.

The exact 128K recovery also restores the historical selective Q4 placements:

- 24 GDN `value_z` tensors in Q4
- 7 attention `gate_value` tensors in Q4
- roughly **210.625 MiB** GPU weight-memory saving versus the heavier comparison artifact

## What made 128K fit

The final solution combined three pieces:

1. **Recovered large-T Q4 attention path**
   - `T <= 256`: known-safe legacy/native route
   - `T >= 257`: independent Q4 RowSplit tensor-core MMA
   - retained output-stride correctness fix

2. **Correct mixed Q3/Q4/Q5 weight profile**
   - ~3.95 BPW effective main-model quantization
   - 24 GDN `value_z` tensors in Q4
   - 7 attention `gate_value` tensors in Q4
   - roughly **210.625 MiB** recovered GPU weight memory in the 128K profile transition

3. **128K runtime-memory recovery**
   - combined persistent + workspace device backing
   - full prefill hidden tensor moved to workspace
   - only the final hidden column kept persistently
   - first-use prewarms for critical Q4 paths

## Why this result is interesting

The target was not merely to squeeze the model into memory. The goal was to keep a useful ~3.95-BPW mixed quantization profile while simultaneously retaining:

- full binary 128K capacity,
- high long-prompt prefill throughput,
- useful decode speed,
- deterministic correctness,
- reproducible source/model/binary hashes.

The final 118K benchmark was essentially identical to the earlier historical optimized result:

| Result | Prefill | Decode |
|---|---:|---:|
| Earlier baseline | 1235.03 tok/s | — |
| Historical optimized | 1371.10 tok/s | — |
| Historical B133 | 1377.66 tok/s | 70.24 tok/s |
| **Final validated** | **1377.81 tok/s** | **71.51 tok/s** |

Final prefill improvement over the older baseline: **+11.56%**.

## Documentation

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — end-to-end reproduction path
- [`docs/TECHNICAL_DEEP_DIVE.md`](docs/TECHNICAL_DEEP_DIVE.md) — architecture and optimization details
- [`docs/FAILURES_AND_LESSONS.md`](docs/FAILURES_AND_LESSONS.md) — dead ends and why they failed
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — validation results and comparisons
- [`docs/MEMORY_PROFILE.md`](docs/MEMORY_PROFILE.md) — how the 16 GB fit was recovered
- [`docs/VALIDATED_MANIFEST.md`](docs/VALIDATED_MANIFEST.md) — exact immutable hashes and settings
- [`docs/HISTORY.md`](docs/HISTORY.md) — chronological engineering journey
- [`docs/YOUTUBE_SCRIPT.md`](docs/YOUTUBE_SCRIPT.md) — video script for sharing the project

## Reproduction philosophy

For a credible reproduction, publish more than a tok/s screenshot. Record:

- GPU model and VRAM
- OS / driver / CUDA / compiler
- source commit
- model SHA256
- binary SHA256
- effective main-model BPW and how it was calculated
- prompt SHA256/token count
- max context and KV capacity
- KV dtype
- MTP settings
- prefill chunk
- cold/warm run status

A result should not be described as “true 128K” unless both max context and KV capacity are actually **131072**.

## Upstream attribution

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches, or converted artifacts. Verify all relevant licenses before publishing model files or binary releases.
