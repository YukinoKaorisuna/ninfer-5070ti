# NInfer RTX 5080 — Qwen3.8-27B at true 128K on 16 GB

This repository preserves and documents a validated NInfer configuration for running **Qwen3.8-27B** on a single **NVIDIA GeForce RTX 5080 16 GB** with a genuine **131,072-token context/KV capacity**.

## Headline result

Final validated workload:

- GPU: **RTX 5080 16 GB**
- Model: **Qwen3.8-27B**
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

## What made 128K fit

The final solution combined three pieces:

1. **Recovered large-T Q4 attention path**
   - `T <= 256`: known-safe legacy/native route
   - `T >= 257`: independent Q4 RowSplit tensor-core MMA
   - retained output-stride correctness fix

2. **Selective mixed-Q4 weight profile**
   - 24 GDN `value_z` tensors in Q4
   - 7 attention `gate_value` tensors in Q4
   - roughly **210.625 MiB** GPU weight-memory saving versus the all-Q5 variant

3. **128K runtime-memory recovery**
   - combined persistent + workspace device backing
   - full prefill hidden tensor moved to workspace
   - only the final hidden column kept persistently
   - first-use prewarms for critical Q4 paths

## Why this result is interesting

The target was not merely to squeeze the model into memory. The goal was to keep a relatively high-quality mixed quantization profile while simultaneously retaining:

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
- prompt SHA256/token count
- max context and KV capacity
- KV dtype
- MTP settings
- prefill chunk
- cold/warm run status

A result should not be described as “true 128K” unless both max context and KV capacity are actually **131072**.

## Upstream attribution

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches, or converted artifacts. Verify all relevant licenses before publishing model files or binary releases.
