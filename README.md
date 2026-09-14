# NInfer RTX 5080 — Qwen3.8-27B at true 128K + Vision on 16 GB

This repository documents a validated NInfer configuration for **Qwen3.8-27B** on one **RTX 5080 16 GB** with a genuine **131,072-token context/KV capacity**, Q4 KV, MTP-3 and Vision enabled.

Vision is now the recommended/default path. The original text-only 128K release remains preserved as the historical baseline.

## Recommended serving command

Use `--vision-max-tokens 1792` as the safer default:

```bash
./build/apps/ninfer-serve /path/to/model.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id qwen3.8-27b \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 896 \
  --kv-dtype q4 \
  --spec mtp \
  --draft-tokens 3 \
  --no-cuda-graph \
  --max-concurrency 1 \
  --vision \
  --vision-max-tokens 1792
```

The maximum validated Vision profile is `--vision-max-tokens 2048`. At 2048 the server measured:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after startup   8.56 MiB
planned slack       10.08 MiB
```

Use a clean GPU for 2048. See [`docs/VISION_128K.md`](docs/VISION_128K.md) for details.

## Validated result

The Vision source was regression-tested with the exact historical 118,001-token corpus:

| Metric | Original text release | Vision source |
|---|---:|---:|
| Prompt tokens | 118,001 | 118,001 |
| Prefill | 1377.81 tok/s | 1375.16 tok/s |
| Decode | 71.51 tok/s | 71.52 tok/s |
| MTP acceptance | 44.74% | 44.74% |
| Acceptance length | 2.31 | 2.31 |
| KV capacity | 131072 | 131072 |

No meaningful text-performance regression was observed.

## Vision changes

The validated Vision path adds four changes on top of the original true-128K result:

1. Vision workspace/token budgeting is decoupled from text context.
2. GDN prefill convolution temporaries are released earlier, recovering about 35 MiB of peak workspace.
3. Vision weights are HostMapped so true 128K text KV and Vision coexist on 16 GB.
4. Cached historical media is not charged repeatedly against the fresh-media preprocessing budget.

Image understanding is empirically validated, including multi-image OpenWebUI history. Cached old images remain in the model prompt but no longer consume the fresh preprocessing cap again.

Video input is supported by the frontend, but **video has not yet been empirically validated on the final 128K HostMapped Vision path**.

## Source and artifacts

Validated Vision source before merge to `main`:

```text
7c10db07ac8c5803f921b83603b707750652873e
```

It was merged to `main` through PR #1. The original immutable text-only release remains:

```text
commit: 473dade56031852a7d96edef049d859da96a6df9
tag:    qwen3.8-27b-rtx5080-128k-v1
```

Model artifact:

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

Vision validation binaries:

```text
ninfer:       5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140
ninfer-serve: 61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```

## Quantization profile

The text core is mixed Q3/Q4/Q5 groupwise, approximately **3.953 effective BPW**:

| Format | Share |
|---|---:|
| Q3G64_F16S | 42.42% |
| Q4G64_F16S | 45.92% |
| Q5G64_F16S | 11.57% |
| BF16 / FP32 | ~0.10% |

The 128K profile retains 24 GDN `value_z` tensors and 7 attention `gate_value` tensors in Q4, recovering about 210.625 MiB versus the heavier comparison artifact.

## Clean-GPU requirement

A representative successful launch began with:

```text
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB
```

Check before launch:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

## Documentation

- [`docs/VISION_128K.md`](docs/VISION_128K.md) — Vision serving profiles and validation
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — end-to-end reproduction
- [`docs/TECHNICAL_DEEP_DIVE.md`](docs/TECHNICAL_DEEP_DIVE.md) — architecture and optimization details
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — results and comparisons
- [`docs/MEMORY_PROFILE.md`](docs/MEMORY_PROFILE.md) — the 16 GB memory fit
- [`docs/VALIDATED_MANIFEST.md`](docs/VALIDATED_MANIFEST.md) — exact hashes and settings
- [`docs/HISTORY.md`](docs/HISTORY.md) — engineering history

A result should not be described as “true 128K” unless both max context and KV capacity are actually **131072**.

## Upstream attribution

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches or converted artifacts. Verify all relevant licenses before publishing model files or binary releases.
