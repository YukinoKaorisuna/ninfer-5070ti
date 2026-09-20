# NInfer RTX 5080 — Qwen3.8-27B at true 128K + Vision on 16 GB

This repository documents a validated NInfer configuration for **Qwen3.8-27B** on one **RTX 5080 16 GB** with a genuine **131,072-token context/KV capacity**, Q4 KV, MTP-3 and Vision enabled.

The Vision-enabled serving profile introduced on the `7c10db07` lineage is the recommended/default path documented here. The original text-only 128K release remains preserved as the historical baseline.

## Recommended serving command

The recommended profile is now empirically validated at `--vision-max-tokens 1792`:

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
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool \
  --vision \
  --vision-max-tokens 1792
```

Measured 1792 startup envelope on a clean RTX 5080:

```text
vision_encode       115.7751 MiB
free after startup   26.56 MiB
planned slack        28.88 MiB
```

A deterministic synthetic image test also passed at this profile: a 512×256 red/blue image was correctly identified as red on the left and blue on the right. The request completed with `prompt=211`, `prefill=685.8 tok/s`, `decode=118.3 tok/s`, `ttft=719 ms`, and MTP `3.10 tok/round (70.0%)`.

A deterministic synthetic video test also passed at the same full-128K profile. A 6-second red → green → blue MP4 was correctly returned as `red, green, blue` with thinking disabled. The request completed with `prompt=572`, `gen=6`, `prefill=1721.9 tok/s`, `decode=97.1 tok/s`, `ttft=1111 ms`, and MTP `4.00 tok/round (100.0%)`.

The maximum validated Vision profile remains `--vision-max-tokens 2048`. At 2048 the server measured:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after startup   8.56 MiB
planned slack       10.08 MiB
```

Use a clean GPU for both true-128K Vision profiles; 2048 is especially tight. See [`docs/VISION_128K.md`](docs/VISION_128K.md) for details.

## Vision-source validation — `7c10db07`

The Vision source represented by commit `7c10db07ac8c5803f921b83603b707750652873e` was regression-tested with the exact historical 118,001-token corpus:

| Metric | Original text release | Vision source |
|---|---:|---:|
| Prompt tokens | 118,001 | 118,001 |
| Prefill | 1377.81 tok/s | 1375.16 tok/s |
| Decode | 71.51 tok/s | 71.52 tok/s |
| MTP acceptance | 44.74% | 44.74% |
| Acceptance length | 2.31 | 2.31 |
| KV capacity | 131072 | 131072 |

No meaningful text-performance regression was observed.

## Feature-complete mainline qualification — `b44b1958`

After the `9e163eee` semantic port and the v1.3 rolling-tool reconciliation were both present, runtime tree `b44b1958c301ec6bf4d18973a97d7b42fa6733aa` was qualified again with the exact historical 118,001-token workload:

| Metric | `b44b1958` |
|---|---:|
| Prompt tokens | 118001 |
| Max context | 131072 |
| KV capacity | 131072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| Prefill | **1378.85 tok/s** |
| Decode | **71.44 tok/s** |
| MTP acceptance | **44.74%** |
| MTP acceptance length | **2.31 tok/round** |

Against the strongest v1.2 reference of 1380.61 tok/s prefill and 71.57 tok/s decode, this run was -0.127% and -0.182% respectively, within normal run-to-run variation.

GitHub `main` later advanced to `c8439fbcb89a4daf74cf2692a9425930998c763f` through PR #4, which changed only `docs/BENCHMARKS.md`. Therefore `b44b1958` remains the exact runtime tree qualified by the benchmark above, while `c8439fb` is a documentation-only descendant.

## Vision changes

The validated Vision path adds four changes on top of the original true-128K result:

1. Vision workspace/token budgeting is decoupled from text context.
2. GDN prefill convolution temporaries are released earlier, recovering about 35 MiB of peak workspace.
3. Vision weights are HostMapped so true 128K text KV and Vision coexist on 16 GB.
4. Cached historical media is not charged repeatedly against the fresh-media preprocessing budget.

Image understanding is empirically validated, including multi-image OpenWebUI history. Cached old images remain in the model prompt but no longer consume the fresh preprocessing cap again.

Video input is also empirically validated on the final 128K HostMapped Vision path with a deterministic synthetic MP4 sequence. The model correctly identified the chronological red → green → blue sequence while retaining the full `131072 / 131072` context/KV allocation.

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

## Upstream synchronization

This fork selectively incorporates upstream NInfer changes rather than tracking `Neroued/ninfer:master` commit-for-commit.

The upstream review performed against fork head `f6088f856627045f280e5be8a76fba068b6979e4` assessed upstream history through `9e163eee4b8acec21ab0ac765107b6a3f287b217` on 2026-09-20. That checkpoint is a historical review record, not a statement that `f6088f85` remains the latest validated fork tree.

The `9e163eee` mechanism was subsequently integrated as RTX 5080-specific semantic port `4b62aca386a0a214049201ebeb2a422b0cb609ce` and merged by PR #3 as `33546d7d5be6d82eaac5e4a87a3f7e578f8a1a13`. The v1.3 rolling-tool feature was then reconciled into that lineage by PR #5, producing runtime-qualified tree `b44b1958c301ec6bf4d18973a97d7b42fa6733aa`.

A GitHub "behind" count does not mean all reported commits still need merging: several upstream fixes are already present here as semantic backports or RTX 5080-specific retunes with different commit SHAs.

See [`docs/UPSTREAM_SYNC_STATUS.md`](docs/UPSTREAM_SYNC_STATUS.md) for the commit-scoped review checkpoint, integration ledger, remaining candidate queue and restart procedure for future upstream review.

A result should not be described as “true 128K” unless both max context and KV capacity are actually **131072**.

## Upstream attribution

This project builds on NInfer and the Qwen3.8-27B / DFlash2 ecosystem. Preserve upstream copyright and license notices when redistributing source, binaries, patches or converted artifacts. Verify all relevant licenses before publishing model files or binary releases.

---

## Qwen3.8-27B RTX 5080 v1.2 final validation

Validated code head: `dd2cb0341c321f8a808a6ed75f0a53225983f718`

Final exact 118,001-token acceptance: **1380.61 tok/s prefill**, **71.57 tok/s decode**, **44.74% MTP acceptance**, **2.31 tok/round**, with full **131,072 context / 131,072 Q4 KV**.

Vision 2048 acceptance also passed deterministic image, video, cached-history (`1/1/0` and `2/1/0`) and strict no-OOM validation. Startup remained intentionally tight at **8.56 MiB free / 10.08 MiB planned slack**.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md`.

---

## Qwen3.8-27B RTX 5080 v1.3 final validation

Validated production runtime code head: `ceb32f7d002edab224a83a2e2609f45fca4f8919`

v1.3 preserves the validated **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** RTX 5080 profile while adding corrected Q4 strided-output handling, server default thinking-budget support, and configurable rolling checkpoints for long agent/tool loops.

For append-only tool-loop clients such as OpenClaw, the validated production profile uses:

```text
--prefix-checkpoint-policy rolling-tool
```

`stable-turn` remains the general default. `rolling-tool` advances the private turn checkpoint through completed tool history so rewritten continuations do not repeatedly fall back to the first assistant boundary.

Production OpenClaw validation passed with the restore checkpoint advancing:

```text
19023 -> 21146 -> 24664 -> 26641
```

with **3 advances, 0 plateaus and 0 regressions**. All five continuation requests required at most 3,520 new prefill tokens.

Production `ninfer-serve` SHA256:

```text
3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
```

The exact v1.2 118,001-token long-context and deterministic Vision/OOM validation remains preserved in the v1.2 release record.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`.
