# Validated Release Manifest

This file records both the immutable original text-only 128K release and the later validated Vision-enabled source that is now recommended on `main`.

## Current recommended Vision source

```text
repository: toddballinger/ninfer-5080
validated Vision commit: 7c10db07ac8c5803f921b83603b707750652873e
Vision branch: qwen3.8-27b-rtx5080-128k-vision
Vision release tag: qwen3.8-27b-rtx5080-128k-vision-v1
merged to main via PR #1
```

The Vision source adds independent Vision budgeting, earlier GDN temporary release, HostMapped Vision weights and cache-aware historical-media accounting.

## Original immutable text-only release

```text
validated commit: 473dade56031852a7d96edef049d859da96a6df9
release tag: qwen3.8-27b-rtx5080-128k-v1
```

The original tag remains unchanged as the historical baseline.

## Model artifact

```text
path used during validation:
/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer

bytes: 16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Vision-source binaries

```text
ninfer:
5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140

ninfer-serve:
61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```

Historical text-release CLI binary:

```text
b38e987a16f7cdda3c5eee81b0ac821f9f2aad86f117b745401a9ef3b5c43012
```

## Model sources

```text
Qwen/Qwen3.8-27B
revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
revision: 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

## Quantization profile

```text
Q3G64_F16S_SHARE=42.42%
Q4G64_F16S_SHARE=45.92%
Q5G64_F16S_SHARE=11.57%
BF16_FP32_SHARE_APPROX=0.10%
Q4_VALUE_Z_COUNT=24
Q4_GATE_VALUE_COUNT=7
MAIN_TEXT_LOGICAL_PARAMS=26895998464
MAIN_TEXT_ENCODED_BYTES=13289938944
MAIN_TEXT_EFFECTIVE_BPW=3.953
QUANTIZED_MATRIX_PARAMS=26869760000
QUANTIZED_MATRIX_BYTES=13237452800
QUANTIZED_MATRIX_WEIGHTED_BPW=3.941
```

Public shorthand: **~3.95 BPW effective main-model quantization**.

## Common true-128K runtime

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
CUDA_GRAPHS=off
MAX_CONCURRENCY=1
```

## Recommended Vision profile

```text
VISION=on
VISION_MAX_TOKENS=1792
VISION_ENCODE_WORKSPACE=115.7751_MiB
FREE_AFTER_STARTUP=26.56_MiB
PLANNED_SLACK=28.88_MiB
```

Maximum validated Vision profile:

```text
VISION_MAX_TOKENS=2048
VISION_ENCODE_WORKSPACE=132.3142_MiB
FREE_AFTER_STARTUP=8.56_MiB
PLANNED_SLACK=10.08_MiB
```

## Vision-source long-context regression

Exact historical 118001-token corpus:

```text
PROMPT_SHA256=078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
PROMPT_TOKENS=118001
PREFILL_TOK_S=1375.16
DECODE_TOK_S=71.52
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
RETURN_CODE=0
```

Original text release on the same acceptance workload:

```text
PREFILL_TOK_S=1377.81
DECODE_TOK_S=71.51
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
```

No meaningful text-path regression was observed.

## Deterministic image validation at 1792

```text
IMAGE_INPUT=VALIDATED
IMAGE_TEST=512x256_RED_LEFT_BLUE_RIGHT
IMAGE_RESULT=LEFT_RED_RIGHT_BLUE
PROMPT_TOKENS=211
PREFILL_TOK_S=682.8
DECODE_TOK_S=118.1
TTFT_MS=725
MTP_TOK_PER_ROUND=3.10
MTP_ACCEPTANCE=70.0%
```

A previous run of the same deterministic image test measured 685.8 tok/s prefill and 719 ms TTFT; the repeated result confirms the functional path.

## Deterministic video validation at 1792

A six-second MP4 containing red, then green, then blue scenes was generated locally with ffmpeg and sent through the OpenAI-compatible server with thinking disabled.

```text
VIDEO_ON_FINAL_128K_HOSTMAPPED_PATH=VALIDATED
VIDEO_EXPECTED=red,green,blue
VIDEO_RESULT=red,green,blue
FINISH_REASON=stop_token
PROMPT_TOKENS=572
GENERATED_TOKENS=6
PREFILL_TOK_S=1721.9
DECODE_TOK_S=97.1
TTFT_MS=1111
WALL_S=1.16
MTP_TOK_PER_ROUND=4.00
MTP_ACCEPTANCE=100.0%
```

This is an end-to-end functional validation of video acquisition, preprocessing, Vision encode and generation. It is not a broad video-quality benchmark.

## Multi-image history validation

```text
OPENWEBUI_MULTI_IMAGE_HISTORY=VALIDATED
CACHED_MEDIA_FRESH_BUDGET_FIX=VALIDATED
OBSERVED_MEDIA_CACHE_PATTERNS=1/1/0,2/1/0
```

Cached historical media remains in the model prompt but is not charged repeatedly against the fresh preprocessing budget.

## Acceptance checks

```text
TRUE_131072_CONTEXT=PASS
TRUE_131072_KV=PASS
Q4_KV=PASS
MTP3=PASS
118001_TOKEN_REGRESSION=PASS
VISION_1792_STARTUP=PASS
VISION_2048_STARTUP=PASS
IMAGE_REQUEST=PASS
VIDEO_REQUEST=PASS
VIDEO_SEMANTIC_RESULT=PASS
MULTI_IMAGE_HISTORY=PASS
OOM_ERROR=NO
NONFINITE_WARNING=NO
```

---

## Qwen3.8-27B RTX 5080 v1.2 final validation

Validated code head: `dd2cb0341c321f8a808a6ed75f0a53225983f718`

Final exact 118,001-token acceptance: **1380.61 tok/s prefill**, **71.57 tok/s decode**, **44.74% MTP acceptance**, **2.31 tok/round**, with full **131,072 context / 131,072 Q4 KV**.

Vision 2048 acceptance also passed deterministic image, video, cached-history (`1/1/0` and `2/1/0`) and strict no-OOM validation. Startup remained intentionally tight at **8.56 MiB free / 10.08 MiB planned slack**.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.2.md`.

---

## Qwen3.8-27B RTX 5080 v1.3 final validation

Validated production runtime code head: `ceb32f7d002edab224a83a2e2609f45fca4f8919`

```text
NINFER_SERVE_SHA256=3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56
PREFIX_CHECKPOINT_POLICY=rolling-tool
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
VISION=on
VISION_MAX_TOKENS=2048
DEFAULT_THINKING_BUDGET=2048
```

v1.3 preserves the validated **131,072 context / 131,072 Q4 KV / MTP-3 / Vision-2048** RTX 5080 profile while adding corrected Q4 strided-output handling, server default thinking-budget support, and rolling tool checkpoints.

Production OpenClaw validation:

```text
RESTORE_CACHE_SEQUENCE=19023,21146,24664,26641
RESTORE_CACHE_ADVANCES=3
RESTORE_CACHE_PLATEAUS=0
RESTORE_CACHE_REGRESSIONS=0
PRODUCTION_ROLLING_CHECKPOINT=PASS
```

All five continuation requests had an uncached prompt suffix below 4,096 tokens.

The exact v1.2 118,001-token long-context and deterministic Vision/OOM validation remains preserved in the v1.2 release record.

Full record: `docs/RELEASE_QWEN3.8_27B_RTX5080_V1.3.md`.
