# Validated Release Manifest

This file records both the immutable original text-only 128K release and the later validated Vision-enabled source that is now recommended on `main`.

## Current recommended Vision source

```text
repository: toddballinger/ninfer-5080
validated Vision commit: 7c10db07ac8c5803f921b83603b707750652873e
Vision branch: qwen3.8-27b-rtx5080-128k-vision
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

## Vision validation status

```text
IMAGE_INPUT=VALIDATED
OPENWEBUI_MULTI_IMAGE_HISTORY=VALIDATED
CACHED_MEDIA_FRESH_BUDGET_FIX=VALIDATED
VIDEO_FRONTEND_SUPPORT=PRESENT
VIDEO_ON_FINAL_128K_HOSTMAPPED_PATH=NOT_YET_EMPIRICALLY_VALIDATED
```

## Acceptance checks

```text
TRUE_131072_CONTEXT=PASS
TRUE_131072_KV=PASS
Q4_KV=PASS
MTP3=PASS
118001_TOKEN_REGRESSION=PASS
VISION_2048_STARTUP=PASS
IMAGE_REQUEST=PASS
MULTI_IMAGE_HISTORY=PASS
OOM_ERROR=NO
NONFINITE_WARNING=NO
```
