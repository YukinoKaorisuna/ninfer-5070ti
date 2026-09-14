# Validated Release Manifest

This file records the exact state validated by Batches 210, 211 and 212.

## Source

```text
repository: toddballinger/ninfer-5080
validated commit: 473dade56031852a7d96edef049d859da96a6df9
release tag: qwen3.8-27b-rtx5080-128k-v1
```

## Artifact

```text
path used during validation:
/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer

bytes:
16461267456

SHA256:
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Binary

```text
path used during validation:
/models/ninfer-builds/mtp3-128k-q4-memory-fixed/apps/ninfer

SHA256:
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
Q4_VALUE_Z_COUNT=24
Q4_GATE_VALUE_COUNT=7
remaining relevant main-model weights predominantly Q5 groupwise
```

## Runtime

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
```

## Final benchmark

```text
PROMPT_TOKENS=118001
PREFILL_TOK_S=1377.81
DECODE_TOK_S=71.51
MTP_ACCEPTANCE_RATE=44.74%
MTP_ACCEPTANCE_LENGTH=2.31
```

## Memory

```text
GPU_WEIGHTS_APPROX=12.64_GiB
FREE_AFTER_WEIGHTS=2.56_GiB
FREE_AFTER_STARTUP=10.56_MiB
RUNTIME_RESERVATION=2.55_GiB
KV_CACHE_PAYLOAD=2.26_GiB
PLANNED_SLACK=11.39_MiB
```

## Acceptance checks

```text
RUNTIME_RESERVATION_ERROR=NO
MATRIX_WINDOW_ERROR=NO
CONTRACT_ERROR=NO
OOM_ERROR=NO
NONFINITE_WARNING=NO
```

## Validation sequence

```text
Batch 210: TRUE_131072_Q4_MTP3_CAPACITY_AND_SHORT_ORACLE_PASS
Batch 211: MIXED_Q4_LONG_ORACLE_TRUE_131072_PASS
Batch 212: FINAL_TRUE_128K_MTP3_Q4_RECOVERY_PASS
```

## Freeze/publish sequence

```text
Batch 213: FINAL_VALIDATED_SOURCE_FROZEN_AND_TAGGED
Batch 214B: RTX5080_VALIDATED_SOURCE_AND_TAG_PUBLISHED
```

The annotated release tag must remain on the validated commit above. Documentation commits intentionally come later.
