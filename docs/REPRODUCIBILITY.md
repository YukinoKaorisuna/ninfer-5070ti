# Reproducibility Guide

This guide describes the validated path for reproducing the RTX 5080 true-128K result.

## Validated hardware/software

Validated system:

- NVIDIA GeForce RTX 5080 16 GB
- Linux
- GCC 15.2.0
- CUDA 13.3.73
- NInfer source commit `473dade56031852a7d96edef049d859da96a6df9`

The result may transfer to other Blackwell GPUs, but memory margins and performance will differ.

## Pin all source revisions

Do not reproduce from floating `main` branches if you want comparable results.

Validated upstream model revisions:

```text
Qwen/Qwen3.8-27B
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

Validated NInfer source:

```text
473dade56031852a7d96edef049d859da96a6df9
```

## Build

Use a Release build. The validated environment used Ninja and ccache:

```bash
export CCACHE_BASEDIR="$HOME"
export CCACHE_NOHASHDIR=true

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

ninja -C build -j16 ninfer
```

If benchmark parity matters, record the compiler, CUDA toolkit, driver and final binary SHA256.

## Mixed quantization profile

The validated 128K artifact uses selective Q4 tensors rather than reducing the entire model aggressively:

```text
24 x GDN value_z       -> Q4 groupwise
7  x attention gate_value -> Q4 groupwise
remaining relevant weights -> predominantly Q5 groupwise
```

This profile saves approximately **210.625 MiB** of GPU weight memory relative to the all-Q5 variant.

## Artifact conversion

GPU-side conversion hit a PyTorch scratch-allocation OOM on the 16 GB card, so the final artifact was converted on CPU.

The validated source directories were pinned to the exact upstream revisions above.

Conceptually:

```bash
python -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out /path/to/model.ninfer \
  --device cpu
```

Use the converter options from the validated source tree for the exact mixed-Q4 profile.

Final artifact:

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## True 128K acceptance configuration

For this project, “true 128K” means both the maximum context and allocated KV capacity are exactly 131072:

```text
--max-context 131072
--kv-capacity 131072
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--no-cuda-graph
```

A configuration such as 129024 is useful for 896-token chunk alignment and historical performance comparison, but it is not the final binary 128K acceptance target.

## Validation sequence

Run progressively rather than jumping directly to an 118K prompt:

1. Short deterministic JSON oracle.
2. 3,201-token long oracle.
3. Confirm the long path hits `T=896` and the final partial chunk.
4. Confirm `131072 / 131072` startup succeeds.
5. Run the 118,001-token final workload.

## Deterministic short oracle

Prompt:

```text
You are auditing deterministic model arithmetic.
Return a JSON object with exactly these keys:
{"answer":"","number":0,"valid":false}
Set answer to the lowercase word produced by joining alpha and beta with a hyphen.
Set number to the result of 137 + 286.
Set valid to true if 17 multiplied by 19 equals 323.
Do not include markdown or any additional text.
```

Expected output:

```json
{"answer":"alpha-beta","number":423,"valid":true}
```

Expected output SHA256:

```text
4c509613de14990e22aeb295ef7c8aff44cb704ffadf836d5cc25518ecce36a9
```

## Final acceptance conditions

The final validated run showed:

```text
MODEL_RC=0
PROMPT_TOKENS=118001
PREFILL_TOK_S=1377.81
DECODE_TOK_S=71.51

RUNTIME_RESERVATION_ERROR=NO
MATRIX_WINDOW_ERROR=NO
CONTRACT_ERROR=NO
OOM_ERROR=NO
NONFINITE_WARNING=NO
```

Memory:

```text
GPU_WEIGHTS≈12.64 GiB
FREE_AFTER_WEIGHTS=2.56 GiB
FREE_AFTER_STARTUP=10.56 MiB
RUNTIME_RESERVATION=2.55 GiB
KV_CACHE_PAYLOAD=2.26 GiB
PLANNED_SLACK=11.39 MiB
```

Small run-to-run performance variation is expected; hashes and configuration are the first reproducibility checks.
