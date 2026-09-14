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

The validated text core is a mixed **Q3/Q4/Q5 groupwise** profile.

Approximate parameter-weighted distribution:

```text
Q3G64_F16S   42.42%   3.25 bpw encoded
Q4G64_F16S   45.92%   4.25 bpw encoded
Q5G64_F16S   11.57%   5.25 bpw encoded
BF16 / FP32  ~0.10%   small norms / miscellaneous tensors
```

Main-model accounting:

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953

quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

For GGUF-style public comparisons, report this as **~3.95 BPW effective main-model quantization**.

The final 128K profile also uses these specific Q4 placements:

```text
24 x GDN value_z          -> Q4 groupwise
7  x attention gate_value -> Q4 groupwise
```

Those placements recovered approximately **210.625 MiB** of GPU weight memory relative to the heavier comparison artifact used during the 128K recovery work.

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

Use the converter options from the validated source tree for the exact mixed Q3/Q4/Q5 profile.

Final artifact:

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The full artifact size is **not** the GGUF-comparable BPW numerator because the container includes auxiliary/non-main-model content in addition to the text-core weights.

## Clean-GPU prerequisite for true 128K

The full `131072 / 131072` configuration has only about **11 MiB of planned slack**. Reproduction should therefore begin with the RTX 5080 effectively idle.

A validated clean-card server start reported:

```text
NVIDIA-SMI before launch:
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB

NInfer after weights and prewarm:
free before runtime reservation = 2624.56 MiB
runtime reservation             = 2613.17 MiB
planned slack                   = 11.39 MiB
free after startup              = 10.56 MiB
```

Check the card before launching:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

If another process is holding GPU memory, stop it before attempting the true-128K profile. A nominal 16 GB card is not sufficient by itself; the validated configuration assumes essentially the full usable framebuffer is available to NInfer at startup.

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

The same settings were also successfully started under `ninfer-serve`, reaching the listening state with `131072 / 131072`, Q4 KV, MTP-3 and prefill chunk 896 unchanged.

A configuration such as 129024 is useful for 896-token chunk alignment and historical performance comparison, but it is not the final binary 128K acceptance target.

## Validation sequence

Run progressively rather than jumping directly to an 118K prompt:

1. Confirm the GPU is effectively idle.
2. Short deterministic JSON oracle.
3. 3,201-token long oracle.
4. Confirm the long path hits `T=896` and the final partial chunk.
5. Confirm `131072 / 131072` startup succeeds.
6. Run the 118,001-token final workload.

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
