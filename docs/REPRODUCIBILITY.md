# Reproducibility Guide

This guide describes the validated RTX 5080 true-128K path with Vision enabled.

## Validated hardware/software

- NVIDIA GeForce RTX 5080 16 GB
- Linux
- GCC 15.2.0
- CUDA 13.3.73
- validated Vision source commit: `7c10db07ac8c5803f921b83603b707750652873e`

The original text-only release remains frozen at `473dade56031852a7d96edef049d859da96a6df9` / tag `qwen3.8-27b-rtx5080-128k-v1`.

## Pin model revisions

```text
Qwen/Qwen3.8-27B
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

z-lab/Qwen3.8-27B-DFlash2
50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

## Build

Use a Release build and record compiler/CUDA/driver details if benchmark parity matters:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

ninja -C build -j16 ninfer ninfer-serve
```

## Model artifact

Validated artifact:

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

The text core is a mixed Q3/Q4/Q5 groupwise profile at approximately 3.953 effective BPW. The full `.ninfer` container size is not itself a GGUF-comparable BPW numerator.

## Recommended true-128K Vision server

Use `1792` Vision tokens as the safer default:

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

The maximum validated Vision setting is `--vision-max-tokens 2048`.

At 2048 the measured startup envelope was:

```text
vision_encode workspace  132.3142 MiB
free after weights         2.56 GiB
free after startup          8.56 MiB
planned slack              10.08 MiB
```

## Clean-GPU prerequisite

A representative successful launch started from:

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

The 2048 profile is extremely tight; even a small competing GPU allocation can make startup fail.

## True 128K definition

For this project, true 128K means both values are exactly 131072:

```text
--max-context 131072
--kv-capacity 131072
```

Do not describe a reduced KV allocation as the same result.

## Long-context regression acceptance

The Vision source was re-tested with the exact historical corpus:

```text
prompt SHA256: 078d726e07b6c610d3136751fb2bdfbf4965ebdd9d8afc1a07dedb9ac03fe0fd
prompt tokens: 118001
max context:   131072
KV capacity:   131072
prefill chunk: 896
KV dtype:      q4
speculation:   MTP-3
max-new:       32
```

Observed result:

```text
return code:           0
prefill:               1375.16 tok/s
decode:                  71.52 tok/s
MTP acceptance:          44.74%
MTP acceptance length:    2.31 tok/round
workspace peak:          116.00 MiB
free after startup:       44.56 MiB
planned slack:            46.39 MiB
```

Historical text release on the same acceptance workload:

```text
prefill:               1377.81 tok/s
decode:                  71.51 tok/s
MTP acceptance:          44.74%
MTP acceptance length:    2.31 tok/round
```

No meaningful text-path regression was observed.

## Vision acceptance

Validated on the final HostMapped Vision path:

- image input through the OpenAI-compatible server;
- ordinary photos and small-text/receipt input;
- OpenWebUI multi-image history;
- cached historical media plus a newly uploaded image;
- full 131072 text context/KV retained.

Observed cache patterns included `media_cache=1/1/0` and `media_cache=2/1/0`, proving old cached images were no longer charged repeatedly against the fresh preprocessing cap.

Video input is supported by the frontend but has not yet been empirically validated on this final 128K HostMapped configuration.

## Validation hashes

```text
ninfer SHA256:
5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140

ninfer-serve SHA256:
61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```

Rebuilt binaries can differ byte-for-byte if compiler/toolkit inputs change, so always record the full build environment alongside hashes.
