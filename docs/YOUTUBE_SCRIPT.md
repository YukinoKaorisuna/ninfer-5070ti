# YouTube Production Script

## Suggested title

**Qwen3.8-27B at True 128K on a 16GB RTX 5080**

## Opening

I got Qwen3.8-27B running on a single RTX 5080 16 GB with both maximum context and KV capacity set to 131,072.

The final validation used an actual 118,001-token prompt and reached 1,377.81 tokens per second in prefill and 71.51 tokens per second in decode.

The text-model core works out to approximately **3.95 effective bits per weight** on a GGUF-comparable weighted-storage basis, and the runtime finished with only about 11.39 MiB of planned GPU-memory slack.

## The problem

The target was not simply to make the model load. The goal was to keep a useful mixed Q3/Q4/Q5 quantization profile, allocate a genuine 128K KV cache, preserve fast long-prompt processing, and retain useful decode performance.

A heavier comparison artifact could run the model, but it was roughly 166 MiB short when reserving the large-context runtime memory.

## Large-token-width optimization

The recovered attention-input path uses the known-safe route for token widths up to 256 and an independent Q4 RowSplit tensor-core MMA route for widths of 257 and above.

A 3,201-token validation prompt created 896, 896, 896 and 509-token prefill chunks. The final artifact reached 2,236.99 tok/s, almost exactly matching the historical 2,238.05 tok/s result.

## The quantization profile

The final text core is not predominantly Q5. It is a mixed groupwise profile:

- about **42.42% Q3G64**
- about **45.92% Q4G64**
- about **11.57% Q5G64**
- a tiny remainder in BF16 / FP32 norms and miscellaneous tensors

That gives approximately **3.953 BPW** across the main text-model storage, or **3.941 BPW** when counting only the quantized matrix weights. For ordinary GGUF-style comparisons, I use **~3.95 BPW**.

## The memory breakthrough

Git history showed that the earlier exact-128K profile specifically placed 24 GDN `value_z` tensors and 7 attention `gate_value` tensors in Q4.

Those placements recovered about **210.625 MiB** of GPU weight memory in the 128K recovery comparison and supplied the headroom needed for the full KV allocation.

They are part of the broader Q3/Q4/Q5 profile rather than the complete description of the model's quantization.

## Runtime-memory recovery

The working runtime also restored a combined persistent/workspace device allocation, moved the full prefill hidden tensor into workspace, kept only the final hidden column persistent, and prewarmed critical Q4 paths before VRAM was fully committed.

## Final configuration

- RTX 5080 16 GB
- Qwen3.8-27B
- **~3.95 effective main-model BPW**
- mixed Q3/Q4/Q5 groupwise weights
- 24 Q4 `value_z` tensors in the exact-128K profile
- 7 Q4 `gate_value` tensors in the exact-128K profile
- Q4 KV cache
- MTP-3
- prefill chunk 896
- max context 131,072
- KV capacity 131,072
- approximately 12.64 GiB GPU weights
- 2.26 GiB KV payload
- 11.39 MiB planned slack

## Final benchmark

With 118,001 prompt tokens and the full 131,072-token KV capacity allocated:

- Prefill: **1,377.81 tok/s**
- Decode: **71.51 tok/s**
- MTP acceptance: **44.74%**

The earlier baseline was 1,235.03 tok/s prefill, so the final implementation improved prefill by about 11.56%.

The historical optimized B133 result was 1,377.66 tok/s prefill and 70.24 tok/s decode, making the recovered result effectively identical in prefill and slightly faster in decode.

## Reproducibility section

Show the exact Git commit, release tag, model SHA256, binary SHA256, CUDA/compiler versions, runtime flags, actual prompt length, effective BPW calculation, and the final memory planner values on screen.

Explain that the `.ninfer` container's total file size is not itself a GGUF-comparable BPW measurement because the container includes auxiliary/non-main-model data. The comparable figure is the parameter-weighted main-model storage: approximately **3.95 BPW**.

Explain that this project defines true 128K as both maximum context and KV capacity being 131,072. The earlier 129,024 configuration was useful for 896-token chunk alignment but is not the final acceptance target.

## Closing

The repository contains the exact validated source tag, reproduction guide, benchmark history, memory profile, failure analysis and issue templates for independent results.

Viewers with Blackwell GPUs can reproduce the configuration and submit their hardware and benchmark details through the GitHub issue templates.
