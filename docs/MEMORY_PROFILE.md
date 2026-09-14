# Memory Profile

The true-128K result is a near-capacity fit on a 16 GB RTX 5080. Vision is now part of the recommended configuration, so both text and Vision memory lifetimes matter.

## Text-model quantization

The text core is mixed Q3/Q4/Q5 groupwise:

| Format | Share | Encoded storage |
|---|---:|---:|
| Q3G64_F16S | 42.42% | 3.25 bpw |
| Q4G64_F16S | 45.92% | 4.25 bpw |
| Q5G64_F16S | 11.57% | 5.25 bpw |
| BF16 / FP32 | ~0.10% | small norms / misc. |

```text
logical parameters:          26,895,998,464
encoded main-model bytes:    13,289,938,944
effective main-model BPW:    3.953
quantized matrix parameters: 26,869,760,000
quantized matrix bytes:      13,237,452,800
weighted matrix BPW:         3.941
```

The exact-128K profile retains 24 GDN `value_z` tensors and 7 attention `gate_value` tensors in Q4, recovering about **210.625 MiB** versus the heavier comparison artifact.

## Artifact

```text
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

## Text-only runtime on current Vision source

The current Vision source was regression-tested with the historical 118001-token workload and no Vision input:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
PREFILL_CHUNK=896
KV_DTYPE=q4
SPECULATION=mtp
DRAFT_TOKENS=3
```

Measured values:

```text
GPU weights             ~12.64 GiB
KV cache payload          2.26 GiB
workspace peak          116.00 MiB
free after weights        2.56 GiB
free after startup       44.56 MiB
planned slack            46.39 MiB
```

The earlier text-only release had only about 11 MiB of planned slack. The GDN temporary lifetime reduction in the Vision source recovered roughly 35 MiB of peak workspace while preserving text performance.

## Why Vision originally did not fit

The historical Vision planner tied Vision workspace capacity to the large text-context capacity. That produced an impractically large Vision workspace at true 128K.

After decoupling the Vision token budget from text context, `--vision-max-tokens 1024` reduced Vision workspace to about 66.16 MiB. At that point the remaining shortfall was dominated by resident Vision weights, not Vision scratch.

Keeping the Vision weights GPU-resident still left the full-128K configuration short by roughly 240 MiB.

## HostMapped Vision weights

The validated solution keeps the Vision weights HostMapped instead of permanently resident in device memory. This recovers enough framebuffer to keep:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
KV_DTYPE=q4
PREFILL_CHUNK=896
MTP_DRAFT_TOKENS=3
VISION=enabled
```

Image tests confirm the HostMapped path is usable. It can incur PCIe traffic versus fully GPU-resident Vision weights, but it makes true 128K + Vision possible on the 16 GB card.

## Vision token profiles

Measured Vision workspace:

```text
vision-max-tokens 1024 -> vision_encode  66.1580 MiB
vision-max-tokens 2048 -> vision_encode 132.3142 MiB
```

At 1024, Vision workspace remains below the 116 MiB text-prefill peak. At 2048, Vision becomes the workspace peak.

### Recommended profile

Use `--vision-max-tokens 1792` as the safer default. It offers more visual detail than 1024 while retaining more startup margin than the maximum profile.

### Maximum validated profile

At `--vision-max-tokens 2048`:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after weights   2.56 GiB
free after startup    8.56 MiB
planned slack        10.08 MiB
```

This is a valid but extremely tight fit. A clean GPU is required.

## Cached historical media

OpenWebUI resends prior media as chat history. The frontend now distinguishes fresh preprocessing work from cached historical media: cache hits remain part of the prompt but are not charged repeatedly against the fresh media preprocessing cap.

This prevents a long multimodal chat from failing simply because previously processed images are resent. Historical Vision tokens still count toward the context window normally.

## Clean-start requirement

A representative successful launch began from:

```text
memory.total = 16303 MiB
memory.used  = 1 MiB
memory.free  = 15841 MiB
```

Recommended preflight:

```bash
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

At this utilization level, tens of MiB are material: they determine whether full `131072 / 131072` plus Vision starts or fails during runtime reservation.
