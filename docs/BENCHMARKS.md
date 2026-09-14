# Benchmarks

## Current recommended result: true 128K + Vision source

The Vision-enabled source was regression-tested with the exact historical 118,001-token workload while keeping the full 131,072-token maximum context and KV capacity allocated.

| Metric | Vision source |
|---|---:|
| Prompt tokens | 118001 |
| Max context | 131072 |
| KV capacity | 131072 |
| Prefill chunk | 896 |
| KV dtype | Q4 |
| Speculation | MTP-3 |
| Prefill | **1375.16 tok/s** |
| Decode | **71.52 tok/s** |
| MTP acceptance rate | **44.74%** |
| MTP acceptance length | **2.31 tok/round** |
| Workspace peak, text-only CLI | 116.00 MiB |
| Free after startup, text-only CLI | 44.56 MiB |
| Planned slack, text-only CLI | 46.39 MiB |

Validated Vision source commit before merge to `main`:

```text
7c10db07ac8c5803f921b83603b707750652873e
```

## Original text-only release comparison

| Result | Prefill | Decode | MTP acceptance | Acceptance length |
|---|---:|---:|---:|---:|
| Original validated text release | 1377.81 tok/s | 71.51 tok/s | 44.74% | 2.31 |
| Vision source, same 118001-token corpus | 1375.16 tok/s | 71.52 tok/s | 44.74% | 2.31 |

The Vision source is about 0.19% lower in prefill and effectively identical in decode/MTP behavior. This is within normal run-to-run variation; no meaningful text-path regression was observed.

## Recommended Vision 1792 serving profile

The recommended Vision profile is empirically validated with the full `131072 / 131072` text context/KV allocation and:

```text
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--vision
--vision-max-tokens 1792
```

Measured startup envelope:

| Item | Value |
|---|---:|
| Text prefill workspace | 116.0127 MiB |
| MTP prefill workspace | 116.0127 MiB |
| Vision encode workspace | 115.7751 MiB |
| Free after startup | 26.56 MiB |
| Planned slack | 28.88 MiB |

A deterministic synthetic image request also passed. The input was a 512×256 image with a red left half and blue right half; the model correctly returned that the left half was red and the right half blue.

Request metrics:

```text
prompt=211
generated=62
prefill=685.8 tok/s
decode=118.3 tok/s
ttft=719 ms
MTP=3.10 tok/round (70.0%)
```

## Video validation

Video is empirically validated on the final 128K HostMapped Vision configuration. The test used a deterministic 6-second MP4 containing three solid-color scenes in chronological order: red, green and blue. With thinking disabled and a constrained answer format, the model returned `red, green, blue`.

The server retained the full `131072 / 131072` context/KV allocation. Request metrics were `prompt=572`, `generated=6`, `prefill=1721.9 tok/s`, `decode=97.1 tok/s`, `ttft=1111 ms`, `wall=1.16 s`, MTP `4.00 tok/round`, and `finish=stop_token`.

This is an end-to-end functional validation of video acquisition, preprocessing, Vision encode and generation, not a broad video-understanding quality benchmark.

## Vision 2048 serving profile

The maximum validated Vision profile kept the full `131072 / 131072` text context/KV allocation and used:

```text
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--vision
--vision-max-tokens 2048
```

Measured startup envelope:

| Item | Value |
|---|---:|
| Text prefill workspace | 116.0127 MiB |
| MTP prefill workspace | 116.0127 MiB |
| Vision encode workspace | 132.3142 MiB |
| Free after weights | 2.56 GiB |
| Free after startup | 8.56 MiB |
| Planned slack | 10.08 MiB |

A representative OpenWebUI image request at the 2048 profile reported:

```text
prompt=9134
prefill=2153.3 tok/s
decode=113.7 tok/s
ttft=8.35 s
MTP=3.01 tok/round (67.1%)
```

This includes the complete OpenWebUI prompt/tool payload and is not a standalone Vision-kernel microbenchmark.

## Multi-image history validation

After the cache-aware media-budget fix, OpenWebUI conversations containing old images plus a newly uploaded image completed instead of failing with `media_budget_exceeded`.

Observed request patterns included:

```text
media_cache=1/1/0
media_cache=2/1/0
```

Cached historical images remain present in the model prompt but do not consume the fresh preprocessing budget again.

## Historical comparison

| Result | Prefill | Decode |
|---|---:|---:|
| Older baseline | 1235.03 tok/s | — |
| Historical optimized | 1371.10 tok/s | — |
| Historical B133 | 1377.66 tok/s | 70.24 tok/s |
| Original final text release | 1377.81 tok/s | 71.51 tok/s |
| Current Vision source | **1375.16 tok/s** | **71.52 tok/s** |

## Comparing results fairly

Public comparisons should report GPU, VRAM, quantization profile, actual prompt tokens, max context, allocated KV capacity, KV dtype, prefill chunk, speculation settings, Vision settings, prefill speed and decode speed. Short-prompt decode numbers are not directly comparable with the 118K active-prompt result.
