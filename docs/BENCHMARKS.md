# Benchmarks

## Final validated result

Batch 212 ran the final 118,001-token workload while keeping the full 131,072-token maximum context and KV capacity allocated.

| Metric | Value |
|---|---:|
| Prompt tokens | 118001 |
| Max context | 131072 |
| KV capacity | 131072 |
| Prefill chunk | 896 |
| KV dtype | Q4 |
| Speculation | MTP-3 |
| Prefill | 1377.81 tok/s |
| Decode | 71.51 tok/s |
| MTP acceptance rate | 44.74% |
| MTP acceptance length | 2.31 |
| Free after weights | 2.56 GiB |
| Free after startup | 10.56 MiB |
| Runtime reservation | 2.55 GiB |
| KV cache payload | 2.26 GiB |
| Planned slack | 11.39 MiB |

All final runtime checks passed: no reservation failure, matrix-window error, contract error, out-of-memory error, or non-finite warning.

## Historical comparison

| Result | Prefill | Decode |
|---|---:|---:|
| Older baseline | 1235.03 tok/s | — |
| Historical optimized | 1371.10 tok/s | — |
| Historical B133 | 1377.66 tok/s | 70.24 tok/s |
| Final validated | **1377.81 tok/s** | **71.51 tok/s** |

The final prefill result is approximately **11.56% faster** than the older baseline, **0.49% faster** than the historical optimized mean, and effectively identical to the historical B133 run.

## Long-oracle validation

Batch 211 used a 3,201-token prompt with the full 131072 capacity. Prefill reached **2236.99 tok/s**, versus **2238.05 tok/s** historically, a difference of about **-0.047%**. The run exercised both `T=896` and `T=509` large-token-width paths.

## Short-oracle capacity validation

Batch 210 proved that the final mixed-Q4 artifact could start with both max context and KV capacity set to 131072 and return the deterministic JSON oracle correctly. It reported about **11.39 MiB planned slack**.

## Earlier compute-path isolation

Batch 201 tested the recovered large-T Q4 attention path before the final memory-profile fix. It produced **2226.37 tok/s** prefill on the 3,201-token oracle, confirming that the optimized compute path had been recovered before the final 128K artifact was produced.

## Comparing results fairly

Short-prompt decode numbers are not directly comparable with the 118K active-prompt result. Public comparisons should report GPU, VRAM, quantization profile, actual prompt tokens, max context, allocated KV capacity, KV dtype, prefill chunk, speculation settings, prefill speed and decode speed.
