# Publishing and Sharing Guide

## Recommended message

Lead with the complete workload, not only the decode number:

> Qwen3.8-27B on a single RTX 5080 16 GB with a full 131,072-token Q4 KV allocation. At an actual 118,001-token prompt: 1,377.81 tok/s prefill and 71.51 tok/s decode.

Avoid claiming a world record unless a controlled public comparison establishes it. The reproducible combination is strong enough without overclaiming.

## Recommended launch order

1. GitHub repository as the canonical technical source.
2. Hugging Face model card or conversion recipe, subject to redistribution rights.
3. r/LocalLLaMA technical post.
4. YouTube demonstration with live terminal evidence.
5. Hacker News if the technical write-up is mature.
6. Short social posts that link back to GitHub rather than replacing the technical record.
7. Relevant NInfer, Qwen, DFlash2, Blackwell and local-inference communities.

## What to show publicly

For every benchmark publish:

- GPU and VRAM
- system software versions
- source commit/tag
- model SHA256
- binary SHA256
- actual prompt token count
- maximum context
- allocated KV capacity
- KV dtype
- prefill chunk
- speculation settings
- prefill tok/s
- decode tok/s
- memory planner output

Raw logs are more useful than screenshots alone.

## Suggested forum title

**Qwen3.8-27B at true 131K context on one RTX 5080 16GB: 1377.8 tok/s prefill + 71.5 tok/s decode at 118K prompt**

## Independent reproduction

Encourage other users to submit results through the repository's Benchmark Result issue template.

A table of independent results across RTX 5080, RTX 5060 Ti 16 GB, RTX 5090 and other Blackwell cards will make the project much more useful than a single benchmark.

## Licensing

Before uploading converted model files, prebuilt binaries, or source bundles, verify the redistribution terms of NInfer and all incorporated upstream components. Preserve required license and copyright notices.

If redistribution is uncertain, publish the source changes, exact conversion recipe, pinned revisions and hashes first.
