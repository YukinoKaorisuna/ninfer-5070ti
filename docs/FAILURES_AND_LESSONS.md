# Failures and Lessons

The final result depended heavily on preserving the failures, because several incorrect directions looked plausible until they were tested against the real model.

## 1. The broad historical merge was a dead end

A broad reconstructed merge imported stale runtime behavior from an older branch.

The most visible regression was a `matrix_window shape mismatch` caused by code that attempted to reuse a persistent `prefill_hidden_` buffer sized only for one hidden column as though it contained the full prefill width.

That branch was abandoned rather than patched incrementally because it had changed too many unrelated files and made provenance difficult to reason about.

Lesson: when recovering performance work from Git history, replay only the smallest known-good changes needed for the target behavior.

## 2. All-Q5-ish current artifact could not reserve long-context runtime memory

The current-contract artifact initially loaded correctly but failed before the 118K benchmark because the runtime reservation exceeded available capacity.

Observed shortfall at the earlier large-KV test:

```text
174020864 bytes
~165.96 MiB
```

This was not a compute-kernel failure. It exposed a model-weight memory-profile regression.

Lesson: separate model-weight residency from runtime/KV memory when diagnosing a near-capacity failure.

## 3. Historical mixed quantization was the decisive memory recovery

Git history revealed that the exact-128K configuration had selectively quantized 24 `value_z` tensors and 7 `gate_value` tensors to Q4.

Restoring that profile saved about 210.625 MiB of GPU weight memory—enough to recover the missing runtime headroom.

Lesson: selective quantization can be much more attractive than globally dropping model precision when only a few hundred MiB are needed.

## 4. GPU artifact conversion OOM

The converter attempted a large temporary PyTorch allocation during quantization and failed on the 16 GB GPU.

The observed failure included a roughly 4.74 GiB attempted allocation with most GPU memory already allocated or reserved.

The final artifact was therefore converted on CPU, which succeeded quickly and avoided competing with model-conversion scratch space on the target GPU.

Lesson: conversion-time VRAM requirements can be far higher than final inference residency. Do not assume a model that fits for inference can also be converted on the same GPU.

## 5. Synthetic Q4 tests were not sufficient

Some small-width synthetic tests passed while real-model outputs exposed numerical problems in the tensor-core route.

The validated solution therefore kept `T <= 256` on the known-safe route and used the optimized independent Q4 MMA path only for `T >= 257`.

Lesson: operator tests are necessary but not sufficient. Preserve deterministic real-model oracles.

## 6. Output stride mattered

The generic Q4 RowSplit MMA path required an output-stride fix. Without it, padded/non-default output layouts could be handled incorrectly.

Lesson: high-performance kernels often fail at layout boundaries before they fail at arithmetic boundaries.

## 7. CLI argument-size limit on the 118K prompt

The final benchmark prompt was approximately 491 KB and could not safely be passed as one shell argument.

A temporary `--prompt-file` CLI option was added only for the benchmark, the binary was rebuilt, the workload was run, and then the source was restored and rebuilt.

The restored binary SHA matched the original exactly.

Lesson: benchmark harness changes should be isolated from inference changes and proven reversible by hash.

## 8. 129024 versus 131072

129024 was used historically because it is exactly `144 x 896` and therefore convenient for chunk alignment.

It is not true binary 128K.

Final acceptance required:

```text
MAX_CONTEXT=131072
KV_CAPACITY=131072
```

Lesson: context-window claims should report both configured max context and actual KV capacity.

## 9. Near-zero memory margin changes engineering priorities

The final planner slack was only about 11.39 MiB.

At that margin, allocation order, first-use initialization, workspace aliasing and avoiding unnecessary persistent buffers all matter.

Lesson: once VRAM utilization approaches 100%, seemingly small runtime allocations become first-class architecture decisions.

## 10. Preserve the exact working state before publishing

The validated source was frozen before adding documentation:

```text
commit 473dade56031852a7d96edef049d859da96a6df9
tag    qwen3.8-27b-rtx5080-128k-v1
```

Documentation is being added on a separate branch so the public release tag always points to the code that actually produced the benchmark.

Lesson: freeze first, document second.
