# Qwen3.8-27B RTX 5080 128K + Vision v1.3

## Validated runtime source

`ceb32f7d002edab224a83a2e2609f45fca4f8919`

The validated v1.3 runtime is based on the v1.2 runtime head and adds:

- corrected physical output strides for the Q4/Q4 small-T attention projection path;
- updated Qwen3.6 runtime/load-plan test expectations for the current runtime contracts;
- server-wide default thinking-budget support with per-request client override;
- configurable stable-turn / rolling-tool prefix checkpoints for agent tool loops.

The published v1.2 release commit and tag are preserved as ancestors of the v1.3 release history. Documentation-only release commits after the validated runtime head do not change the validated binary.

## Runtime profile

| Item | Result |
| --- | --- |
| Model | Qwen3.8-27B |
| GPU | NVIDIA GeForce RTX 5080 16 GB |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| CUDA graphs | disabled |
| Max concurrency | 1 |
| Vision | enabled |
| Vision token profile | 2048 |
| Default thinking budget | 2048 |
| Prefix checkpoint policy | `rolling-tool` for validated OpenClaw production |
| Default output tokens | 8192 |

## Artifact identities

`MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`

`NINFER_SERVE_SHA256=3179bfbcb88a72c04b983f28c25c62db468fbc8ef267fe043899de30a4281c56`

Validated production server binary:

`/models/ninfer-builds/releases/qwen3.8-27b-rtx5080-128k-vision-v1.3/ninfer-serve`

Validated model artifact:

`/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer`

## v1.3 runtime changes

### Q4/Q4 strided attention output

The small-T Q4 split-output launcher previously supplied logical row counts as output leading dimensions even when the wrapper accepted padded row-contiguous output tensors.

The launcher now supplies the physical BF16 column strides:

- `q.nb[1] / sizeof(__nv_bfloat16)`
- `key.nb[1] / sizeof(__nv_bfloat16)`

This matches the kernel's existing `out_ld` / `out_tail_ld` contract and the behavior already used by the larger-T Q4 MMA path.

The dedicated strided regression test now passes while contiguous production geometry remains numerically unchanged.

### Default thinking budget

`ninfer-serve` now supports:

`--default-thinking-budget N`

For thinking-enabled requests:

1. an explicit client `reasoning_budget` takes precedence;
2. otherwise the server default is used;
3. the budget is a maximum, not a requirement that the model consume every available reasoning token.

Thinking-disabled requests do not receive an effective reasoning budget.

Request logging schema version 11 records the resolved budget and its source.

## Rolling tool checkpoint reuse

`ninfer-serve` now supports:

```text
--prefix-checkpoint-policy stable-turn|rolling-tool
```

The global default remains `stable-turn`, preserving previous behavior for ordinary clients.

`rolling-tool` is intended for append-only agent/tool-loop workloads. After completed tool history exists within the current real user turn, the private `TurnClosure` checkpoint advances to the current generation opener. A new true user turn establishes a new stable boundary. `preserve_thinking=true` continues to use the existing response-replay checkpoint behavior.

This avoids the previous failure mode where repeated `restore_turn_checkpoint` operations stayed anchored near the first assistant response while the prompt continued to grow.

### Isolated OpenClaw validation

| Request | Prompt | Cache hit | New prefill | Reuse path |
| ---: | ---: | ---: | ---: | --- |
| 1 | 21,597 | 0 | 21,597 | `full_reset` |
| 2 | 21,936 | 21,853 | 83 | `append_frontier` |
| 3 | 22,371 | 21,934 | 437 | `restore_turn_checkpoint` |
| 4 | 24,122 | 22,369 | 1,753 | `restore_turn_checkpoint` |
| 5 | 27,365 | 24,120 | 3,245 | `restore_turn_checkpoint` |
| 6 | 30,454 | 27,363 | 3,091 | `restore_turn_checkpoint` |
| 7 | 31,294 | 30,452 | 842 | `restore_turn_checkpoint` |

Restore checkpoint sequence:

```text
21934 -> 22369 -> 24120 -> 27363 -> 30452
```

Result: **4 advances, 0 plateaus, 0 regressions**.

### Production OpenClaw smoke validation

| Request | Prompt | Cache hit | New prefill | Reuse path |
| ---: | ---: | ---: | ---: | --- |
| 1 | 18,551 | 0 | 18,551 | `full_reset` |
| 2 | 19,025 | 18,718 | 307 | `append_frontier` |
| 3 | 21,148 | 19,023 | 2,125 | `restore_turn_checkpoint` |
| 4 | 24,666 | 21,146 | 3,520 | `restore_turn_checkpoint` |
| 5 | 26,643 | 24,664 | 1,979 | `restore_turn_checkpoint` |
| 6 | 27,122 | 26,641 | 481 | `restore_turn_checkpoint` |

Restore checkpoint sequence:

```text
19023 -> 21146 -> 24664 -> 26641
```

Result: **3 advances, 0 plateaus, 0 regressions**. All five continuation requests had an uncached suffix below 4,096 tokens.

## Focused validation

| Validation | Result |
| --- | --- |
| Candidate configure/build | PASS |
| `ninfer_qwen3_6_runtime_mechanisms_test` | PASS |
| `ninfer_qwen3_6_27b_load_plan_test` | SKIPPED - external artifact fixture absent; target builds successfully |
| `ninfer_serve_options_test` | PASS |
| `ninfer_qwen3_6_frontend_test` rolling-checkpoint regression | PASS |
| Isolated OpenClaw rolling-tool validation | PASS |
| Production OpenClaw rolling-tool smoke test | PASS |
| `ninfer_request_log_test` | PASS |
| `ninfer_attn_input_proj_test` | PASS |
| Q4/Q4 strided T=17 / T=129 regression | PASS |
| Candidate 128K/Q4/MTP3 startup | PASS |
| Live service restoration after validation | PASS |

## Combined v1.3 MTP sanity sample

Three 512-token requests using an explicit 64-token reasoning budget produced:

| Run | Decode | MTP tok/round | Acceptance |
| --- | ---: | ---: | ---: |
| 1 | 86.8 tok/s | 2.26 | 42.1% |
| 2 | 83.7 tok/s | 2.19 | 39.7% |
| 3 | 83.6 tok/s | 2.19 | 39.6% |
| Average | **84.7 tok/s** | **2.213** | **40.47%** |

These short-request measurements are a runtime sanity check and are not a replacement for the v1.2 exact 118,001-token benchmark.

The previously validated v1.2 exact long-context result remains:

- 1380.61 tok/s prefill;
- 71.57 tok/s decode;
- 44.74% MTP acceptance;
- 2.31 tok/round;
- 131,072 context and 131,072 Q4 KV.

## Thinking-budget acceptance

Explicit client override:

`reasoning_budget=64`, source `client` -> **PASS**, actual reasoning 64 tokens.

Server default:

`reasoning_budget=2048`, source `server_default` -> **PASS**.

The default-budget validation request naturally completed after 81 reasoning tokens with a stop token. This is correct: the 2048 value is the maximum available reasoning budget, not a forced token count.

## Vision and memory

The candidate successfully started with the same validated v1.2 runtime profile:

- Vision enabled;
- Vision maximum 2048 tokens;
- full 131,072 context;
- full 131,072 Q4 KV;
- MTP-3;
- approximately 8.6 MiB device memory remaining after startup.

The deterministic image/video/cached-history and strict OOM acceptance suite was established for v1.2. The v1.3 changes do not modify the Vision pipeline or the validated memory geometry.

## Test-suite status

Three unrelated full-suite items remain intentionally deferred from the earlier v1.2 audit:

- `ninfer_speculative_round_test`;
- `ninfer_linear_swiglu_q3_a16_test`;
- `ninfer_qwen3_8_27b_dflash2_real_test`.

They are not required by the validated Qwen3.8-27B RTX 5080 Q4/MTP3 runtime profile and were not changed as part of v1.3.

## Validation status

**V1.3 PRODUCTION RELEASE VALIDATED.**

The production release passed the focused source, build, attention-stride, serving, request-logging, default-thinking-budget, live MTP, isolated rolling-tool and production OpenClaw rolling-checkpoint validation gates.
