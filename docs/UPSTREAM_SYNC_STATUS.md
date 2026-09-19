# Upstream Sync Status

This fork selectively integrates changes from [Neroued/ninfer](https://github.com/Neroued/ninfer) rather than attempting to remain commit-for-commit identical with upstream.

The fork contains RTX 5080-specific work, true-128K memory-fit changes, mixed-Q4/Q5 tuning, MTP/DFlash2 integration, and Vision changes that intentionally diverge from upstream. For that reason, GitHub's "ahead/behind" count is not a reliable statement of which upstream work has or has not been considered.

This file is the authoritative restart point for future upstream reviews.

## Current upstream review checkpoint

- Upstream repository: `Neroued/ninfer`
- Upstream branch: `master`
- Upstream commits assessed through: `9e163eee1f...`
- Checkpoint commit: `9e163eee` — `perf(ops): route the q4/q5 a16 input projections by column band`
- Assessment date: **2026-09-20**
- Fork branch assessed against: `main`
- Fork head at assessment: `f6088f856627045f280e5be8a76fba068b6979e4`
- Fork checkpoint description: RTX 5080 128K Vision v1.3 validation

**Future upstream review should start with commits after `9e163eee`.**

"Assessed through" means the upstream history reachable from that commit has been considered for relevance to this fork. It does **not** mean every upstream commit was merged. Some changes were already present under different SHAs, some were semantically backported, some were retuned for the RTX 5080, and some were intentionally deferred or skipped.

## Status vocabulary

| Status | Meaning |
|---|---|
| `MERGED` | Upstream change, or its semantic equivalent, is present in the fork. |
| `MERGED_AND_RETUNED` | Upstream mechanism is present, with RTX 5080-specific routing/tuning. |
| `ALREADY_PRESENT` | Equivalent functionality was already in the fork under a different history. |
| `PORT_AND_RETUNE` | Relevant upstream work identified for a future RTX 5080 integration pass. |
| `RECONCILE` | Relevant, but overlaps fork-specific code and must be compared rather than cherry-picked blindly. |
| `DEFER` | Potentially useful but not currently important to the validated Qwen3.8-27B RTX 5080 path. |
| `SKIP` | Not relevant to the current fork target. |
| `REJECTED_AFTER_TESTING` | Evaluated experimentally and deliberately not adopted. |

## Upstream-derived changes already integrated

The current validated branch already contains a number of upstream fixes and optimisations. Several were applied as semantic backports, so their fork SHAs intentionally differ from upstream.

| Upstream commit | Fork commit | Status | Notes |
|---|---|---|---|
| `b88c0f6f` | `3230f587` | `MERGED` | Completes host uploads before returning. |
| `641ef3e7` | `37800a4d` | `MERGED` | Skips NFC normalisation for text that is already pure ASCII. |
| `9f0575bb` | `77683b0d` | `MERGED` | Restores required BF16 definitions for the Q4 top-k kernel build. |
| `6d1da9ce` | `52776b66` + `0c3caf9a` | `MERGED_AND_RETUNED` | Backports the Q5 LinearAdd aggregate-cliff fix and then retunes the affected routing for RTX 5080. |
| `4cece118` | `4f453463` | `MERGED` | Semantic backport fixing malformed generated UTF-8. |
| `1d13c213` | `e5ce9863` | `MERGED` | Semantic backport binding the configured CUDA device across the engine lifecycle. |
| — | `dd2cb034` | `MERGED` | Follow-up compile repair required by the CUDA-device lifecycle backport. |

These changes were consolidated into the validated upstream integration history, including:

- `357a683b` — **Merge validated RTX 5080 upstream v1.1 integration checkpoint**
- `a2c2afb7` — **merge: validated RTX 5080 128K Vision v1.2**
- `f6088f85` — **docs: publish RTX 5080 128K Vision v1.3 validation**

The v1.1 integration checkpoint was validated on the RTX 5080 16 GB with Qwen3.8-27B, 131072 context, 131072 Q4 KV capacity, prefill chunk 896, MTP-3, and Vision. Its merge record explicitly includes the host-upload fix, ASCII NFC fast path, Q4 top-k BF16 build fix, and RTX 5080-qualified Q5 LinearAdd routing.

## Upstream commits assessed but not yet integrated

The following upstream work was identified during the 2026-09-20 review and should **not** be rediscovered from scratch during the next review.

| Upstream commit | Status | Current assessment |
|---|---|---|
| `9e163eee` | `PORT_AND_RETUNE` | Q4/Q5 A16 fused attention/GDN input-projection routing. Highly relevant to the mixed Q4/Q5 Qwen3.8 path, but route boundaries should be benchmarked on RTX 5080 rather than copied blindly. |
| `a9a0d10a` | `RECONCILE` | Q5 A16 LinearAdd T=1 and >512-column tail routing. Overlaps the fork's `52776b66` / `0c3caf9a` RTX 5080-specific Q5 work. Compare mechanisms and retain the best 5080 routing. |
| `bb844c43` | `PORT_AND_RETUNE` | Q4 4096x5120 Linear dispatch tuning. Candidate schedules are relevant; crossover points require RTX 5080 measurement. |
| `beedffa0` | `PORT_AND_RETUNE` | Q4 7168x5120 Linear dispatch tuning. Candidate schedules are relevant; crossover points require RTX 5080 measurement. |
| `d3c125ed` | `PORT_AND_RETUNE` | Q4 34816x5120 Linear dispatch tuning. Candidate schedules are relevant; crossover points require RTX 5080 measurement. |
| `5b4303c0` | `PORT_AND_RETUNE` | Q4 34816x5120 LinearSwiGLU follow-on tuning. Integrate only after the underlying Q4 linear routing has been evaluated. |
| `028eb61e` | `DEFER` | Q8 predicated GEMM cache-policy tuning. Potentially relevant to MTP/Q8 paths but lower priority than mixed-Q4/Q5 execution. |
| `b39de4d5` | `DEFER` | Q5 pure-Linear route tuning with strong microbenchmark gains, but upstream notes no confirmed model-level caller for the relevant current paths. |
| `dc58675f` | `SKIP` | Sparse-MoE Q5 routed-down tuning for Qwen3.6-35B-A3B, not the dense Qwen3.8-27B target of this fork. |
| `1d8587bc`, `05507ab0`, `5f5fccab` | `DEFER` | NVFP4/W4A4 improvements. Useful upstream work, but not part of the current validated mixed-Q4 true-128K artifact. |
| `b9219f3f` → `98dada0e` → `8eaed538` | `DEFER` | Custom Jinja/chat-template stack and literal-content fix. Desirable for serving compatibility, but large enough to treat as a separate frontend integration milestone. |

## Why GitHub can still report many commits "behind"

The fork and upstream histories diverged and some upstream commits were reapplied or adapted as new fork commits. Git therefore sees different commit identities even when equivalent code is already present.

For example:

- upstream `b88c0f6f` corresponds to fork `3230f587`
- upstream `9f0575bb` corresponds to fork `77683b0d`
- upstream `1d13c213` corresponds to fork `e5ce9863`
- upstream `4cece118` corresponds to fork `4f453463`

Accordingly, a large "behind" count must **not** be interpreted as "all of these commits still need merging."

## Procedure for the next upstream review

1. Read the checkpoint at the top of this file.
2. Fetch upstream `Neroued/ninfer:master`.
3. Review only commits after the recorded upstream checkpoint SHA.
4. Classify each relevant change using the status vocabulary above.
5. Integrate candidate work on a dedicated branch, not directly on `main`.
6. Benchmark hardware-sensitive routing on the RTX 5080 rather than assuming upstream GPU crossover values apply.
7. Re-run the true-128K text and Vision validation before merging into `main`.
8. Update this file with:
   - the new upstream checkpoint SHA,
   - assessment date,
   - fork head used for the review,
   - decisions made,
   - fork SHAs for any integrated/backported work.
9. After the integration is validated, optionally add a Git tag such as `upstream-reviewed-through-<short-sha>` to make the review point machine-readable as well as documented.

## Principle

The goal of this fork is **not** to reach "0 commits behind upstream."

The goal is to preserve the validated RTX 5080 / Qwen3.8-27B / true-128K configuration while selectively incorporating upstream fixes and performance mechanisms that improve it without breaking its memory envelope, correctness, MTP behaviour, or Vision support.
