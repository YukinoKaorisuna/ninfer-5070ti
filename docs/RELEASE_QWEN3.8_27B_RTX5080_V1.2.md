# Qwen3.8-27B RTX 5080 128K + Vision v1.2

## Validated source

`dd2cb0341c321f8a808a6ed75f0a53225983f718`

This release keeps the full 131,072-token text context and 131,072-token Q4 KV capacity while enabling Vision on a single RTX 5080 16 GB.

## Final acceptance

| Item | Result |
| --- | --- |
| Exact historical prompt | 118,001 tokens |
| Max context | 131,072 |
| KV capacity | 131,072 |
| KV dtype | Q4 group64 |
| Prefill chunk | 896 |
| Speculation | MTP-3 |
| CUDA graphs | disabled |
| Text prefill | 1380.61 tok/s |
| Text decode | 71.57 tok/s |
| MTP acceptance | 44.74% |
| MTP acceptance length | 2.31 tok/round |
| Vision profile | 2048 tokens |
| Vision workspace | 132.3142 MiB |
| Free after Vision startup | 8.56 MiB |
| Planned Vision slack | 10.08 MiB |
| Deterministic image | PASS |
| Deterministic video | PASS |
| Cached historical media 1/1/0 | PASS |
| Cached historical media 2/1/0 | PASS |
| Server survival | PASS |
| Strict OOM check | PASS - no OOM |

The 2048 Vision profile is intentionally extremely tight and should be started on a clean GPU.

## Artifact identities

`MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`

`CLI_SHA256=d07d3107cf946a0a2cd4d5e912e8c80d54901cea9de776f8c77194f062984c21`

`NINFER_SERVE_SHA256=3902e0a2ee93810d087366f7b0ec860aa5b4a2bb1c7b1a2a16aae45cac648ec0`

## Upstream audit included in v1.2

- malformed generated UTF-8 repair: backported and validated;
- speculative terminal KV coverage: equivalent lower-bound invariant already present and verified;
- CUDA device binding lifecycle: backported and validated;
- cooperative GDN launch capacity: existing runtime-SM-aware protection verified on the RTX 5080; the upstream tiled-launch implementation remains a future performance candidate.

Lower-priority serving/frontend robustness commits discovered during the audit were intentionally deferred because the fork has materially diverged in those areas and they are not required for this validated runtime profile.

Optional RTX 5090-derived Q4 route tuning was also deferred for separate RTX 5080 performance qualification.

## Validation status

**RELEASE CANDIDATE ACCEPTED.**

The release validation covered the exact 118,001-token historical regression workload plus Vision-2048 startup, deterministic image input, deterministic video input, progressive cached-image history and strict allocation-failure inspection.
