# Qwen3.8-27B true 128K + Vision on RTX 5080 16 GB

Vision is now the recommended path in this repository. The original text-only release remains preserved as the historical baseline.

Validated Vision source commit before merge to `main`:

`7c10db07ac8c5803f921b83603b707750652873e`

## Recommended serving command

Use `--vision-max-tokens 1792` as the safer default:

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

The maximum validated profile is `--vision-max-tokens 2048`. At 2048 the measured workspace/startup envelope was:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after startup   8.56 MiB
planned slack       10.08 MiB
```

A clean GPU is required for the 2048 profile.

## Validation

The Vision source was re-tested with the exact historical 118,001-token corpus using full 131,072 context/KV, Q4 KV, chunk 896 and MTP-3:

| Metric | Original | Vision source |
|---|---:|---:|
| Prefill | 1377.81 tok/s | 1375.16 tok/s |
| Decode | 71.51 tok/s | 71.52 tok/s |
| MTP acceptance | 44.74% | 44.74% |
| Acceptance length | 2.31 | 2.31 |

No meaningful text-performance regression was observed.

Image understanding is empirically validated. OpenWebUI multi-image history is also validated: cached historical media no longer consumes the fresh preprocessing budget again, while genuinely new media still does.

Video input is supported by the frontend, but video has not yet been empirically validated on the final 128K HostMapped Vision configuration.

## Validation hashes

```text
model SHA256:        c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
ninfer SHA256:       5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140
ninfer-serve SHA256: 61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```
