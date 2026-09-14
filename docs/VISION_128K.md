# Qwen3.8-27B true 128K + Vision on RTX 5080 16 GB

Vision is now the recommended path in this repository. The original text-only release remains preserved as the historical baseline.

Validated Vision source commit before merge to `main`:

`7c10db07ac8c5803f921b83603b707750652873e`

## Recommended serving command

The recommended profile is empirically validated at `--vision-max-tokens 1792`:

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

Measured 1792 workspace/startup envelope:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      115.7751 MiB
free after startup  26.56 MiB
planned slack       28.88 MiB
```

The server reached the listening state with the full `131072 / 131072` text context/KV allocation unchanged. A deterministic 512×256 synthetic image containing a red left half and blue right half was correctly described as: “The left half is red and the right half is blue.” The request reported `prompt=211`, `prefill=685.8 tok/s`, `decode=118.3 tok/s`, `ttft=719 ms`, and MTP `3.10 tok/round (70.0%)`.

A deterministic 6-second synthetic MP4 containing red, then green, then blue scenes was also processed successfully at the same profile. With thinking disabled and a constrained output format, the model returned exactly `red, green, blue`. The video request reported `prompt=572`, `gen=6`, `prefill=1721.9 tok/s`, `decode=97.1 tok/s`, `ttft=1111 ms`, wall time `1.16 s`, and MTP `4.00 tok/round (100.0%)`.

The maximum validated profile is `--vision-max-tokens 2048`. At 2048 the measured workspace/startup envelope was:

```text
text_prefill       116.0127 MiB
mtp_prefill        116.0127 MiB
vision_encode      132.3142 MiB
free after startup   8.56 MiB
planned slack       10.08 MiB
```

A clean GPU is required for these true-128K profiles; the 2048 setting is especially tight.

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

Video input is now empirically validated on the final 128K HostMapped Vision configuration using a deterministic chronological-color MP4 test. This validates the end-to-end video acquisition, preprocessing, Vision encode and generation path on the RTX 5080 configuration; it is not a broad video-quality benchmark.

## Validation hashes

```text
model SHA256:        c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
ninfer SHA256:       5ef4df2862ac5b2f63359cc86188a5f91636e54bd07d6417e6567c7a86076140
ninfer-serve SHA256: 61dbffa243a54bf32db8c6f55f4b1db288c1ebcfe390dff50ec9a1ba7a2f7399
```
