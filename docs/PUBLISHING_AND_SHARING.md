# Publishing and Sharing Guide

## Canonical public locations

Use these locations consistently in release notes, forum posts and videos:

- **Source / technical record:** https://github.com/toddballinger/ninfer-5080
- **Official RTX 5080 artifact:** https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080

The GitHub repository is the canonical source and validation ledger. The Hugging Face repository is the canonical distribution location for the validated `.ninfer` artifact.

Do not present a contributor fork or a third-party Hugging Face mirror as the official project release, even if it contains a byte-identical artifact. Independent reproductions are useful and should be credited as such.

## Canonical artifact identity

The official Qwen3.8-27B RTX 5080 artifact is:

```text
file:   qwen3_8_27b.ninfer
bytes:  16461267456
SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

Public documentation should use the SHA-256 as the primary identity rather than relying on local filenames or build-directory names.

The runtime and model artifact are versioned separately. A newer NInfer runtime does not imply a new model artifact unless the artifact SHA changes.

## Recommended message

Lead with the complete workload, not only the decode number:

> Qwen3.8-27B on a single RTX 5080 16 GB with a full 131,072-token Q4 KV allocation. The validated project artifact is published on Hugging Face with an immutable SHA-256, and the runtime has been qualified at an actual 118,001-token prompt.

When quoting performance, attach the exact validated commit and workload. For example, the `b44b1958` qualification measured 1378.85 tok/s prefill and 71.44 tok/s decode at 118,001 prompt tokens; the later `00e8e47f` Q5 semantic-port qualification measured 1376.30 tok/s prefill and 71.53 tok/s decode on the same workload.

Avoid world-record or “fastest” claims unless a controlled public comparison establishes them.

## Release publication order

For an official release or artifact update:

1. Validate the exact source commit, binary and artifact.
2. Record the immutable identities in [VALIDATED_MANIFEST.md](VALIDATED_MANIFEST.md).
3. Update the project README/release documentation.
4. Publish or update the project-owned Hugging Face repository.
5. Verify the Hub file size and SHA against the local validated artifact.
6. Only then publish forum, video or social announcements.

For future automated builds, CI should refuse publication when the produced artifact does not match the expected byte size and SHA-256 for a release that claims byte-identical reproduction.

## What to show publicly

For every benchmark publish:

- GPU and VRAM
- operating system / driver / CUDA versions where relevant
- exact source commit or release tag
- model SHA-256
- binary SHA-256
- actual prompt token count
- configured maximum context
- allocated KV capacity
- KV dtype
- prefill chunk
- speculation settings
- Vision token budget if enabled
- prefill tok/s
- decode tok/s
- MTP acceptance/acceptance length when relevant
- memory planner/startup envelope

Raw logs are more useful than screenshots alone.

## Suggested forum title

A neutral technical title is preferable to a superlative:

**Qwen3.8-27B at true 131K context on one RTX 5080 16GB — validated artifact, Vision and MTP-3**

Use the body of the post for exact performance numbers and commit-scoped comparisons.

## Independent reproduction

Independent reproduction is encouraged. Contributors can publish their own mirrors or build results, provided the distinction between an independent reproduction and the project-maintained release is clear.

A particularly useful independent result includes:

- source commit;
- pinned model revisions;
- produced artifact byte size and SHA-256;
- host RAM used during conversion;
- GPU/runtime validation details;
- any difference from the canonical artifact.

## GitHub Actions publication

The intended automation flow is:

```text
contributor PR
    -> review/merge in toddballinger/ninfer-5080
    -> build in project-owned GitHub Actions
    -> verify exact release gates
    -> publish with project-owned HF_TOKEN
    -> ninfer-5080/Qwen3.8-27B-RTX5080
```

The current public artifact predates completion of that automated flow and is the manually verified reference artifact. When CI reproduction is accepted, the documentation should be updated to state that the exact same artifact can be rebuilt and published automatically.

## Licensing and attribution

Before uploading converted model files, prebuilt binaries or source bundles, verify the redistribution terms of NInfer and all incorporated upstream/model components. Preserve required license and copyright notices.

Credit upstream projects and substantial contributor work explicitly, while keeping project ownership/provenance unambiguous.
