# NInfer RTX 5080 / Qwen3.8-27B Research History

## Purpose

This document records the development and optimization history of the
`ruwwww/ninfer-5060ti` fork for running Qwen3.8-27B on an NVIDIA RTX 5080
16 GB GPU.

It is intentionally broader than the later formal research log. Significant
engineering work occurred before the B000 research baseline was created.

The history should therefore be understood as:

    Original ruwwww/ninfer-5060ti fork
            |
            v
    Qwen3.8 enablement and kernel development
            |
            v
    B000 formal research baseline
            |
            v
    E101 Q3 A8/INT8 prefill
            |
            v
    E200 A8 attention/GDN prefill

Do not treat B000 as the untouched upstream/fork baseline.

---

# 1. Original Repository

Original fork:

    https://github.com/ruwwww/ninfer-5060ti.git

Additional research remotes used later:

    upstream:
    https://github.com/Neroued/ninfer.git

    ninfer4090:
    https://github.com/tensorninja/ninfer-4090.git

    ninfer5090:
    https://github.com/sergiuszm/ninfer-5090.git

Target hardware:

- NVIDIA GeForce RTX 5080 16 GB
- Blackwell / SM120
- Ubuntu VM ("Brain")
- CUDA 13.3
- NVIDIA driver 595.x
- Host PCIe Gen3 x16
- GPU production power limit: 330 W

---

# 2. Qwen3.8-27B Enablement

The original fork did not represent the final Qwen3.8-27B configuration now
in use.

The shared Qwen3.6/Qwen3.8 target and artifact tooling were adapted to support
the Qwen3.8-27B model and custom mixed-precision artifact.

Relevant areas included:

    src/targets/qwen3_6_27b/
    tools/convert/qwen3_8_27b/
    tools/convert/qwen3_6/common/
    tools/artifact/

Historical backups in the repository such as:

    *.pre-q3
    *.bak
    *.v1

are evidence of this pre-B000 development and must not be mistaken for active
production code.

---

# 3. Current Custom Qwen3.8 Artifact

Production artifact:

    /models/ninfer-custom/qwen3_8_27b_5080_v7_q3_downq4_embq5.ninfer

Approximate artifact size:

    13.02 GiB

Approximate effective precision:

    4.09 BPW over ~27.32B parameters

Current mixed quantization layout:

| Component | Quantization |
|---|---|
| Embeddings | Q5 |
| LM output | Q4 |
| Attention Q/K | Q4 |
| Attention gate/value | Q5 |
| Attention output | Q4 |
| GDN Q/K | Q4 |
| GDN value/Z | Q5 |
| GDN output | Q4 |
| MLP gate/up | custom Q3 |
| MLP down | Q4 |
| KV cache | Q4 |
| MTP | included |

---

# 4. Custom Q3G64 Support

A custom Q3 groupwise format was introduced for the large MLP gate/up
projection.

Qwen3.8 geometry:

    gate_up rows = 34816
    intermediate = 17408
    K            = 5120
    group K      = 64

Q3 signed values are represented approximately as:

    -4 .. +3

with group-64 FP16 scales.

This required both runtime kernel support and artifact/converter support.

Numerous Q3 kernel experiments remain as untracked historical snapshots,
including variants involving:

- direct-X
- fast GEMV
- funnel kernels
- x-load batching
- different warp configurations
- T=4 / T=8 / T=16 / T=32 / T=40 / T=48 / T=64
- BF16 Tensor Core prefill experiments

These are historical experiments and are not necessarily active code.

---

# 5. Q4 Qwen3.8 Kernel Work

Q4 runtime paths were adapted and tuned for important Qwen3.8 geometries,
including:

    N=5120   K=6144
    N=5120   K=17408
    N=248320 K=5120

Areas investigated included:

    src/ops/linear/q4/q4_dispatch.cpp
    src/ops/linear/q4/q4_dispatch.h
    src/ops/linear/q4/q4_launch.h
    src/ops/linear/q4/q4_small_t_mma.cu
    src/ops/linear/q4/q4_small_t_mma.cuh

Historical experiments included:

- alternate CTA geometry
- Grid-K
- small-T MMA paths
- group double buffering
- fragment ping-pong
- Qwen3.8-specific T=4 routes
- vocabulary/head-specific paths

Most experimental variants were rejected when they failed to provide useful
end-to-end gains.

---

# 6. Q4 KV Cache Support and Correctness

Q4 KV cache support required correctness work in the GQA/KV path.

Relevant areas included:

    src/ops/kernel/gqa_attention_decode_i8.cuh
    src/ops/kernel/gqa_attention_kv_quant.cuh
    src/ops/launcher/gqa_attention_decode.cu

Historical snapshots include:

    gqa_attention_decode_i8.cuh.pre-q4-decode
    gqa_attention_kv_quant.cuh.pre-q4-kv
    gqa_attention_decode.cu.pre-q4-decode

The corrected Q4 KV implementation was validated at approximately:

    18K
    32K
    60K
    90K

and supports the current production configuration:

    --max-context 98304
    --kv-dtype q4

---

# 7. Q3 Decode Optimization

Significant work occurred before E101 to make Q3 decode viable.

Experimental strategies included:

- fast GEMV
- direct-X
- funnel kernels
- warp-count changes
- x-load batching
- small-T MMA
- T=4 through T=64 routes

The final T=1 Q3 decode kernel became a known-good path and is now considered
FROZEN unless explicitly revisited.

Approximate compiled characteristics:

    REG:    40
    STACK:  0
    SHARED: 36096
    LOCAL:  0

Project rule:

    DO NOT MODIFY THE FROZEN Q3 T=1 DECODE KERNEL
    WITHOUT AN EXPLICIT EXPERIMENT.

Current realistic decode performance is approximately:

    ~127 tok/s

with MTP in OpenClaw-style testing.

---

# 8. MTP Integration

The current artifact includes MTP support.

Typical production options:

    --spec mtp
    --draft-tokens 3

MTP has been validated as part of the current Qwen3.8 runtime configuration.

---

# 9. 98K Context / Memory Fit

The target production configuration became:

    context = 98304
    KV      = Q4
    MTP     = 3

Typical startup memory:

    weights             ~12.58 GiB
    KV runtime           ~2.07 GiB
    free after weights   ~2.62 GiB

With E200 the server has reported approximately:

    free-after-startup ~570 MiB

Fitting the custom artifact, 98K Q4 KV, MTP and runtime workspaces into a
16 GB RTX 5080 has been a primary design constraint throughout the project.

---

# 10. Prefill Chunk Tuning

Prefill chunk sizes investigated:

    256
    512
    768
    1024
    1280
    1536+

Representative old ~90K prefill results:

    chunk 512   ~735.5 tok/s
    chunk 768   ~749.9 tok/s
    chunk 1024  ~752.4 tok/s

1280 failed runtime reservation by approximately 11.9 MiB.

The gain from 768 to 1024 was only about 0.34%, so larger chunks were not
considered worth sacrificing context/memory for.

Current production choice:

    --prefill-chunk 1024

---

# 11. RTX 5080 Stability / 330 W Power Limit

During earlier heavy testing, unrestricted GPU power resulted in:

    NVIDIA Xid 79
    GPU fell off bus

A 330 W power cap was tested extensively and became the production setting.

Ten consecutive heavy ~90K prefills passed at 330 W.

Comparable llama.cpp performance showed essentially no loss:

    330 W mean       ~1262.0 tok/s
    unrestricted     ~1260.8 tok/s

Therefore:

    Production GPU power limit = 330 W

The host is PCIe Gen3 x16 by platform design. 8.0 GT/s x16 is therefore the
expected physical platform limit and should not be described as degraded.

---

# 12. Formal B000 Research Baseline

Formal research logging began only after substantial earlier engineering work.

Branch:

    research/5080-qwen38-20260908

Commit:

    8de67f959e660839d41a154c288378bd67b30cea

Tag:

    b000-5080-qwen38-98k

IMPORTANT:

    B000 IS NOT THE ORIGINAL FORK.

It already contains substantial Qwen3.8 compatibility, Q3, Q4, KV,
long-context, conversion and performance work.

For research documentation, distinguish:

    Original fork baseline
        vs
    B000 engineering/research baseline

---

# 13. E101 - Q3 A8 INT8 Tensor-Core Prefill

E101 was the largest performance breakthrough.

Previous large-T Q3 prefill used a T32-oriented BF16 Tensor Core route.

At T=1024 this effectively caused:

    32 x T32 jobs

per dispatcher call, repeated across thousands of calls.

This became the dominant prefill bottleneck.

## E101 architecture

The new path became:

    BF16 activation
        ->
    G64 symmetric INT8 activation quantization
        ->
    INT8 activation

and:

    Q3 signed weight
        ->
    decode directly to INT8

followed by:

    INT8 x INT8 Tensor Core MMA
        ->
    INT32 accumulation
        ->
    activation scale x weight scale
        ->
    fused SwiGLU

Tensor Core instruction family:

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32

Imported/adapted infrastructure from tensorninja/ninfer-4090 included:

    src/ops/common/act_quant_g64.cu
    src/ops/common/act_quant_g64.h
    src/ops/common/int8_mma.cuh

Representative schedule:

    Q3Int8SwiGluSchedule<64,256,16,128,3,1>

Approximate shared memory:

    78,848 bytes

Workspace:

    5440 bytes/token

with up to a 4096-token activation tile.

---

# 14. E101 Routing

The final promoted policy is approximately:

    Q3 T=1
        -> frozen custom decode kernel

    Q3 T=2..256
        -> existing A16/small-T paths

    Q3 large prefill T>=257
        -> A8 INT8 Tensor Core path

The target maps:

    QType::Q3G64_F16S
        -> LinearPolicy::AllowA8

Initial E101 implementation:

    commit ce1f6fea8f0102b2c96706d10bddc651ba850b8c

Promoted policy cleanup:

    commit 631db68ab02d9b4689ea66c2468d1db5d08d9e1f

Tag:

    e101-q3-a8-policy-promoted

---

# 15. E101 Performance

Controlled ~90K comparison:

    E101 ON    1421.1 tok/s
    A16 OFF     761.7 tok/s

Improvement:

    +86.6%

Five-run validation:

    1389.4
    1389.5
    1394.6
    1394.5
    1391.5

Mean:

    1391.9 tok/s

CV:

    0.18%

Comparable llama.cpp long-context result:

    ~1264 tok/s

E101 therefore moved NInfer ahead of llama.cpp at ~90K context.

---

# 16. OpenClaw V2 after E101

Representative prefill results:

| Prompt | NInfer | llama.cpp |
|---:|---:|---:|
| 650 | 1909.77 | 738.19 |
| 2.5K | 2043.82 | 1388.36 |
| 10K | 2022.55 | 1672.97 |
| 32K | 1709.42 | 1575.15 |
| 64K | 1506.72 | 1401.02 |
| 90K | 1370.07 | 1265.78 |

Decode512:

    NInfer     127.03 tok/s
    llama.cpp  121.61 tok/s

OpenClaw V2 showed no measurable NInfer intelligence regression caused by
E101.

Do not claim an intelligence win; the semantic result was effectively tied.

---

# 17. E102 Decode Investigation

CUDA graph and LM-head placement were tested.

At ~77K:

    baseline       74.70
    CUDA graph     75.27   (+0.76%)
    LM head        75.70   (+1.34%)

At 98K:

    CUDA graph failed by ~47.7 MiB
    LM head failed by ~306 MiB
    both failed by ~388 MiB

Conclusion:

    Not worth adopting.

Production remains:

    --no-cuda-graph

and LM head is not forced to the GPU.

---

# 18. Other Decode Research Not Adopted

Investigated external work included:

## sergiuszm/ninfer-5090

FP16 PV accumulation on consumer architectures.

Whole-server gains appeared approximately:

    2-5%

Not adopted as a priority.

## Upstream linear-attention state L2 residency

Potentially useful, but no evidence yet of >10% whole-model improvement on
this exact configuration.

## DFlash2

One of the few plausible decode technologies capable of materially larger
improvement, but companion artifact/weight requirements and VRAM demands make
it difficult with:

    RTX 5080 16 GB
    98K context

Not currently implemented.

---

# 19. E200 - A8 Attention and GDN Input Projection

Current development branch:

    e200-a8-attn-gdn-input

E200 imports/adapts INT8 A8 projection infrastructure from
`tensorninja/ninfer-4090`.

New infrastructure includes:

    src/ops/common/int8_proj_launch.cu
    src/ops/common/int8_proj_launch.h
    src/ops/common/int8_rowsplit_gemm.cuh

New attention implementation:

    src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_int8.cu

New GDN implementation:

    src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_int8.cu

Corresponding plans, kernels, wrappers, public APIs and CMake sources were
updated.

---

# 20. E200 Attention Projection

Qwen3.8 attention input weights:

    query/key      Q4
    gate/value     Q5

A BF16 hidden activation is quantized once to symmetric group-64 INT8 and
shared across the four logical projections:

    Query
    Key
    Gate
    Value

The persistent Q4/Q5 weights are decoded directly into signed INT8 values for
Tensor Core contraction.

---

# 21. E200 GDN Projection

Qwen3.8 GDN input weights:

    query/key      Q4
    value/Z        Q5

The activation is quantized once and shared across the GDN projections.

This avoids separately quantizing the same hidden activation for each
projection.

---

# 22. E200 Prefill-Only Policy

E200 is intentionally restricted to prefill.

Policy:

    TextPhase::Prefill
        Q4/Q5 split attention input -> AllowA8
        Q4/Q5 split GDN input       -> AllowA8

    Decode
        -> A16

    Verify
        -> A16

The shared target uses a helper equivalent to:

    text_proj_pair_policy(weight, phase)

which returns AllowA8 only for:

    Q4G64 weight
    AND
    TextPhase::Prefill

This intentionally protects the established decode path.

---

# 23. Qwen3.6 Compatibility

Because Qwen3.6 and Qwen3.8 share target implementation code, E200 preserves:

    4096-wide Qwen3.6 groupwise paths -> A16 only

while allowing:

    5120-wide Qwen3.8 split paths -> A8 during prefill

This distinction is important and must be retained.

---

# 24. E200 Preliminary Performance

Valid ~90K runs so far:

Run 1:

    prompt tokens = 89908
    prefill       = 1457.3 tok/s

Run 2:

    prompt tokens = 89908
    prefill       = 1478.1 tok/s

Two-run mean:

    1467.7 tok/s

E101 repeatable baseline:

    1391.9 tok/s

Mean improvement:

    ~+5.45%

Second-run improvement:

    +6.19%

Although this is below the original >10% optimization-target threshold, the
gain is real and E200 is being retained for further testing.

---

# 25. E200 Stability

The first apparent "SSH crashes" during E200 testing were NOT NInfer crashes.

They were caused by running:

    set -euo pipefail

directly inside the interactive SSH login shell.

Examples:

- an HTTP 400 caused Python to return non-zero and `set -e` exited Bash;
- `grep` returning no match caused `set -e` to exit Bash;
- after reconnecting, `$OUT` was unset and `set -u` produced:

      bash: OUT: unbound variable

  which exited the interactive shell and disconnected SSH.

Subsequent valid E200 testing showed:

    no NVIDIA Xid
    no GPU fallen-off-bus event
    no AER error
    no OOM
    no kernel failure

Future strict benchmark scripts should therefore use a child shell:

    bash -euo pipefail <<'BASH'
    ...
    BASH

Never enable strict mode globally in the interactive SSH shell.

---

# 26. Current Architecture

Approximate prefill path:

                         Qwen3.8-27B
                              |
                        BF16 hidden
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        Attention input    GDN input       MLP gate/up
          Q4 / Q5           Q4 / Q5            Q3
             |                |                |
             v                v                v
         G64 A8 quant     G64 A8 quant     G64 A8 quant
             |                |                |
             v                v                v
          INT8 TC          INT8 TC          INT8 TC
                                               |
                                          fused SwiGLU

Current decode remains approximately:

    Attention/GDN split inputs -> A16
    Q3 MLP T=1                -> frozen custom decode kernel
    KV                         -> corrected Q4 KV path
    MTP                        -> draft 3

---

# 27. Development Timeline

    ORIGINAL ruwwww/ninfer-5060ti
            |
            +-- Qwen3.8 enablement
            +-- artifact/converter changes
            +-- mixed Q3/Q4/Q5 artifact
            +-- Q3G64 support
            +-- Q4 Qwen3.8 geometry work
            +-- Q3 decode experiments
            +-- frozen fast Q3 T=1 decode
            +-- Q4 KV implementation/corrections
            +-- long-context validation
            +-- MTP integration
            +-- 98K memory fitting
            +-- numerous rejected Q3/Q4 experiments
            |
            v
    B000 FORMAL RESEARCH BASELINE
    8de67f959e660839d41a154c288378bd67b30cea
    tag: b000-5080-qwen38-98k
            |
            +-- prefill chunk tuning -> 1024
            |
            v
    E101
    ce1f6fea8f0102b2c96706d10bddc651ba850b8c
            |
            +-- A8 activation quantization
            +-- Q3 -> INT8 decode
            +-- INT8 Tensor Core SwiGLU
            +-- ~80%+ long-prefill improvement
            |
            v
    E101 POLICY PROMOTED
    631db68ab02d9b4689ea66c2468d1db5d08d9e1f
    tag: e101-q3-a8-policy-promoted
            |
            +-- proper LinearPolicy::AllowA8 routing
            +-- experimental environment gate removed
            |
            v
    E102
            |
            +-- CUDA graph investigation
            +-- LM-head investigation
            +-- gains only ~1%
            |
            +-- REJECTED
            |
            v
    E200
    branch: e200-a8-attn-gdn-input
            |
            +-- A8 Q4/Q5 attention input
            +-- A8 Q4/Q5 GDN input
            +-- shared activation quantization
            +-- prefill-only policy
            +-- decode remains unchanged
            +-- preliminary improvement ~5-6%
            |
            v
       CURRENT TREE

---

# 28. Experimental Policy

Current optimization policy:

1. Preserve correctness first.
2. Preserve 98K context where practical.
3. Preserve the frozen Q3 T=1 decode kernel.
4. Compare whole-model performance, not just microbenchmarks.
5. Use native request-complete prefill timing:

       computed_prefill_tokens / timings_seconds.prefill

   or the equivalent final server request timing.

6. Do not use arbitrary periodic throughput samples as the headline result.
7. Re-run OpenClaw V2 intelligence testing after numerical-path changes.
8. Historically, >10% expected gain was the threshold for prioritizing an
   optimization, although smaller proven gains may still be retained when
   they are stable and composable.
9. "Competitive with llama.cpp" means >= llama.cpp, not merely closer.
10. Keep experiments reproducible and preserve failed results where useful.

---

# 29. Important Baseline Definitions

For future reports and papers, use these distinct baselines:

## Original Fork Baseline

The original downloaded `ruwwww/ninfer-5060ti` repository before our
Qwen3.8-specific engineering.

## B000 Engineering Baseline

Commit:

    8de67f959e660839d41a154c288378bd67b30cea

Tag:

    b000-5080-qwen38-98k

This already contains significant compatibility and performance engineering.

## E101 Baseline

Commit/tag:

    631db68ab02d9b4689ea66c2468d1db5d08d9e1f
    e101-q3-a8-policy-promoted

This contains the major Q3 A8/INT8 prefill redesign.

## E200 Current Baseline

Current `e200-a8-attn-gdn-input` working tree.

This adds prefill-only A8 INT8 Q4/Q5 attention and GDN input projection.

A permanent E200 commit/tag should be created after stability validation.

---

# 30. Critical Things Not To Forget

- B000 is NOT the original fork.
- A substantial body of Qwen3.8 work predates formal research logging.
- Q4 KV correctness was specifically validated at long context.
- The custom mixed Q3/Q4/Q5 artifact is fundamental to the current result.
- The Q3 T=1 decode kernel is frozen.
- Production prefill chunk is 1024.
- Production context is 98304.
- Production KV is Q4.
- Production GPU power cap is 330 W.
- MTP draft count is normally 3.
- E101 was the dominant prefill breakthrough.
- E200 currently provides an additional approximately 5-6%.
- Interactive `set -euo pipefail` caused SSH disconnects and must not be
  mistaken for NInfer/GPU instability.
- Strict benchmark scripts should run in child shells.
- Historical `.pre-*` files document experiments but are not production code.

---

Last major update:

    2026-09-08
    E200 stability/performance validation in progress.

---

# 31. E200 Final Stability Validation

E200 was subjected to five consecutive fresh-server ~90K prefill runs.

Result directory:

    /home/toddballinger/benchmark-results/e200-stability5-20260908-122521

Each run:

- restarted `ninfer-serve`
- loaded the model from scratch
- used max_context=98304
- used Q4 KV
- used prefill_chunk=1024
- disabled prefix reuse
- used the same 89,908-token prompt
- completed normally

Results:

| Run | Prefill tok/s | TTFT |
|---:|---:|---:|
| 1 | 1511.9 | 59.538 s |
| 2 | 1507.0 | 59.731 s |
| 3 | 1503.8 | 59.860 s |
| 4 | 1504.5 | 59.829 s |
| 5 | 1503.8 | 59.859 s |

Summary:

    mean = 1506.2 tok/s
    min  = 1503.8 tok/s
    max  = 1511.9 tok/s
    SD   = 3.45 tok/s
    CV   = 0.23%

E101 repeatable baseline:

    1391.9 tok/s

E200 improvement over E101:

    +8.21%

Kernel/system validation after the five-run test:

    no NVIDIA Xid
    no GPU fallen-off-bus event
    no AER error
    no OOM event
    no relevant kernel error

Conclusion:

    E200 PASSED STABILITY VALIDATION.

The earlier apparent SSH failures during E200 development were caused by
`set -euo pipefail` being enabled in the interactive SSH login shell and were
not NInfer or GPU crashes.

E200 is therefore retained and promoted as the current performance baseline.

Final validated E200 long-context prefill:

    1506.2 tok/s @ 89,908 prompt tokens

Improvement relative to E101:

    +8.21%

Although the original experiment prioritization threshold was >10%, the
measured ~8.2% gain is highly repeatable, stable, composable with E101, and
therefore worth retaining.
