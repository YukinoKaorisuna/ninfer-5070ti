from pathlib import Path
import json
import random
import sys

out = Path(sys.argv[1])

rng = random.Random(5080118001)

projects = [
    "runtime-qualification",
    "embedding-offload",
    "vision-validation",
    "tool-loop-regression",
    "context-planning",
    "kernel-optimization",
    "artifact-release",
    "benchmark-reproduction",
]

stages = [
    "inspect",
    "plan",
    "configure",
    "build",
    "validate",
    "benchmark",
    "review",
    "finalize",
]

tool_names = [
    "read_file",
    "write_file",
    "run_command",
    "git_status",
    "inspect_gpu",
    "query_model",
    "fetch_logs",
    "compare_results",
]

def header(index, kind):
    return (
        f"\n============================================================\n"
        f" WORKFLOW SEGMENT {index:05d} - {kind}\n"
        f"============================================================\n"
    )

def agent_planning(i):
    project = projects[i % len(projects)]
    stage = stages[(i * 3) % len(stages)]

    return header(i, "AGENT PLANNING") + f"""
System:
You are working on the synthetic project "{project}". Preserve provenance, do not
change unrelated files, verify hashes before benchmarking, prefer reproducible
commands, and return control to the existing SSH session after every test.

User:
Continue the {stage} phase. Accuracy and correct tool calls matter more than
speed. Keep the production service available unless the GPU must be isolated
for a benchmark.

Assistant analysis summary:
The next operation should verify the current branch, source commit, artifact
identity, runtime flags, GPU state, and expected output contract. If a benchmark
needs a clean accelerator, stop the local model service first, wait for compute
processes to drain, then confirm memory usage before loading the candidate.

Expected checks:
- repository state is clean except for explicitly scoped files
- model artifact SHA256 matches the documented identity
- context and KV capacity are consistent
- prefill and decode metrics are captured separately
- MTP acceptance and accepted tokens per round are recorded
- production service is restarted and health checked afterward
"""

def bash_transcript(i):
    branch = f"feature/synthetic-{i % 37:02d}"
    commit = f"{rng.getrandbits(160):040x}"

    return header(i, "BASH SSH TRANSCRIPT") + f"""
toddballinger@brain:~/ninfer-worktree$ git status --short
 M src/runtime/engine.cpp
?? tests/test_runtime_case_{i:05d}.cpp

toddballinger@brain:~/ninfer-worktree$ git rev-parse HEAD
{commit}

toddballinger@brain:~/ninfer-worktree$ git branch --show-current
{branch}

toddballinger@brain:~/ninfer-worktree$ nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
16303, {1 + (i % 5)}, {15841 - (i % 5)}

toddballinger@brain:~/ninfer-worktree$ ninja -C build -j16 ninfer ninfer-serve
[1/4] Building CUDA object src/ops/kernel_{i % 19:02d}.cu.o
[2/4] Building CXX object src/runtime/runtime_{i % 13:02d}.cpp.o
[3/4] Linking CXX static library libninfer.a
[4/4] Linking CXX executable apps/ninfer-serve

BUILD_RC=0
SSH_SESSION_REMAINS_OPEN=YES
"""

def powershell_transcript(i):
    return header(i, "POWERSHELL") + f"""
PS C:\\Users\\SyntheticUser> $Model = "local-model"
PS C:\\Users\\SyntheticUser> $Endpoint = "http://192.168.1.216:8080/v1"
PS C:\\Users\\SyntheticUser> $Context = 131072
PS C:\\Users\\SyntheticUser> $Result = [ordered]@{{
>>     Model = $Model
>>     Endpoint = $Endpoint
>>     ContextWindow = $Context
>>     MaxTokens = {8192 + (i % 4) * 8192}
>>     Reasoning = $true
>>     ToolSchemaProfile = "llamacpp"
>> }}
PS C:\\Users\\SyntheticUser> $Result | ConvertTo-Json -Depth 6
{{
  "Model": "local-model",
  "Endpoint": "http://192.168.1.216:8080/v1",
  "ContextWindow": 131072,
  "MaxTokens": {8192 + (i % 4) * 8192},
  "Reasoning": true,
  "ToolSchemaProfile": "llamacpp"
}}
"""

def json_config(i):
    return header(i, "OPENCLAW CONFIG JSON") + f"""
{{
  "agents": {{
    "defaults": {{
      "workspace": "C:\\\\Users\\\\SyntheticUser\\\\.openclaw\\\\workspace",
      "thinkingDefault": "low",
      "model": {{
        "primary": "llama-server/local-model"
      }},
      "compaction": {{
        "timeoutSeconds": 900,
        "keepRecentTokens": {12000 + (i % 5) * 1000},
        "model": "llama-server-compact/local-model"
      }},
      "timeoutSeconds": 7200
    }}
  }},
  "providers": {{
    "llama-server": {{
      "baseUrl": "http://192.168.1.216:8080/v1",
      "models": [{{
        "id": "local-model",
        "reasoning": true,
        "contextWindow": 131072,
        "maxTokens": 32768
      }}]
    }}
  }}
}}
"""

def tool_history(i):
    tool = tool_names[i % len(tool_names)]
    call_id = f"call_synth_{i:08d}"

    return header(i, "TOOL CALL HISTORY") + f"""
USER:
Inspect the current synthetic runtime state and use the appropriate tool rather
than answering from memory.

ASSISTANT TOOL CALL:
{{
  "id": "{call_id}",
  "type": "function",
  "function": {{
    "name": "{tool}",
    "arguments": "{{\\"project\\":\\"runtime-{i % 23:02d}\\",\\"detail\\":\\"full\\"}}"
  }}
}}

TOOL RESULT:
{{
  "project": "runtime-{i % 23:02d}",
  "status": "ready",
  "branch": "integration/test-{i % 17:02d}",
  "gpu_free_mib": {800 + i % 53},
  "changed_files": {i % 7},
  "qualification": "pass"
}}

ASSISTANT:
The runtime is ready. The reported GPU headroom is {800 + i % 53} MiB and the
tool result shows {i % 7} scoped changed files. I would proceed with the
controlled validation while preserving the current source and artifact identity.
"""

def python_code(i):
    return header(i, "PYTHON") + f"""
from pathlib import Path
import hashlib
import json

MODEL = Path("/models/ninfer-custom/model_{i % 11:02d}.ninfer")
EXPECTED = "{rng.getrandbits(256):064x}"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

result = {{
    "artifact": str(MODEL),
    "expected_sha256": EXPECTED,
    "actual_sha256": sha256(MODEL) if MODEL.exists() else None,
    "context": 131072,
    "kv_capacity": 131072,
    "prefill_chunk": 896,
    "draft_tokens": 3,
}}

print(json.dumps(result, indent=2))
"""

def cpp_cuda(i):
    return header(i, "C++ CUDA") + f"""
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

struct RuntimeConfig {{
    std::uint32_t max_context = 131072;
    std::uint32_t kv_capacity = 131072;
    std::uint32_t prefill_chunk = 896;
    std::uint32_t draft_tokens = 3;
    bool use_cuda_graph = false;
}};

__global__ void gather_embedding_{i % 29:02d}(
    const std::uint16_t* token_ids,
    const std::uint8_t* weights,
    float* output,
    int token_count) {{
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row < token_count && lane < 128) {{
        output[row * 128 + lane] =
            static_cast<float>(weights[token_ids[row] * 128 + lane]);
    }}
}}

void validate_runtime_{i % 31:02d}(const RuntimeConfig& cfg) {{
    if (cfg.max_context < cfg.prefill_chunk) {{
        throw std::runtime_error("invalid runtime geometry");
    }}
}}
"""

def server_logs(i):
    request = 1000 + i

    return header(i, "NINFER SERVER LOG") + f"""
[2026-09-22 12:{i % 60:02d}:01.114] [info] ninfer-serve: [req {request}] openai_chat_completions non-stream msgs={2 + i % 8} max_tokens=128 tools={i % 4} tool_choice=auto thinking=off sampler=[greedy] -> submitted
[2026-09-22 12:{i % 60:02d}:01.482] [info] ninfer-serve: [req {request}] prefill chunk=896 prompt={300 + (i * 37) % 7000} cache={i % 2048} reuse={"rolling_tool" if i % 3 == 0 else "full_reset"}
[2026-09-22 12:{i % 60:02d}:01.771] [info] ninfer-serve: [req {request}] done finish=stop_token prompt={300 + (i * 37) % 7000} gen={8 + i % 120} reasoning=0 ttft={180 + i % 900}ms prefill={1200 + (i % 800):.1f}tok/s decode={65 + (i % 70):.1f}tok/s wall={0.4 + (i % 17) / 10:.2f}s speculative=mtp {2.1 + (i % 20) / 10:.2f}tok/round ({40 + i % 55:.1f}%)
[MEMORY] weights=12734333952 runtime=2741487616 workspace=138741504 free_after_startup={800 + i % 37}MiB planned_slack={802 + i % 37}MiB
"""

def benchmark(i):
    a = 1360 + (i % 31) * 0.73
    b = a - (1 + i % 6) * 0.41
    da = 71.30 + (i % 11) * 0.02
    db = da - (i % 4) * 0.01

    return header(i, "BENCHMARK RESULT") + f"""
TEST=synthetic_ab_{i:05d}

Configuration:
  model: qwen3.8-27b
  device: RTX 5080 16GB
  context: 131072
  kv_capacity: 131072
  kv_dtype: q4-group64
  prefill_chunk: 896
  speculative_backend: mtp
  draft_tokens: 3

Baseline:
  prefill_tps: {a:.2f}
  decode_tps: {da:.2f}
  mtp_acceptance: {44.0 + (i % 9) / 10:.2f}%
  free_after_startup_mib: {40 + i % 13}

Candidate:
  prefill_tps: {b:.2f}
  decode_tps: {db:.2f}
  mtp_acceptance: {44.0 + (i % 9) / 10:.2f}%
  free_after_startup_mib: {830 + i % 17}

Verdict:
The candidate reclaimed persistent GPU memory while preserving the measured
decode path. Prefill variance should be interpreted across repeated alternating
runs rather than from one observation.
"""

def git_review(i):
    return header(i, "GIT REVIEW") + f"""
$ git status --short
 M apps/cli/options.cpp
 M docs/BENCHMARKS.md
?? tests/test_prompt_case_{i:05d}.cpp

$ git diff --stat
 apps/cli/options.cpp            | {6 + i % 20} +++++++++
 docs/BENCHMARKS.md              | {12 + i % 30} ++++++++++++++++++
 tests/test_prompt_case_{i:05d}.cpp | {25 + i % 40} +++++++++++++++++++++++++

Review notes:
1. Keep the implementation change narrowly scoped.
2. Verify the default behavior remains unchanged.
3. Separate benchmark plumbing from runtime behavior where practical.
4. Preserve source commit, binary hash, model hash and request configuration.
5. Add a regression test for the new behavior before merge.
"""

def conversation(i):
    return header(i, "FOLLOW-UP CONVERSATION") + f"""
User:
The benchmark looks promising, but before merging can we verify whether the
memory gain still holds with Vision enabled and whether tool calling continues
to work under the same runtime profile?

Assistant:
Yes. I would keep the model artifact and source fixed, enable the production
Vision profile, then issue a deterministic image request followed by a
structured function-call round trip. The acceptance criteria are successful
HTTP responses, correct tool arguments, correct tool-result consumption, and
the expected startup memory envelope.

User:
Also make sure the benchmark is representative of the actual workflow rather
than optimizing for a synthetic edge case.

Assistant:
Agreed. The workload should contain realistic repeated engineering vocabulary,
configuration, logs, code, structured tool history, and natural-language task
context. Diversity should be measured afterward as a diagnostic rather than
being the primary construction target.
"""

generators = [
    agent_planning,
    agent_planning,
    conversation,
    bash_transcript,
    bash_transcript,
    powershell_transcript,
    json_config,
    tool_history,
    tool_history,
    python_code,
    cpp_cuda,
    server_logs,
    server_logs,
    benchmark,
    git_review,
]

blocks = []

for i in range(2200):
    fn = generators[i % len(generators)]
    blocks.append(fn(i))

out.write_text(json.dumps(blocks))
