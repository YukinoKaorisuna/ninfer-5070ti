# Windows + RTX 5070 Ti build guide

This fork's upstream and the `ninfer-5080` reference are validated on **64-bit Linux
only**. Nothing about the 5070 Ti itself needs a port: it is GB203 / compute capability
12.0, so `sm_120a` is already the correct architecture, and it reports the same
16303 MiB of VRAM as the RTX 5080, so the `groupwise-int-5080` quantization recipe and
its 131072-token KV allocation fit identically. What this document covers is the
**host platform** port.

Status of the platform layer: implemented and compile-verified on MSVC 14.50 with
Windows SDK 10.0.26100. Full link + runtime is not yet verified because the toolkit
prerequisites below are not installed yet.

## 1. Prerequisites

### CUDA Toolkit >= 13.1 — required, 13.0 is rejected

`CMakeLists.txt` hard-fails below 13.1. Install a Windows toolkit from the
[CUDA archive](https://developer.nvidia.com/cuda-toolkit-archive).

- Preferred: **13.3.x** — matches the upstream validated toolkit (13.3.73).
- Acceptable: 13.4.x, which is the first branch with documented Visual Studio 2026
  support.

Since CUDA 13.1 the Windows GPU driver is no longer bundled with the toolkit, so it
must be present separately. Driver 591.86 is already installed and is sufficient.

### Visual Studio: use the VS 2022 toolset, not VS 2026

**Do not build with the installed Visual Studio 2026 Build Tools (18.x, MSVC
14.50.35717).** CUDA documentation lists MSVC 195x / VS 2026 18.x as supported, but
nvcc rejects this toolchain in practice with:

```text
nvcc fatal : Host compiler targets unsupported OS.
```

That failure is reproducible on this exact MSVC 14.50.35717 + SDK 10.0.26100.0
combination. Use the explicitly supported row instead:

```text
MSVC Version 193x | Visual Studio 2022 17.x | native x86_64 | C++14, C++17, C++20
```

`Microsoft Visual Studio\2022\Community` is already installed but is missing the C++
workload. Add it via the Visual Studio Installer:

```text
Workload:  使用 C++ 的桌面开发 (Desktop development with C++)
Components: MSVC v143 build tools, Windows 11 SDK
```

If VS 2026 must be used as a fallback, the flag `--allow-unsupported-compiler` can be
appended to `CMAKE_CUDA_FLAGS`. It is untested here and is not the default.

### FFmpeg and libcurl (prebuilt, no pkg-config)

The Windows build path takes explicit prefixes instead of pkg-config:

| Variable | Content |
|---|---|
| `NINFER_FFMPEG_ROOT` | tree containing `include/` and `lib/` for avformat, avcodec, avutil, swscale |
| `NINFER_CURL_ROOT` | tree containing `include/` and `lib/libcurl.lib` |

Any distribution that ships `include/` + `lib/` works. A prebuilt FFmpeg 6.x or 7.x
build satisfies the `libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7`
requirement. `vcpkg install ffmpeg:x64-windows curl:x64-windows` also works; the
CMake helper resolves `libcurl.lib`, `libcurld.lib` or `libcurl_a.lib`.

## 2. Build

Open a **Developer Command Prompt for VS 2022** (so `cl.exe` and Ninja are on PATH):

```bat
cmake -S . -B build-windows -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DNINFER_FFMPEG_ROOT=C:/deps/ffmpeg ^
  -DNINFER_CURL_ROOT=C:/deps/curl

cmake --build build-windows -j
```

Targets produced: `build-windows/apps/ninfer.exe` and `build-windows/apps/ninfer-serve.exe`.

No operator, target or scheduling source is modified by the platform port, so a build
that links is expected to produce numerically identical output to Linux.

## 3. Free the VRAM first

This configuration is a near-capacity fit. It is not "can the model load" but
"is there tens of MiB left", so the GPU must be clean:

```text
weights             ~13220 MiB
runtime reservation  ~2577 MiB
startup edge           44.56 MiB   (text-only, measured on the 5080)
```

Minimum free device memory before launch, derived from the reference envelope:

| Profile | Required free |
|---|---|
| text-only | ~15797 MiB |
| Vision 1792 | ~15814 MiB |
| Vision 2048 | ~15832 MiB |

Check before every launch:

```bat
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
```

On this machine the desktop plus browsers plus NVIDIA Broadcast, Wallpaper Engine,
ComfyUI and the SD WebUI launcher hold several GiB. All must be closed.

The cleanest fix on Windows is to take the desktop off the discrete GPU entirely:
the installed **Intel Core Ultra 7 265K has integrated Xe graphics that is currently
not enabled**. Connecting the display to the motherboard output and enabling the iGPU
in firmware removes the compositor's VRAM footprint from the 5070 Ti, which is what
makes the reference envelope reachable.

## 4. Launch

Text-only profile (milestone 1):

```bat
build-windows\apps\ninfer-serve.exe C:\models\qwen3_8_27b.ninfer ^
  --host 0.0.0.0 ^
  --port 8080 ^
  --model-id qwen3.8-27b ^
  --max-context 131072 ^
  --kv-capacity 131072 ^
  --prefill-chunk 896 ^
  --kv-dtype q4 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --no-cuda-graph ^
  --max-concurrency 1
```

Add `--vision --vision-max-tokens 1792` only after the text path is green and the
free-memory table above has been met.

### Fallback ladder if startup fails on reservation

Apply in order and stop as soon as it starts:

1. `--vision-max-tokens 1024` — Vision workspace 132 MiB -> 66 MiB
2. drop `--vision` entirely — recovers the full 132 MiB
3. `--kv-capacity 114688` (112K) — about 289 MiB, at roughly 18.1 MiB per 1024 tokens

## 5. Expected performance

The 5070 Ti has 70 SMs against the 5080's 84, and 896 GB/s against 960 GB/s. Decode is
bandwidth-bound and prefill is compute-bound, so against the 5080 reference
(1378.85 tok/s prefill, 71.44 tok/s decode, 44.74% MTP acceptance):

| Metric | Expect |
|---|---|
| prefill | ~1100-1180 tok/s |
| decode | ~65-68 tok/s |
| MTP acceptance | unchanged (model-determined, not device-determined) |

## 6. Known open work

- Kernel launch-threshold bands in `src/ops/linear/q4/q4_dispatch.cpp` and
  `src/ops/linear/q5/q5_dispatch.cpp` were swept on 84 SMs. They will still run on
  70 SMs; they are simply not optimal. Re-sweeping needs a working build first.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` already adapts to the
  real SM count via `cudaGetDeviceProperties`, and is the pattern the linear dispatch
  tables should follow.
- Full link and a first inference run are pending CUDA 13.1+ and the VS 2022 C++
  workload.
