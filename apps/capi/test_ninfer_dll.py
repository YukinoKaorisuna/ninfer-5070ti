"""PoC: load ninfer_capi.dll into a Python process and drive the engine in-process.

This verifies the two things a ComfyUI integration depends on:

  1. ctypes can load and call the C ABI at all.
  2. ninfer and PyTorch can coexist in ONE process / ONE CUDA context — i.e. we can
     build a CUDA context with torch first, then load ninfer into the same process,
     and neither blows up.

Run (from the repo root, after building the ninfer_capi target):

    python apps/capi/test_ninfer_dll.py ^
        --dll build-windows/apps/ninfer_capi.dll ^
        --model models/qwen3_8_27b.ninfer

Optional:
    --mtp 3            enable MTP speculative decoding with 3 draft tokens
    --thinking         keep the reasoning pass (default: OFF, which is the fast path)
    --max-context 4096 --max-tokens 128

If step [3/4] or [4/4] fails, read the printed ninfer_last_error() string.
"""

import argparse
import ctypes
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, help="path to ninfer_capi.dll")
    parser.add_argument("--model", required=True, help="path to the .ninfer artifact")
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--mtp", type=int, default=0,
                        help="0 = off, else MTP draft tokens (1..5)")
    parser.add_argument("--thinking", action="store_true",
                        help="enable the reasoning pass (default: off = fast path)")
    parser.add_argument("--system", default="You are a senior prompt engineer.",
                        help="system prompt")
    parser.add_argument("--prompt",
                        default="Rewrite this as a rich, detailed image prompt: a cat on a roof.")
    return parser.parse_args()


def bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restypes so ctypes marshals correctly."""
    lib.ninfer_create.restype = ctypes.c_void_p
    lib.ninfer_create.argtypes = [
        ctypes.c_char_p,  # artifact_path (UTF-8)
        ctypes.c_int,     # device
        ctypes.c_int,     # max_context
        ctypes.c_int,     # speculative_mtp
        ctypes.c_int,     # kv_tokens (0 = follow max_context)
    ]

    lib.ninfer_generate.restype = ctypes.c_int
    lib.ninfer_generate.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_char_p,  # system text (UTF-8, may be NULL)
        ctypes.c_char_p,  # user text (UTF-8)
        ctypes.c_int,     # max_new_tokens
        ctypes.c_int,     # enable_thinking
        ctypes.c_char_p,  # out buffer
        ctypes.c_int,     # out capacity (bytes, incl. NUL)
    ]

    lib.ninfer_destroy.restype = None
    lib.ninfer_destroy.argtypes = [ctypes.c_void_p]

    lib.ninfer_last_error.restype = ctypes.c_char_p
    lib.ninfer_last_error.argtypes = []


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # [1/4] Bring up torch + its CUDA context FIRST, simulating ComfyUI.
    #       If ninfer then works, the two share a process cleanly.
    # ------------------------------------------------------------------
    print("[1/4] initialising torch CUDA (simulating ComfyUI's existing context)...")
    try:
        import torch

        if torch.cuda.is_available():
            probe = torch.zeros(1024, device="cuda")
            name = torch.cuda.get_device_name(0)
            print(f"      torch OK  device={name}  "
                  f"allocated={torch.cuda.memory_allocated() / 2**20:.1f} MiB")
            del probe
        else:
            print("      torch present but CUDA unavailable (continuing anyway)")
    except ImportError:
        print("      torch not installed in this interpreter (continuing anyway)")

    # ------------------------------------------------------------------
    # [2/4] Load the DLL INTO this process.
    # ------------------------------------------------------------------
    print(f"[2/4] loading {args.dll} via ctypes ...")
    try:
        lib = ctypes.CDLL(args.dll)
    except OSError as error:
        print(f"      FAILED to load: {error}")
        return 1
    bind_signatures(lib)
    print("      loaded")

    # ------------------------------------------------------------------
    # [3/4] Load the artifact (slow, happens once).
    # ------------------------------------------------------------------
    print(f"[3/4] creating engine (max_context={args.max_context}, mtp={args.mtp}) ...")
    started = time.perf_counter()
    handle = lib.ninfer_create(args.model.encode("utf-8"), 0,
                               args.max_context, args.mtp, 0)
    if not handle:
        print("      FAILED:", lib.ninfer_last_error().decode("utf-8", "replace"))
        return 1
    print(f"      engine ready in {time.perf_counter() - started:.2f}s")

    # ------------------------------------------------------------------
    # [4/4] Generate.
    # ------------------------------------------------------------------
    print("[4/4] generating ...")
    buffer = ctypes.create_string_buffer(1 << 16)
    started = time.perf_counter()
    written = lib.ninfer_generate(
        handle,
        args.system.encode("utf-8"),
        args.prompt.encode("utf-8"),
        args.max_tokens,
        1 if args.thinking else 0,
        buffer,
        len(buffer),
    )
    elapsed = time.perf_counter() - started

    if written < 0:
        print("      FAILED:", lib.ninfer_last_error().decode("utf-8", "replace"))
        lib.ninfer_destroy(handle)
        return 1

    if elapsed > 0:
        print(f"      {written} bytes in {elapsed:.2f}s ({written / elapsed:.1f} bytes/s)")
    else:
        print(f"      {written} bytes")
    print("      --- output ---")
    print("      " + buffer.value.decode("utf-8", "replace"))

    # Release every GPU allocation so a diffusion pass can take the card.
    lib.ninfer_destroy(handle)
    print("engine destroyed — GPU memory released")

    # Confirm torch is still healthy after ninfer released its memory: this is the
    # observable proof that the two coexist in one process.
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("torch still healthy, "
                  f"allocated={torch.cuda.memory_allocated() / 2**20:.1f} MiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
