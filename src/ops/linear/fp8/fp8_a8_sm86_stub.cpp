// SM86 (Ampere / RTX 30-series) has no FP8 tensor cores: ptxas rejects
// "mma with FP8 floating point type ... requires .target sm_89 or higher".
//
// The A8 route is therefore compiled out on this target and the five A8 kernel
// translation units are swapped for these stubs (same pattern the tree already
// uses for nvfp4 on Windows via nvfp4_tma_windows_stub.cpp).
//
// Nothing calls these on sm_86: the FP8 dispatcher resolves to A16 (which widens
// E4M3 weights to BF16 and runs plain mma_bf16, available since sm_80).  The
// definitions exist only to keep the link step intact.
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void fp8_a8_requires_sm89() {
    throw std::runtime_error("fp8 A8 tensor-core route requires sm_89 or newer");
}

} // namespace

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    fp8_a8_requires_sm89();
}

void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    fp8_a8_requires_sm89();
}

void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    fp8_a8_requires_sm89();
}

void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    fp8_a8_requires_sm89();
}

void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    fp8_a8_requires_sm89();
}

void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    fp8_a8_requires_sm89();
}

} // namespace ninfer::ops::detail
