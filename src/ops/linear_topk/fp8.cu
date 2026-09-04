#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Fp8KSplitTopKOutput {
    std::uint64_t* partial_keys;
    std::int32_t producer_groups;
    std::int32_t valid_rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t column,
                                          float value) const {
        const std::int32_t group = row / kLinearTopK;
        const std::int32_t rank  = row % kLinearTopK;
        const std::int64_t offset =
            (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
        partial_keys[offset] = row < valid_rows ? score_id_order_key(value, row) : 0;
    }
};

void launch_ksplit_t7(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                      const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Geometry        = Fp8VocabularyGeometry;
    using Schedule        = typename Fp8VocabularyA16SmallTMmaProductionSchedule<7>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8KSplitTopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                                     workspace.producer_groups, valid_rows};
    fp8_a16_small_t_mma_kernel<Geometry, 7, Schedule, Fp8KSplitTopKOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data),
                                                     static_cast<const std::uint8_t*>(head.qdata),
                                                     static_cast<const __nv_bfloat16*>(head.scales),
                                                     output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (workspace.rows_per_producer == kLinearTopKKSplitRowsPerGroup) {
        launch_ksplit_t7(hidden, head, valid_rows, workspace, stream);
    } else if (hidden.ne[1] <= 21) {
        linear_topk_fp8_grouped_ksplit_launch(hidden, head, valid_rows, workspace, stream);
    } else {
        linear_topk_fp8_m64_launch(hidden, head, valid_rows, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
