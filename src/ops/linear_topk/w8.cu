#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct W8KSplitTopKOutput {
    std::uint64_t* partial_keys;
    std::int32_t producer_groups;
    std::int32_t valid_rows;

    template <int ActiveColumns>
    __device__ __forceinline__ void store(std::int32_t row,
                                          const float (&values)[ActiveColumns]) const {
        const std::int32_t group = row / kLinearTopK;
        const std::int32_t rank  = row % kLinearTopK;
#pragma unroll
        for (int column = 0; column < ActiveColumns; ++column) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
            partial_keys[offset] = row < valid_rows ? score_id_order_key(values[column], row) : 0;
        }
    }
};

void launch_ksplit_t7(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                      const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Geometry        = W8VocabularyProjectionGeometry;
    using Schedule        = typename W8LinearSmallTProductionSchedule<Geometry, 7>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const W8ContiguousOutput unused{nullptr, Geometry::kOutputRows};
    const W8KSplitTopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                                    workspace.producer_groups, valid_rows};
    w8_small_t_mma_kernel<Geometry, 7, Schedule, W8ContiguousOutput, W8KSplitTopKOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data),
                                                     static_cast<const std::uint8_t*>(head.qdata),
                                                     static_cast<const std::uint8_t*>(head.scales),
                                                     unused, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_topk_w8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (workspace.rows_per_producer == kLinearTopKKSplitRowsPerGroup) {
        launch_ksplit_t7(hidden, head, valid_rows, workspace, stream);
    } else if (hidden.ne[1] <= 21) {
        linear_topk_w8_grouped_ksplit_launch(hidden, head, valid_rows, workspace, stream);
    } else {
        linear_topk_w8_m64_launch(hidden, head, valid_rows, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
