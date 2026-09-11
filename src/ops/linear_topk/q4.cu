#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Q4KSplitTopKOutput {
    std::uint64_t* partial_keys;
    const std::int32_t* row_to_global_ids;
    std::int32_t producer_groups;
    std::int32_t valid_rows;

    template <int ActiveColumns>
    __device__ __forceinline__ void store(std::int32_t row0, std::int32_t column0,
                                          float4 values) const {
        const std::int32_t group = row0 / kLinearTopK;
        const std::int32_t rank0 = row0 % kLinearTopK;
        const auto put           = [&](std::int32_t row, std::int32_t rank, std::int32_t column,
                             float value) {
            if (column >= ActiveColumns) { return; }
            const std::int64_t offset =
                (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
            if (row >= valid_rows) {
                partial_keys[offset] = 0;
                return;
            }
            const std::int32_t global_id =
                row_to_global_ids != nullptr ? row_to_global_ids[row] : row;
            partial_keys[offset] = score_id_order_key(value, global_id);
        };
        put(row0, rank0, column0, values.x);
        put(row0, rank0, column0 + 1, values.y);
        put(row0 + 8, rank0 + 8, column0, values.z);
        put(row0 + 8, rank0 + 8, column0 + 1, values.w);
    }
};

template <int OutputRows, int ActiveColumns>
void launch_ksplit_rows(const Tensor& hidden, const Weight& head,
                        const Tensor* row_to_global_ids, std::int32_t valid_rows,
                        const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Geometry             = Q4LinearGeometry<OutputRows, kLinearTopKHidden>;
    using Schedule             = Q4DraftSmallTSchedule;
    constexpr int kTileColumns = ((ActiveColumns + 7) / 8) * 8;
    constexpr int kBlocks      = Geometry::kOutputRows / Schedule::kRowsPerCta;

    const Q4KSplitTopKOutput output{
        static_cast<std::uint64_t*>(workspace.partial_keys.data),
        row_to_global_ids != nullptr
            ? static_cast<const std::int32_t*>(row_to_global_ids->data)
            : nullptr,
        workspace.producer_groups,
        valid_rows};

    q4_small_t_mma_kernel<Geometry, kTileColumns, ActiveColumns, Q4KSplitTopKOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data),
                                                     static_cast<const std::uint8_t*>(head.qdata),
                                                     static_cast<const std::uint8_t*>(head.scales),
                                                     nullptr, output);
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveColumns>
void launch_ksplit(const Tensor& hidden, const Weight& head,
                   const Tensor* row_to_global_ids, std::int32_t valid_rows,
                   const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (head.n == kLinearTopKFullRows) {
        launch_ksplit_rows<kLinearTopKFullRows, ActiveColumns>(
            hidden, head, row_to_global_ids, valid_rows, workspace, stream);
    } else {
        launch_ksplit_rows<kLinearTopKOptimizedRows, ActiveColumns>(
            hidden, head, row_to_global_ids, valid_rows, workspace, stream);
    }
}

} // namespace

void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                            const Tensor* row_to_global_ids, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (workspace.rows_per_producer == kLinearTopKKSplitRowsPerGroup) {
        if (hidden.ne[1] == 7) {
            launch_ksplit<7>(
                hidden, head, row_to_global_ids, valid_rows, workspace, stream);
        } else {
            launch_ksplit<14>(
                hidden, head, row_to_global_ids, valid_rows, workspace, stream);
        }
        return;
    }

    linear_topk_q4_m64_launch(
        hidden, head, row_to_global_ids, valid_rows, workspace, stream);
}

} // namespace ninfer::ops::detail
