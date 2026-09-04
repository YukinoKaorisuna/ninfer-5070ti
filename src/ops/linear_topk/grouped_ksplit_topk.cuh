#pragma once

#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cub/warp/warp_merge_sort.cuh>

#include <cstdint>

namespace ninfer::ops::detail {

using LinearTopKPairSort = cub::WarpMergeSort<std::uint64_t, 1, 32>;

template <int ActiveColumns, int Warps>
struct GroupedKSplitTopKStorage {
    std::uint64_t top_keys[ActiveColumns][kLinearTopK];
    typename LinearTopKPairSort::TempStorage sort[Warps];
};

template <int ActiveColumns, int Warps>
__device__ __forceinline__ void
grouped_ksplit_topk_initialize(GroupedKSplitTopKStorage<ActiveColumns, Warps>& storage) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int p = tid; p < ActiveColumns * kLinearTopK; p += blockDim.x) {
        storage.top_keys[p / kLinearTopK][p % kLinearTopK] = 0;
    }
    __syncthreads();
}

template <int ActiveColumns, int TileColumns, int Warps, bool MappedIds>
__device__ __forceinline__ void
grouped_ksplit_topk_consume(const float* scores,
                            GroupedKSplitTopKStorage<ActiveColumns, Warps>& storage,
                            std::int32_t row_begin, std::int32_t valid_rows,
                            const std::int32_t* row_to_global_ids = nullptr) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int column = warp; column < ActiveColumns; column += Warps) {
        std::uint64_t key[1] = {0};
        if (lane < kLinearTopK) {
            const int row = row_begin + lane;
            if (row < valid_rows) {
                const int global_id = MappedIds ? row_to_global_ids[row] : row;
                key[0] = score_id_order_key(scores[lane * TileColumns + column], global_id);
            }
        } else {
            key[0] = storage.top_keys[column][lane - kLinearTopK];
        }
        LinearTopKPairSort(storage.sort[warp]).Sort(key, ScoreIdOrderGreater{});
        if (lane < kLinearTopK) { storage.top_keys[column][lane] = key[0]; }
    }
    __syncthreads();
}

template <int ActiveColumns, int Warps>
__device__ __forceinline__ void
grouped_ksplit_topk_publish(const GroupedKSplitTopKStorage<ActiveColumns, Warps>& storage,
                            std::uint64_t* partial_keys, std::int32_t producer_groups) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int group = static_cast<int>(blockIdx.x);
    for (int p = tid; p < ActiveColumns * kLinearTopK; p += blockDim.x) {
        const int column = p / kLinearTopK;
        const int rank   = p - column * kLinearTopK;
        const std::int64_t destination =
            (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
        partial_keys[destination] = storage.top_keys[column][rank];
    }
}

} // namespace ninfer::ops::detail
