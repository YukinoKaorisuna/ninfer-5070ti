#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cub/warp/warp_merge_sort.cuh>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlockRows       = 64;
constexpr int kBlockK          = 128;
constexpr int kRowsPerProducer = kLinearTopKMRowsPerGroup;
constexpr int kRowTiles        = kRowsPerProducer / kBlockRows;
constexpr int kKTiles          = kLinearTopKHidden / kBlockK;

using M64WarpSort = cub::WarpMergeSort<std::uint64_t, 3, 32>;

template <int ActiveColumns>
struct M64Schedule {
    // Exact 8-column partitions remove padded MMA work at T=28/35/42. At T=49/56 the measured
    // winner is two wider column partitions: retaining two CTAs per SM outweighs the padded tail.
    static constexpr int kColumnWarps  = ActiveColumns == 28   ? 4
                                         : ActiveColumns == 35 ? 5
                                         : ActiveColumns == 42 ? 6
                                                               : 2;
    static constexpr int kWarps        = ActiveColumns == 28   ? 16
                                         : ActiveColumns == 35 ? 20
                                         : ActiveColumns == 42 ? 24
                                                               : 8;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kBlockColumns = ActiveColumns == 35   ? 40
                                         : ActiveColumns <= 32 ? 32
                                         : ActiveColumns <= 48 ? 48
                                                               : 64;
};

template <int ActiveColumns>
struct alignas(16) Fp8M64MainloopStorage {
    static constexpr int kBlockColumns = M64Schedule<ActiveColumns>::kBlockColumns;
    __nv_bfloat16 weights[kBlockRows][kBlockK];
    __nv_bfloat16 activations[kBlockColumns][kBlockK];
    std::uint8_t codes[kBlockRows][kBlockK];
};

template <int ActiveColumns>
struct Fp8M64ReductionStorage {
    float scores[ActiveColumns][kBlockRows];
    typename M64WarpSort::TempStorage sort[M64Schedule<ActiveColumns>::kWarps];
};

template <int ActiveColumns>
union alignas(16) Fp8M64ReusableStorage {
    Fp8M64MainloopStorage<ActiveColumns> mainloop;
    Fp8M64ReductionStorage<ActiveColumns> reduction;
};

__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

template <int ActiveColumns>
__global__
__launch_bounds__(M64Schedule<ActiveColumns>::kThreads, 2) void fp8_m64_linear_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, std::int32_t valid_rows,
    std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups) {
    constexpr int kWarps        = M64Schedule<ActiveColumns>::kWarps;
    constexpr int kThreads      = M64Schedule<ActiveColumns>::kThreads;
    constexpr int kColumnWarps  = M64Schedule<ActiveColumns>::kColumnWarps;
    constexpr int kBlockColumns = Fp8M64MainloopStorage<ActiveColumns>::kBlockColumns;
    constexpr int kWarpColumns  = kBlockColumns / kColumnWarps;
    constexpr int kTokenMmas    = kWarpColumns / 8;
    constexpr int kTopBytes     = ActiveColumns * kLinearTopK * sizeof(std::uint64_t);
    static_assert(kRowsPerProducer == 128 && kRowTiles == 2);
    static_assert(ActiveColumns == 28 || ActiveColumns == 35 || ActiveColumns == 42 ||
                  ActiveColumns == 49 || ActiveColumns == 56);
    static_assert(sizeof(Fp8M64ReusableStorage<ActiveColumns>) + kTopBytes <= 48 * 1024);

    __shared__ Fp8M64ReusableStorage<ActiveColumns> reusable;
    __shared__ std::uint64_t top_keys[ActiveColumns][kLinearTopK];
    auto& mainloop = reusable.mainloop;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_row = warp / kColumnWarps;
    const int warp_col = warp - warp_row * kColumnWarps;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_rowoff = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_coloff = (a_matrix >> 1) << 3;
    const int b_row    = lane & 7;
    const int b_coloff = ((lane >> 3) & 1) << 3;

    for (int p = tid; p < ActiveColumns * kLinearTopK; p += kThreads) {
        top_keys[p / kLinearTopK][p % kLinearTopK] = 0;
    }
    __syncthreads();

    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        const int row_begin =
            static_cast<int>(blockIdx.x) * kRowsPerProducer + row_tile * kBlockRows;
        {
            float accumulators[kTokenMmas][4] = {};

            const auto stage_activation = [&](int k_tile) {
                const int k_begin    = k_tile * kBlockK;
                constexpr int kItems = kBlockColumns * (kBlockK / 8);
                for (int item = tid; item < kItems; item += kThreads) {
                    const int column  = item / (kBlockK / 8);
                    const int k8      = item - column * (kBlockK / 8);
                    auto* destination = &mainloop.activations[column][swizzle_128(column, k8 * 8)];
                    if (column < ActiveColumns) {
                        cp_async<16, Cache::ca>(destination, hidden +
                                                                 static_cast<std::int64_t>(column) *
                                                                     kLinearTopKHidden +
                                                                 k_begin + k8 * 8);
                    } else {
                        cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8, 0);
                    }
                }
            };

            const auto stage_codes = [&](int k_tile) {
                const int k_begin     = k_tile * kBlockK;
                constexpr int kChunks = kBlockRows * (kBlockK / 16);
                for (int item = tid; item < kChunks; item += kThreads) {
                    const int local_row = item / (kBlockK / 16);
                    const int chunk     = item - local_row * (kBlockK / 16);
                    cp_async<16, Cache::cg>(&mainloop.codes[local_row][chunk * 16],
                                            weight_codes +
                                                static_cast<std::int64_t>(row_begin + local_row) *
                                                    kLinearTopKHidden +
                                                k_begin + chunk * 16);
                }
            };

            const auto decode_codes = [&]() {
                constexpr int kChunksPerRow = kBlockK / 8;
                for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                    const int row      = item / kChunksPerRow;
                    const int chunk    = item - row * kChunksPerRow;
                    const int col      = chunk * 8;
                    const uint2 packed = *reinterpret_cast<const uint2*>(&mainloop.codes[row][col]);
                    uint4 decoded;
                    decoded.x = fp8_e4m3x2_to_bf16x2_bits(packed.x & 0xffffu);
                    decoded.y = fp8_e4m3x2_to_bf16x2_bits(packed.x >> 16);
                    decoded.z = fp8_e4m3x2_to_bf16x2_bits(packed.y & 0xffffu);
                    decoded.w = fp8_e4m3x2_to_bf16x2_bits(packed.y >> 16);
                    store_vec(&mainloop.weights[row][swizzle_128(row, col)], decoded);
                }
            };

            stage_activation(0);
            stage_codes(0);
            cp_commit();

#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
                cp_wait<0>();
                __syncthreads();
                decode_codes();
                __syncthreads();

                const int next = k_tile + 1;
                if (next < kKTiles) {
                    stage_codes(next);
                    cp_commit();
                }

                const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                                unsigned(&b)[kTokenMmas][2]) {
                    const int weight_row = warp_row * 16 + a_rowoff;
                    const int weight_col = k_step * 16 + a_coloff;
                    ldmatrix_x4(
                        a[0], a[1], a[2], a[3],
                        smem_addr(
                            &mainloop.weights[weight_row][swizzle_128(weight_row, weight_col)]));
#pragma unroll
                    for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                        const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                        const int activation_col = k_step * 16 + b_coloff;
                        ldmatrix_x2(b[token_mma][0], b[token_mma][1],
                                    smem_addr(&mainloop.activations[activation_row][swizzle_128(
                                        activation_row, activation_col)]));
                    }
                };

                unsigned a_fragments[2][4];
                unsigned b_fragments[2][kTokenMmas][2];
                load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
                for (int k_step = 0; k_step < kBlockK / 16; ++k_step) {
                    const int slot = k_step & 1;
                    if (k_step + 1 < kBlockK / 16) {
                        load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
                    }
#pragma unroll
                    for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                        mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                                 accumulators[token_mma][2], accumulators[token_mma][3],
                                 a_fragments[slot][0], a_fragments[slot][1], a_fragments[slot][2],
                                 a_fragments[slot][3], b_fragments[slot][token_mma][0],
                                 b_fragments[slot][token_mma][1]);
                    }
                }

                if (next < kKTiles) {
                    __syncthreads();
                    stage_activation(next);
                    cp_commit();
                }
            }

            __syncthreads();
            auto& scores         = reusable.reduction.scores;
            const int local_row0 = warp_row * 16 + gid;
            const int local_row1 = local_row0 + 8;
            const float scale0   = __bfloat162float(row_scales[row_begin + local_row0]);
            const float scale1   = __bfloat162float(row_scales[row_begin + local_row1]);
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const int column0 = warp_col * kWarpColumns + token_mma * 8 + 2 * lid;
                if (column0 < ActiveColumns) {
                    scores[column0][local_row0] = accumulators[token_mma][0] * scale0;
                    scores[column0][local_row1] = accumulators[token_mma][2] * scale1;
                }
                if (column0 + 1 < ActiveColumns) {
                    scores[column0 + 1][local_row0] = accumulators[token_mma][1] * scale0;
                    scores[column0 + 1][local_row1] = accumulators[token_mma][3] * scale1;
                }
            }
        }
        __syncthreads();

        const int reducer_warp = tid >> 5;
        const int reducer_lane = tid & 31;
        for (int column = reducer_warp; column < ActiveColumns; column += kWarps) {
            std::uint64_t keys[3] = {0, 0, 0};
#pragma unroll
            for (int item = 0; item < 2; ++item) {
                const int local_row = reducer_lane * 2 + item;
                const int row       = row_begin + local_row;
                if (row < valid_rows) {
                    keys[item] =
                        score_id_order_key(reusable.reduction.scores[column][local_row], row);
                }
            }
            if (reducer_lane < kLinearTopK) { keys[2] = top_keys[column][reducer_lane]; }
            M64WarpSort(reusable.reduction.sort[reducer_warp]).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
            for (int item = 0; item < 3; ++item) {
                const int rank = reducer_lane * 3 + item;
                if (rank < kLinearTopK) { top_keys[column][rank] = keys[item]; }
            }
        }
        __syncthreads();
    }

    for (int p = tid; p < ActiveColumns * kLinearTopK; p += kThreads) {
        const int column = p / kLinearTopK;
        const int rank   = p - column * kLinearTopK;
        const std::int64_t destination =
            (static_cast<std::int64_t>(column) * producer_groups + blockIdx.x) * kLinearTopK + rank;
        partial_keys[destination] = top_keys[column][rank];
    }
}

template <int ActiveColumns>
void launch_exact(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                  const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    fp8_m64_linear_topk_kernel<ActiveColumns>
        <<<workspace.producer_groups, M64Schedule<ActiveColumns>::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const __nv_bfloat16*>(head.scales), valid_rows,
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_topk_fp8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                                const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (hidden.ne[1] == 28) {
        launch_exact<28>(hidden, head, valid_rows, workspace, stream);
    } else if (hidden.ne[1] == 35) {
        launch_exact<35>(hidden, head, valid_rows, workspace, stream);
    } else if (hidden.ne[1] == 42) {
        launch_exact<42>(hidden, head, valid_rows, workspace, stream);
    } else if (hidden.ne[1] == 49) {
        launch_exact<49>(hidden, head, valid_rows, workspace, stream);
    } else {
        launch_exact<56>(hidden, head, valid_rows, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
