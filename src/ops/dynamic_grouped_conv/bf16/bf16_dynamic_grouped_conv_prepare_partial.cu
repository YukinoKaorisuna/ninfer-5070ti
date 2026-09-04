#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden          = 5120;
constexpr int kWidth           = 8;
constexpr int kCoefficientRows = 1280;
constexpr int kBlockRows       = 16;
constexpr int kBlockK          = 64;
constexpr int kStages          = 2;
constexpr int kRowTiles        = kCoefficientRows / kBlockRows;
static_assert(kCoefficientRows % kBlockRows == 0);
static_assert(kHidden % kBlockK == 0);

__device__ __forceinline__ int shared_col(int row, int col) { return gemm_swz64(row, col); }

template <int BatchSize, int SplitK>
__global__ __launch_bounds__(BatchSize * 32, 1) void dynamic_grouped_conv_prepare_partial_kernel(
    const __nv_bfloat16* __restrict__ normalized,
    const __nv_bfloat16* __restrict__ kernel_projection, float* __restrict__ projection_partial) {
    static_assert(BatchSize >= 1 && BatchSize <= 8);
    static_assert(SplitK == 4 || SplitK == 8);
    constexpr int kTokens         = BatchSize * kWidth;
    constexpr int kThreads        = BatchSize * 32;
    constexpr int kKTiles         = kHidden / kBlockK;
    constexpr int kTilesPerSplit  = kKTiles / SplitK;
    constexpr int kMmaKFragments  = kBlockK / 16;
    constexpr int kWeightVecs     = kBlockRows * (kBlockK / 8);
    constexpr int kActivationVecs = kTokens * (kBlockK / 8);
    static_assert(kKTiles % SplitK == 0);

    __shared__ __align__(16) __nv_bfloat16 weight_shared[kStages][kBlockRows * kBlockK];
    __shared__ __align__(16) __nv_bfloat16 activation_shared[kStages][kTokens * kBlockK];

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int row_tile  = static_cast<int>(blockIdx.x);
    const int split     = static_cast<int>(blockIdx.z);
    const int row_begin = row_tile * kBlockRows;
    const int kt_begin  = split * kTilesPerSplit;

    float accumulator[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k_begin = kt * kBlockK;
        for (int item = tid; item < kWeightVecs; item += kThreads) {
            const int row   = item / (kBlockK / 8);
            const int k_vec = item - row * (kBlockK / 8);
            const int kk    = k_vec * 8;
            cp_async<16, Cache::cg>(
                &weight_shared[stage][row * kBlockK + shared_col(row, kk)],
                &kernel_projection[static_cast<std::int64_t>(row_begin + row) * kHidden + k_begin +
                                   kk]);
        }

        for (int item = tid; item < kActivationVecs; item += kThreads) {
            const int token = item / (kBlockK / 8);
            const int k_vec = item - token * (kBlockK / 8);
            const int kk    = k_vec * 8;
            cp_async<16, Cache::cg>(
                &activation_shared[stage][token * kBlockK + shared_col(token, kk)],
                &normalized[static_cast<std::int64_t>(token) * kHidden + k_begin + kk]);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        stage_inputs(stage, kt_begin + stage);
        cp_commit();
    }

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int tile = 0; tile < kTilesPerSplit; ++tile) {
        const int stage = tile & (kStages - 1);
        if (tile + kStages <= kTilesPerSplit) {
            cp_wait<kStages - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kMmaKFragments; ++ki) {
            unsigned a_frag[4];
            unsigned b_frag[2];
            const int a_row = a_row_offset;
            const int a_col = ki * 16 + a_col_offset;
            const int b_row = warp * kWidth + b_inner_row;
            const int b_col = ki * 16 + b_k_offset;
            ldmatrix_x4(
                a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                smem_addr(&weight_shared[stage][a_row * kBlockK + shared_col(a_row, a_col)]));
            ldmatrix_x2(
                b_frag[0], b_frag[1],
                smem_addr(&activation_shared[stage][b_row * kBlockK + shared_col(b_row, b_col)]));
            mma_bf16(accumulator[0], accumulator[1], accumulator[2], accumulator[3], a_frag[0],
                     a_frag[1], a_frag[2], a_frag[3], b_frag[0], b_frag[1]);
        }

        __syncthreads();
        const int next = tile + kStages;
        if (next < kTilesPerSplit) { stage_inputs(stage, kt_begin + next); }
        cp_commit();
    }

    const int row0           = row_begin + gid;
    const int row1           = row0 + 8;
    const int token0         = warp * kWidth + 2 * lid;
    const int token1         = token0 + 1;
    const auto store_partial = [&](int row, int token, float value) {
        const std::int64_t offset =
            (static_cast<std::int64_t>(split) * kTokens + token) * kCoefficientRows + row;
        projection_partial[offset] = value;
    };
    store_partial(row0, token0, accumulator[0]);
    store_partial(row0, token1, accumulator[1]);
    store_partial(row1, token0, accumulator[2]);
    store_partial(row1, token1, accumulator[3]);
}

template <int BatchSize, int SplitK>
void launch(const Tensor& normalized, const Weight& kernel_projection_weight,
            float* projection_partial, cudaStream_t stream) {
    static_assert(BatchSize >= 1 && BatchSize <= 8);
    const dim3 grid(kRowTiles, 1u, SplitK);
    dynamic_grouped_conv_prepare_partial_kernel<BatchSize, SplitK>
        <<<grid, BatchSize * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<const __nv_bfloat16*>(kernel_projection_weight.qdata), projection_partial);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_dynamic_grouped_conv_prepare_partial_launch(std::int32_t split_k,
                                                      const Tensor& normalized,
                                                      const Weight& kernel_projection_weight,
                                                      float* projection_partial,
                                                      cudaStream_t stream) {
    switch (split_k) {
    case 4:
        if (normalized.ne[2] == 7) {
            launch<7, 4>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        }
        if (normalized.ne[2] == 8) {
            launch<8, 4>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        }
        break;
    case 8:
        switch (normalized.ne[2]) {
        case 1:
            launch<1, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        case 2:
            launch<2, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        case 3:
            launch<3, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        case 4:
            launch<4, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        case 5:
            launch<5, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        case 6:
            launch<6, 8>(normalized, kernel_projection_weight, projection_partial, stream);
            return;
        default:
            break;
        }
        break;
    default:
        break;
    }
    throw std::invalid_argument("dynamic grouped conv prepare: unsupported production route");
}

} // namespace ninfer::ops::detail
