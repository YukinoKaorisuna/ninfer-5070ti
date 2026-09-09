// ninfer::ops - experimental Q2KV cached small-T GQA launcher.
//
// Kept in a separate CUDA translation unit intentionally: Q2 kernel iteration
// must not force recompilation of the full BF16/I8/Q4 decode launcher.

#include "ops/launcher/gqa_attention.h"

#include "core/device.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_q2.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool Masked>
void launch_q2_partial(
    const Tensor& q,
    const Tensor& pos,
    float scale,
    PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation,
    std::int32_t logical_capacity,
    std::int32_t implementation_window,
    std::int32_t splits,
    Tensor& partial_acc,
    Tensor& partial_m,
    Tensor& partial_l,
    cudaStream_t stream) {

    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;

    const GqaCachedInput input{};

    auto launch =
        [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
            const dim3 grid(
                Geometry::KVHeads,
                splits,
                invocation.batch_size);

            constexpr std::size_t kDynamicBytes =
                DynamicArena
                    ? static_cast<std::size_t>(
                          4 * KeyBlock * kGqaHeadDim)
                    : 0u;

            if constexpr (DynamicArena) {
                static const cudaError_t attr =
                    cudaFuncSetAttribute(
                        gqa_attention_decode_q2_tiled_kernel<
                            Geometry,
                            TokenTile,
                            WarpsPerCta,
                            MinBlocksPerSm,
                            KeyBlock,
                            DynamicArena,
                            false,
                              Masked,
                              GqaCachedInput>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(kDynamicBytes));

                CUDA_CHECK(attr);
            }

            gqa_attention_decode_q2_tiled_kernel<
                Geometry,
                TokenTile,
                WarpsPerCta,
                MinBlocksPerSm,
                KeyBlock,
                DynamicArena,
                false,
                              Masked,
                              GqaCachedInput>
                <<<grid,
                   WarpsPerCta * 32,
                   kDynamicBytes,
                   stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    input,
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data),
                    static_cast<const std::int32_t*>(
                        cache.block_tables.data),
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(
                              invocation.valid_columns->data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(
                              invocation.table_rows->data),
                    cache.block_tables.ne[0],
                    invocation.full_width,
                    invocation.column_begin,
                    logical_capacity,
                    scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data),
                    static_cast<float*>(partial_l.data));
        };

    // Preserve the existing Q4 launch geometry for now.
    if constexpr (TokenTile == 6) {
        if constexpr (Geometry::GroupSize == 4) {
            if (implementation_window <= 2054) {
                launch.template operator()<16, 1, 32, false>();
            } else if (implementation_window <= 8198) {
                launch.template operator()<16, 1, 64, true>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            if (implementation_window <= 2054) {
                launch.template operator()<12, 1, 32, false>();
            } else if (implementation_window <= 8198) {
                launch.template operator()<12, 1, 64, true>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else if constexpr (Geometry::GroupSize == 4) {
            if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }

    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile>
void launch_q2_partial_dispatch(
    const Tensor& q,
    const Tensor& pos,
    float scale,
    PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation,
    std::int32_t logical_capacity,
    std::int32_t implementation_window,
    std::int32_t splits,
    Tensor& partial_acc,
    Tensor& partial_m,
    Tensor& partial_l,
    cudaStream_t stream) {

    if (invocation.valid_columns != nullptr) {
        launch_q2_partial<Geometry, TokenTile, true>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
    } else {
        launch_q2_partial<Geometry, TokenTile, false>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
    }
}

template <typename Geometry>
void dispatch_tokens(
    const Tensor& q,
    const Tensor& pos,
    float scale,
    PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation,
    std::int32_t logical_capacity,
    std::int32_t implementation_window,
    std::int32_t splits,
    Tensor& partial_acc,
    Tensor& partial_m,
    Tensor& partial_l,
    cudaStream_t stream) {

    switch (q.ne[2]) {
    case 1:
        launch_q2_partial_dispatch<Geometry, 1>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    case 2:
        launch_q2_partial_dispatch<Geometry, 2>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    case 3:
        launch_q2_partial_dispatch<Geometry, 3>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    case 4:
        launch_q2_partial_dispatch<Geometry, 4>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    case 5:
        launch_q2_partial_dispatch<Geometry, 5>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    case 6:
        launch_q2_partial_dispatch<Geometry, 6>(
            q, pos, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;

    default:
        throw std::invalid_argument(
            "Q2KV cached small-T requires T=1..6");
    }
}

} // namespace

void gqa_attention_q2_cached_small_t_launch(
    const Tensor& q,
    const Tensor& positions,
    float scale,
    PagedKVBatchLayerView cache,
    const GqaSmallTInvocation& invocation,
    std::int32_t logical_capacity,
    std::int32_t implementation_window,
    std::int32_t splits,
    Tensor& partial_acc,
    Tensor& partial_m,
    Tensor& partial_l,
    cudaStream_t stream) {

    if (q.ne[1] == Gqa27Geometry::QHeads) {
        dispatch_tokens<Gqa27Geometry>(
            q, positions, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;
    }

    if (cache.num_kv_heads == Gqa9Geometry::KVHeads) {
        dispatch_tokens<Gqa9Geometry>(
            q, positions, scale, cache, invocation,
            logical_capacity, implementation_window, splits,
            partial_acc, partial_m, partial_l, stream);
        return;
    }

    dispatch_tokens<Gqa35Geometry>(
        q, positions, scale, cache, invocation,
        logical_capacity, implementation_window, splits,
        partial_acc, partial_m, partial_l, stream);
}

} // namespace ninfer::ops::detail
