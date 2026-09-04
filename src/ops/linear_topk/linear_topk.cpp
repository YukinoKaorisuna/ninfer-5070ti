#include "ninfer/ops/linear_topk.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear_topk/linear_topk_launch.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

enum class HeadProfile : std::uint8_t {
    W8Full,
    Fp8Full,
    Q4Optimized,
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

HeadProfile resolve_profile(QType qtype, std::int32_t head_rows, std::int32_t input_rows) {
    if (input_rows != detail::kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk: unsupported head profile");
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::W8G32_F16S) {
        return HeadProfile::W8Full;
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return HeadProfile::Fp8Full;
    }
    if (head_rows == detail::kLinearTopKOptimizedRows && qtype == QType::Q4G64_F16S) {
        return HeadProfile::Q4Optimized;
    }
    throw std::invalid_argument("linear_topk: unsupported head profile");
}

std::int32_t producer_rows_for(HeadProfile profile, std::int32_t columns) {
    // The Q4 T=14 small-T producer is materially faster than the 64-row family and therefore owns
    // the same 16-row partial-candidate geometry as T=7.
    if (columns == detail::kLinearTopKWidth ||
        (profile == HeadProfile::Q4Optimized && columns == 2 * detail::kLinearTopKWidth)) {
        return detail::kLinearTopKKSplitRowsPerGroup;
    }
    return detail::kLinearTopKMRowsPerGroup;
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
    }
}

std::int32_t validate_io(const Tensor& hidden, const Tensor& candidate_ids,
                         const Tensor& candidate_scores) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != detail::kLinearTopKHidden ||
        hidden.ne[1] <= 0 || (hidden.ne[1] % detail::kLinearTopKWidth) != 0 || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1 || !hidden.is_contiguous() || !aligned_to(hidden.data, 16)) {
        throw std::invalid_argument("linear_topk: invalid hidden");
    }
    const std::int32_t batch = hidden.ne[1] / detail::kLinearTopKWidth;
    if (batch < 1 || batch > detail::kLinearTopKMaxBatch) {
        throw std::invalid_argument("linear_topk: B must be in [1,8]");
    }
    const auto require_output = [&](const Tensor& tensor, DType dtype, const char* label) {
        if (tensor.dtype != dtype || tensor.ne[0] != detail::kLinearTopK ||
            tensor.ne[1] != detail::kLinearTopKWidth || tensor.ne[2] != batch ||
            tensor.ne[3] != 1 || !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
            throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
        }
    };
    require_output(candidate_ids, DType::I32, "candidate_ids");
    require_output(candidate_scores, DType::FP32, "candidate_scores");
    if (overlaps(hidden, candidate_ids) || overlaps(hidden, candidate_scores) ||
        overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("linear_topk: input and outputs must not overlap");
    }
    return batch;
}

void require_w8(const Weight& head) {
    const bool common =
        head.qtype == QType::W8G32_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 32 && head.group == 32 &&
        head.ndim == 2 && head.n == detail::kLinearTopKFullRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid W8 full head"); }
}

void require_q4(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4G64_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == detail::kLinearTopKOptimizedRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q4 optimized head"); }
}

void require_no_weight_overlap(const Weight& head, const Tensor& hidden,
                               const Tensor& candidate_ids, const Tensor& candidate_scores,
                               const Tensor* id_map, const detail::LinearTopKWorkspace& scratch) {
    const std::size_t code_bytes = head.qtype == QType::Q4G64_F16S
                                       ? static_cast<std::size_t>(head.n) * head.k / 2
                                       : static_cast<std::size_t>(head.n) * head.k;
    const std::size_t scale_bytes =
        head.qtype == QType::W8G32_F16S
            ? static_cast<std::size_t>(head.n) * (head.k / 32) * sizeof(std::uint16_t)
        : head.qtype == QType::Q4G64_F16S
            ? static_cast<std::size_t>(head.n) * (head.k / 64) * sizeof(std::uint16_t)
            : static_cast<std::size_t>(head.n) * sizeof(std::uint16_t);
    const Tensor* tensors[]{&hidden,
                            &candidate_ids,
                            &candidate_scores,
                            &scratch.partial_keys,
                            &scratch.group_keys,
                            &scratch.secondary_keys,
                            &scratch.group_done,
                            id_map};
    for (const Tensor* tensor : tensors) {
        if (tensor == nullptr) { continue; }
        if (overlaps(head.qdata, code_bytes, tensor->data, tensor->bytes()) ||
            overlaps(head.scales, scale_bytes, tensor->data, tensor->bytes())) {
            throw std::invalid_argument("linear_topk: weight plane overlaps a live tensor");
        }
    }
}

void require_scratch_nonoverlap(const Tensor& hidden, const Tensor& candidate_ids,
                                const Tensor& candidate_scores, const Tensor* id_map,
                                const detail::LinearTopKWorkspace& scratch) {
    const Tensor* live[]{&hidden, &candidate_ids, &candidate_scores, id_map};
    const Tensor* work[]{&scratch.partial_keys, &scratch.group_keys, &scratch.secondary_keys,
                         &scratch.group_done};
    for (const Tensor* lhs : live) {
        if (lhs == nullptr) { continue; }
        for (const Tensor* rhs : work) {
            if (overlaps(*lhs, *rhs)) {
                throw std::invalid_argument("linear_topk: workspace overlaps a live tensor");
            }
        }
    }
}

} // namespace

std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                 std::int32_t input_rows,
                                                 std::int32_t min_batch_size,
                                                 std::int32_t max_batch_size) {
    const HeadProfile profile = resolve_profile(qtype, head_rows, input_rows);
    if (min_batch_size < 1 || max_batch_size < min_batch_size ||
        max_batch_size > detail::kLinearTopKMaxBatch) {
        throw std::invalid_argument("linear_topk workspace: invalid batch interval");
    }
    WorkspaceLayoutBuilder layout;
    for (std::int32_t batch = min_batch_size; batch <= max_batch_size; ++batch) {
        auto scope                 = layout.scope();
        const std::int32_t columns = detail::kLinearTopKWidth * batch;
        (void)detail::allocate_linear_topk_workspace(layout, head_rows, columns,
                                                     producer_rows_for(profile, columns));
    }
    return layout.peak_bytes();
}

void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    const std::int32_t batch  = validate_io(hidden, candidate_ids, candidate_scores);
    const HeadProfile profile = resolve_profile(head.qtype, head.n, head.k);
    if (profile == HeadProfile::Q4Optimized || valid_rows != detail::kLinearTopKFullValidRows) {
        throw std::invalid_argument("linear_topk: invalid full-head profile or valid_rows");
    }
    if (profile == HeadProfile::W8Full) {
        require_w8(head);
    } else {
        (void)detail::validate_fp8_weight(head, "linear_topk FP8 full head");
    }

    auto scope                                = workspace.scope();
    const detail::LinearTopKWorkspace scratch = detail::allocate_linear_topk_workspace(
        workspace, head.n, detail::kLinearTopKWidth * batch,
        producer_rows_for(profile, detail::kLinearTopKWidth * batch));
    require_scratch_nonoverlap(hidden, candidate_ids, candidate_scores, nullptr, scratch);
    require_no_weight_overlap(head, hidden, candidate_ids, candidate_scores, nullptr, scratch);
    if (profile == HeadProfile::W8Full) {
        detail::linear_topk_w8_launch(hidden, head, valid_rows, scratch, stream);
    } else {
        detail::linear_topk_fp8_launch(hidden, head, valid_rows, scratch, stream);
    }
    detail::linear_topk_merge_launch(scratch, candidate_ids, candidate_scores, stream);
}

void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    const std::int32_t batch = validate_io(hidden, candidate_ids, candidate_scores);
    if (resolve_profile(head.qtype, head.n, head.k) != HeadProfile::Q4Optimized) {
        throw std::invalid_argument("linear_topk: invalid optimized-head profile");
    }
    require_q4(head);
    require_matrix(row_to_global_ids, DType::I32, detail::kLinearTopKOptimizedRows, 1,
                   "row_to_global_ids", 4);
    if (overlaps(hidden, row_to_global_ids) || overlaps(candidate_ids, row_to_global_ids) ||
        overlaps(candidate_scores, row_to_global_ids)) {
        throw std::invalid_argument("linear_topk: id map overlaps input or output");
    }

    auto scope                                = workspace.scope();
    const detail::LinearTopKWorkspace scratch = detail::allocate_linear_topk_workspace(
        workspace, head.n, detail::kLinearTopKWidth * batch,
        producer_rows_for(HeadProfile::Q4Optimized, detail::kLinearTopKWidth * batch));
    require_scratch_nonoverlap(hidden, candidate_ids, candidate_scores, &row_to_global_ids,
                               scratch);
    require_no_weight_overlap(head, hidden, candidate_ids, candidate_scores, &row_to_global_ids,
                              scratch);
    detail::linear_topk_q4_launch(hidden, head, row_to_global_ids, scratch, stream);
    detail::linear_topk_merge_launch(scratch, candidate_ids, candidate_scores, stream);
}

} // namespace ninfer::ops
