// Public cold-cache benchmark for the three registered linear_topk profiles.
#include "ninfer/ops/linear_topk.h"

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

enum class Profile {
    W8,
    Fp8,
    Q4,
};

const char* profile_name(Profile profile) {
    if (profile == Profile::W8) { return "w8-full"; }
    if (profile == Profile::Fp8) { return "fp8-full"; }
    return "q4-optimized";
}

void run(Profile profile, std::int32_t batch) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kTopK   = 16;
    const std::int32_t columns     = 7 * batch;
    const std::int32_t rows        = profile == Profile::Q4 ? 131072 : 248320;
    const QType qtype              = profile == Profile::W8    ? QType::W8G32_F16S
                                     : profile == Profile::Fp8 ? QType::FP8_E4M3FN_ROW_BF16S
                                                               : QType::Q4G64_F16S;

    PackedQuantizedWeight packed =
        profile == Profile::Fp8
            ? make_fp8_weight(rows, kHidden)
            : make_row_split_weight(qtype, rows, kHidden, kHidden,
                                    QuantizedWeightFill{profile == Profile::Q4 ? std::uint8_t{0x11}
                                                                               : std::uint8_t{0x01},
                                                        0, 0x3c00});
    DeviceBuffer hidden = make_bf16(static_cast<std::size_t>(kHidden) * columns);
    DeviceBuffer ids(static_cast<std::size_t>(kTopK) * columns * sizeof(std::int32_t));
    DeviceBuffer scores(static_cast<std::size_t>(kTopK) * columns * sizeof(float));
    DeviceBuffer id_map(
        profile == Profile::Q4 ? static_cast<std::size_t>(rows) * sizeof(std::int32_t) : 1);
    if (profile == Profile::Q4) {
        std::vector<std::int32_t> host_ids(static_cast<std::size_t>(rows));
        for (std::int32_t row = 0; row < rows; ++row) { host_ids[row] = row; }
        id_map.copy_from_host(host_ids.data(), id_map.bytes);
    }

    const std::size_t workspace_bytes =
        ops::linear_topk_workspace_capacity_bytes(qtype, rows, kHidden, batch, batch);
    WorkspaceArena workspace(workspace_bytes);
    Tensor hidden_tensor(hidden.p, DType::BF16, {kHidden, columns});
    Tensor ids_tensor(ids.p, DType::I32, {kTopK, 7, batch});
    Tensor scores_tensor(scores.p, DType::FP32, {kTopK, 7, batch});
    Tensor map_tensor(id_map.p, DType::I32, {rows});
    DeviceBuffer flush(256ULL << 20);
    cudaStream_t stream = nullptr;

    const auto launch = [&](cudaStream_t launch_stream) {
        if (profile == Profile::Q4) {
            ops::linear_topk(hidden_tensor, packed.weight, map_tensor, ids_tensor, scores_tensor,
                             workspace, launch_stream);
        } else {
            ops::linear_topk(hidden_tensor, packed.weight, 248077, ids_tensor, scores_tensor,
                             workspace, launch_stream);
        }
    };
    const ColdTiming timing    = measure_cold_launch(launch, flush, stream, 5, 50);
    const double weight_bytes  = static_cast<double>(packed.model_weight_bytes());
    const double effective_gbs = weight_bytes / timing.median_us / 1000.0;
    const double useful_tflops = 2.0 * static_cast<double>(profile == Profile::Q4 ? rows : 248077) *
                                 kHidden * columns / timing.median_us / 1.0e6;
    std::printf("linear_topk profile=%s B=%d T=%d median=%.3f us min=%.3f us p95=%.3f us "
                "weight_bw=%.1f GB/s useful=%.2f TFLOP/s workspace=%zu bytes\n",
                profile_name(profile), batch, columns, timing.median_us, timing.min_us,
                timing.p95_us, effective_gbs, useful_tflops, workspace_bytes);
}

Profile parse_profile(std::string_view value) {
    if (value == "w8" || value == "w8-full") { return Profile::W8; }
    if (value == "fp8" || value == "fp8-full") { return Profile::Fp8; }
    if (value == "q4" || value == "q4-optimized") { return Profile::Q4; }
    throw std::invalid_argument("profile must be w8-full, fp8-full, or q4-optimized");
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        if (argc == 1) {
            for (Profile profile : {Profile::W8, Profile::Fp8, Profile::Q4}) {
                for (std::int32_t batch = 1; batch <= 8; ++batch) { run(profile, batch); }
            }
            return 0;
        }
        if (argc != 3) {
            std::fprintf(stderr, "usage: %s [w8-full|fp8-full|q4-optimized B]\n", argv[0]);
            return 2;
        }
        const int batch = std::atoi(argv[2]);
        if (batch < 1 || batch > 8) { throw std::invalid_argument("B must be in [1,8]"); }
        run(parse_profile(argv[1]), batch);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "linear_topk benchmark: %s\n", error.what());
        return 1;
    }
}
