// Cold-cache benchmark for the complete DFlash2 five-layer context state transition.

#include "ninfer/ops/context_kv_materialize.h"

#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/rmsnorm_rope.h"

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kLayers       = static_cast<int>(ops::kContextKVMaterializeLayers);
constexpr int kParentRows   = 6144;
constexpr int kKeyRow       = 4096;
constexpr int kValueRow     = 5120;
constexpr int kHidden       = 5120;
constexpr int kRows         = 1024;
constexpr int kHeadDim      = 128;
constexpr int kHeads        = 8;
constexpr int kWidth        = 8;
constexpr int kCapacity     = 2048;
constexpr int kLaneCapacity = 8;

enum class Execution : std::uint8_t { Eager, Graph };

struct Options {
    std::vector<int> batches{1, 2, 3, 4, 5, 6, 7, 8};
    Execution execution     = Execution::Graph;
    int warmup              = 10;
    int repeat              = 100;
    std::size_t flush_bytes = 256ULL << 20;
};

[[noreturn]] void usage(const char* program, const char* error) {
    std::fprintf(stderr,
                 "error: %s\nusage: %s [--batches B,...] [--execution eager|graph] "
                 "[--warmup N] [--repeat N] [--flush-mib N]\n",
                 error, program);
    std::exit(2);
}

int parse_i32(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string("invalid ") + flag);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_batches(std::string_view text) {
    std::vector<int> result;
    while (!text.empty()) {
        const std::size_t comma     = text.find(',');
        const std::string_view item = text.substr(0, comma);
        result.push_back(parse_i32(item, 1, 8, "--batches"));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    if (result.empty()) throw std::invalid_argument("empty --batches");
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (++index >= argc) usage(argv[0], "missing option value");
            return argv[index];
        };
        if (argument == "--batches") {
            options.batches = parse_batches(next());
        } else if (argument == "--execution") {
            const std::string_view value = next();
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else
                usage(argv[0], "--execution expects eager or graph");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next(), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next(), 1, 10000, "--repeat");
        } else if (argument == "--flush-mib") {
            const int mib       = parse_i32(next(), 1, 4096, "--flush-mib");
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0], "help");
        } else {
            usage(argv[0], "unknown option");
        }
    }
    return options;
}

struct Fixture {
    std::array<bench::PackedQuantizedWeight, kLayers> parents;
    std::array<DeviceBuffer, kLayers> norms;
    std::array<DeviceBuffer, kLayers> cache_k;
    std::array<DeviceBuffer, kLayers> cache_v;
    std::array<ops::ContextKVMaterializeLayerView, kLayers> layers;
    DeviceBuffer context = bench::make_bf16(static_cast<std::size_t>(kHidden) * kWidth * 8);
    DeviceBuffer positions;
    DeviceBuffer counts;
    DeviceBuffer slots;
    DeviceBuffer direct_workspace;
    WorkspaceArena direct_arena;
    DeviceBuffer composed_key;
    DeviceBuffer composed_value;

    Fixture()
        : positions(sizeof(std::int32_t) * kWidth * 8), counts(sizeof(std::int32_t) * 8),
          slots(sizeof(std::int32_t) * 8),
          direct_workspace(ops::context_kv_materialize_workspace_capacity_bytes(8, kWidth, kWidth)),
          direct_arena(DeviceSpan{direct_workspace.p, direct_workspace.bytes}),
          composed_key(static_cast<std::size_t>(kRows) * kWidth * 8 * 2),
          composed_value(static_cast<std::size_t>(kRows) * kWidth * 8 * 2) {
        std::vector<std::int32_t> host_positions(kWidth * 8);
        std::vector<std::int32_t> host_counts(8, kWidth);
        std::vector<std::int32_t> host_slots(8);
        for (int batch = 0; batch < 8; ++batch) {
            host_slots[static_cast<std::size_t>(batch)] = batch;
            for (int index = 0; index < kWidth; ++index) {
                host_positions[static_cast<std::size_t>(batch * kWidth + index)] =
                    8192 * batch + 1024 + index;
            }
        }
        positions.copy_from_host(host_positions.data(), positions.bytes);
        counts.copy_from_host(host_counts.data(), counts.bytes);
        slots.copy_from_host(host_slots.data(), slots.bytes);

        const std::size_t cache_bytes = static_cast<std::size_t>(kHeadDim) * kCapacity * kHeads *
                                        kLaneCapacity * sizeof(std::uint16_t);
        for (int layer = 0; layer < kLayers; ++layer) {
            parents[static_cast<std::size_t>(layer)] =
                bench::make_row_split_weight(QType::W8G32_F16S, kParentRows, kHidden, kHidden,
                                             {static_cast<std::uint8_t>(0x31 + layer), 0, 0x2800});
            norms[static_cast<std::size_t>(layer)]   = bench::make_bf16(kHeadDim);
            cache_k[static_cast<std::size_t>(layer)] = DeviceBuffer(cache_bytes);
            cache_v[static_cast<std::size_t>(layer)] = DeviceBuffer(cache_bytes);
            layers[static_cast<std::size_t>(layer)]  = {
                bench::row_view(parents[static_cast<std::size_t>(layer)].weight, kKeyRow, kRows),
                bench::row_view(parents[static_cast<std::size_t>(layer)].weight, kValueRow, kRows),
                Tensor(norms[static_cast<std::size_t>(layer)].p, DType::BF16, {kHeadDim}),
                CyclicKVCacheLayerView{
                     .k        = Tensor(cache_k[static_cast<std::size_t>(layer)].p, DType::BF16,
                                        {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                     .v        = Tensor(cache_v[static_cast<std::size_t>(layer)].p, DType::FP16,
                                        {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                     .capacity = kCapacity,
                     .padded_capacity = kCapacity,
                     .num_kv_heads    = kHeads,
                     .head_dim        = kHeadDim,
                     .lane_capacity   = kLaneCapacity,
                },
            };
        }
    }

    void direct(int batch, cudaStream_t stream) {
        Tensor x(context.p, DType::BF16, {kHidden, kWidth, batch});
        Tensor pos(positions.p, DType::I32, {kWidth, batch});
        Tensor count(counts.p, DType::I32, {batch});
        Tensor lane(slots.p, DType::I32, {batch});
        ops::context_kv_materialize(x, pos, count, lane, layers, {kWidth, kWidth}, direct_arena,
                                    stream);
    }

    void composed(int batch, cudaStream_t stream) {
        const int columns = kWidth * batch;
        Tensor x(context.p, DType::BF16, {kHidden, columns});
        Tensor pos(positions.p, DType::I32, {kWidth, batch});
        Tensor count(counts.p, DType::I32, {batch});
        Tensor lane(slots.p, DType::I32, {batch});
        Tensor key(composed_key.p, DType::BF16, {kRows, columns});
        Tensor value(composed_value.p, DType::BF16, {kRows, columns});
        for (const auto& layer : layers) {
            ops::linear_pair(x, layer.key_weight, layer.value_weight, key, value, stream);
            Tensor key_heads = key.view({kHeadDim, kHeads, columns});
            ops::rmsnorm_rope(pos.view({columns}), layer.key_norm_weight, key_heads, stream);
            ops::kv_cache_append_prefix(key_heads.view({kHeadDim, kHeads, kWidth, batch}),
                                        value.view({kHeadDim, kHeads, kWidth, batch}), pos, count,
                                        lane, {kWidth, kWidth}, layer.cache, stream);
        }
    }
};

template <class Launch>
bench::ColdTiming measure(const Options& options, Launch&& launch, DeviceBuffer& flush,
                          cudaStream_t stream, std::size_t* graph_nodes) {
    if (options.execution == Execution::Eager) {
        *graph_nodes = 0;
        return bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
    }
    bench::TimedGraph graph;
    graph.capture(stream, launch);
    *graph_nodes = graph.nodes();
    return bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        Fixture fixture;
        DeviceBuffer flush(options.flush_bytes);
        std::printf("# gpu=%s public=context_kv_materialize geometry=L5_W8_K5120_N1024 cache=cold "
                    "flush_mib=%zu execution=%s\n",
                    properties.name, options.flush_bytes >> 20,
                    options.execution == Execution::Graph ? "graph" : "eager");
        for (const int batch : options.batches) {
            std::size_t direct_nodes   = 0;
            std::size_t composed_nodes = 0;
            const auto direct          = measure(
                options, [&](cudaStream_t s) { fixture.direct(batch, s); }, flush, stream,
                &direct_nodes);
            const auto composed = measure(
                options, [&](cudaStream_t s) { fixture.composed(batch, s); }, flush, stream,
                &composed_nodes);
            std::printf(
                "B=%d T=%d direct=%.3f_us direct_nodes=%zu composed=%.3f_us composed_nodes=%zu "
                "speedup=%.3fx direct_min=%.3f_us direct_p95=%.3f_us\n",
                batch, kWidth * batch, direct.median_us, direct_nodes, composed.median_us,
                composed_nodes, composed.median_us / direct.median_us, direct.min_us,
                direct.p95_us);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_context_kv_materialize_bench: %s\n", error.what());
        return 1;
    }
}
