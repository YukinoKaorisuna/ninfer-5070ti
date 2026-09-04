#include "ninfer/ops/context_kv_materialize.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kLayers             = static_cast<int>(ops::kContextKVMaterializeLayers);
constexpr int kHidden             = 5120;
constexpr int kRows               = 1024;
constexpr int kHeadDim            = 128;
constexpr int kHeads              = 8;
constexpr int kCapacity           = 2048;
constexpr int kLaneCapacity       = 3;
constexpr std::uint16_t kSentinel = 0xa5a5U;
constexpr double kEpsilon         = 1.0e-6;
constexpr double kTheta           = 1.0e7;

constexpr ReductionCriterion kKeyCriterion{
    3.0e-3,
    1.0e-2,
    6.0e-3,
};

constexpr ReductionCriterion kValueCriterion{
    2.9e-3,
    4.0e-3,
    3.8e-3,
};

std::size_t cache_elements() {
    return static_cast<std::size_t>(kHeadDim) * kCapacity * kHeads * kLaneCapacity;
}

std::size_t cache_index(int lane, int head, int slot, int dim) {
    return static_cast<std::size_t>(dim) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kCapacity) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(kHeads) * lane));
}

std::vector<float> make_context(int columns, std::uint32_t seed) {
    std::vector<float> result(static_cast<std::size_t>(kHidden) * columns);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < kHidden; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 29U +
                                             static_cast<std::uint32_t>(column) * 47U + seed * 13U;
            const int centered = static_cast<int>(coordinate % 257U) - 128;
            result[static_cast<std::size_t>(column) * kHidden + row] =
                bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * (1.0F / 192.0F)));
        }
    }
    return result;
}

std::vector<float> make_norm(int layer) {
    std::vector<float> result(kHeadDim);
    for (int dim = 0; dim < kHeadDim; ++dim) {
        const float value = 0.75F + static_cast<float>(layer) * 0.03125F +
                            static_cast<float>((dim * 7 + layer) % 19) * (1.0F / 128.0F);
        result[static_cast<std::size_t>(dim)] = bf16_to_f32(f32_to_bf16(value));
    }
    return result;
}

struct LayerStorage {
    quantized_weight::PackedWeight key_host;
    quantized_weight::PackedWeight value_host;
    DeviceBuffer key_device;
    DeviceBuffer value_device;
    std::vector<float> norm_host;
    DeviceBuffer norm_device;
    DeviceBuffer cache_k;
    DeviceBuffer cache_v;
};

struct Fixture {
    std::array<LayerStorage, kLayers> storage;
    std::array<ops::ContextKVMaterializeLayerView, kLayers> views;

    Fixture() {
        const quantized_weight::PatternedWeightOptions weight_options{
            quantized_weight::RowSplitScalePattern::Tiny};
        for (int layer = 0; layer < kLayers; ++layer) {
            LayerStorage& target = storage[static_cast<std::size_t>(layer)];
            target.key_host      = quantized_weight::make_patterned_weight(
                QType::W8G32_F16S, kRows, kHidden, 0x310U + 2U * layer, weight_options);
            target.value_host = quantized_weight::make_patterned_weight(
                QType::W8G32_F16S, kRows, kHidden, 0x311U + 2U * layer, weight_options);
            target.key_device   = to_device(target.key_host.payload);
            target.value_device = to_device(target.value_host.payload);
            target.norm_host    = make_norm(layer);
            target.norm_device  = to_device_bf16(target.norm_host);
            target.cache_k      = DeviceBuffer(cache_elements() * sizeof(std::uint16_t));
            target.cache_v      = DeviceBuffer(cache_elements() * sizeof(std::uint16_t));
            target.cache_k.fill(0xa5);
            target.cache_v.fill(0xa5);

            views[static_cast<std::size_t>(layer)] = {
                target.key_host.device_weight(target.key_device.p),
                target.value_host.device_weight(target.value_device.p),
                Tensor(target.norm_device.p, DType::BF16, {kHeadDim}),
                CyclicKVCacheLayerView{
                    .k               = Tensor(target.cache_k.p, DType::BF16,
                                              {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                    .v               = Tensor(target.cache_v.p, DType::FP16,
                                              {kHeadDim, kCapacity, kHeads, kLaneCapacity}),
                    .capacity        = kCapacity,
                    .padded_capacity = kCapacity,
                    .num_kv_heads    = kHeads,
                    .head_dim        = kHeadDim,
                    .lane_capacity   = kLaneCapacity,
                },
            };
        }
    }

    void reset_cache() {
        for (LayerStorage& layer : storage) {
            layer.cache_k.fill(0xa5);
            layer.cache_v.fill(0xa5);
        }
    }
};

float represented_projection(const quantized_weight::PackedWeight& weight, int row,
                             const float* input) {
    const double dot = quantized_weight::dot_fp64(weight, row, input, kHidden);
    return bf16_to_f32(f32_to_bf16(static_cast<float>(dot)));
}

void append_key_oracle(const LayerStorage& layer, const float* input, int head, int position,
                       std::vector<double>& expected) {
    std::array<double, kHeadDim> raw{};
    double square_sum = 0.0;
    for (int dim = 0; dim < kHeadDim; ++dim) {
        const int row                      = head * kHeadDim + dim;
        raw[static_cast<std::size_t>(dim)] = represented_projection(layer.key_host, row, input);
        square_sum += raw[static_cast<std::size_t>(dim)] * raw[static_cast<std::size_t>(dim)];
    }
    const double inverse = 1.0 / std::sqrt(square_sum / kHeadDim + kEpsilon);
    for (int pair = 0; pair < kHeadDim / 2; ++pair) {
        const double first = raw[static_cast<std::size_t>(pair)] * inverse *
                             layer.norm_host[static_cast<std::size_t>(pair)];
        const double second = raw[static_cast<std::size_t>(pair + kHeadDim / 2)] * inverse *
                              layer.norm_host[static_cast<std::size_t>(pair + kHeadDim / 2)];
        const double angle = static_cast<double>(position) *
                             std::pow(kTheta, -2.0 * static_cast<double>(pair) / kHeadDim);
        const double cosine = std::cos(angle);
        const double sine   = std::sin(angle);
        expected.push_back(first * cosine - second * sine);
        expected.push_back(second * cosine + first * sine);
    }
}

int verify_state_effect(const std::string& label, const Fixture& fixture,
                        const std::vector<int>& positions, const std::vector<int>& counts,
                        const std::vector<int>& state_slots, int width) {
    std::vector<std::set<int>> written_slots(kLaneCapacity);
    for (std::size_t batch = 0; batch < counts.size(); ++batch) {
        for (int index = 0; index < counts[batch]; ++index) {
            written_slots[static_cast<std::size_t>(state_slots[batch])].insert(
                positions[batch * static_cast<std::size_t>(width) + index] & (kCapacity - 1));
        }
    }

    int failures = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const LayerStorage& source = fixture.storage[static_cast<std::size_t>(layer)];
        const auto cache_k         = from_device<std::uint16_t>(source.cache_k, cache_elements());
        const auto cache_v         = from_device<std::uint16_t>(source.cache_v, cache_elements());
        int first_bad              = -1;
        for (int lane = 0; lane < kLaneCapacity && first_bad < 0; ++lane) {
            for (int head = 0; head < kHeads && first_bad < 0; ++head) {
                for (int slot = 0; slot < kCapacity && first_bad < 0; ++slot) {
                    const bool written =
                        written_slots[static_cast<std::size_t>(lane)].contains(slot);
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        const std::size_t index = cache_index(lane, head, slot, dim);
                        const bool k_changed    = cache_k[index] != kSentinel;
                        const bool v_changed    = cache_v[index] != kSentinel;
                        if (k_changed != written || v_changed != written) {
                            first_bad = static_cast<int>(index);
                            break;
                        }
                    }
                }
            }
        }
        if (first_bad >= 0) {
            std::cerr << label << " layer=" << layer
                      << ": invalid state-effect footprint at element " << first_bad << '\n';
            ++failures;
        }
    }
    return failures;
}

int verify_numeric_samples(const std::string& label, const Fixture& fixture,
                           const std::vector<float>& context, const std::vector<int>& positions,
                           const std::vector<int>& counts, const std::vector<int>& state_slots,
                           int width) {
    std::vector<std::pair<int, int>> samples;
    for (int batch = 0; batch < static_cast<int>(counts.size()); ++batch) {
        if (counts[static_cast<std::size_t>(batch)] == 0) continue;
        samples.emplace_back(batch, 0);
        if (counts[static_cast<std::size_t>(batch)] > 1) {
            samples.emplace_back(batch, counts[static_cast<std::size_t>(batch)] - 1);
        }
    }

    int failures = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        const LayerStorage& source = fixture.storage[static_cast<std::size_t>(layer)];
        const auto cache_k_bits    = from_device<std::uint16_t>(source.cache_k, cache_elements());
        const auto cache_v_bits    = from_device<std::uint16_t>(source.cache_v, cache_elements());
        std::vector<double> key_got;
        std::vector<double> key_expected;
        std::vector<double> value_got;
        std::vector<double> value_expected;
        const std::array<int, 3> heads{0, 3, 7};
        const std::array<int, 4> value_rows{0, 127, 511, 1023};
        for (const auto [batch, local] : samples) {
            const int column   = batch * width + local;
            const int position = positions[static_cast<std::size_t>(column)];
            const int slot     = position & (kCapacity - 1);
            const int lane     = state_slots[static_cast<std::size_t>(batch)];
            const float* input = context.data() + static_cast<std::size_t>(column) * kHidden;
            for (const int head : heads) {
                std::vector<double> head_expected;
                append_key_oracle(source, input, head, position, head_expected);
                for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                    const int first_dim  = pair;
                    const int second_dim = pair + kHeadDim / 2;
                    key_got.push_back(
                        bf16_to_f32(cache_k_bits[cache_index(lane, head, slot, first_dim)]));
                    key_expected.push_back(head_expected[static_cast<std::size_t>(2 * pair)]);
                    key_got.push_back(
                        bf16_to_f32(cache_k_bits[cache_index(lane, head, slot, second_dim)]));
                    key_expected.push_back(head_expected[static_cast<std::size_t>(2 * pair + 1)]);
                }
            }
            for (const int row : value_rows) {
                const int head          = row / kHeadDim;
                const int dim           = row - head * kHeadDim;
                const float represented = represented_projection(source.value_host, row, input);
                const std::uint16_t expected_bits =
                    quantized_weight::detail::f32_to_f16(represented);
                value_expected.push_back(quantized_weight::detail::f16_to_f32(expected_bits));
                value_got.push_back(quantized_weight::detail::f16_to_f32(
                    cache_v_bits[cache_index(lane, head, slot, dim)]));
            }
        }
        failures += verify_reduction(label + " K layer=" + std::to_string(layer), key_got,
                                     key_expected, kKeyCriterion);
        failures += verify_reduction(label + " V layer=" + std::to_string(layer), value_got,
                                     value_expected, kValueCriterion);
    }
    return failures;
}

int run_case(Fixture& fixture, const std::string& label, int width, int batch,
             const std::vector<int>& counts, const std::vector<int>& state_slots,
             std::vector<int> positions, std::uint32_t input_seed) {
    const int columns             = width * batch;
    std::vector<float> context    = make_context(columns, input_seed);
    DeviceBuffer context_device   = to_device_bf16(context);
    DeviceBuffer positions_device = to_device_i32(positions);
    DeviceBuffer counts_device    = to_device_i32(counts);
    DeviceBuffer slots_device     = to_device_i32(state_slots);
    const std::size_t workspace_bytes =
        ops::context_kv_materialize_workspace_capacity_bytes(batch, width, width);
    WorkspaceArena workspace(workspace_bytes);
    Tensor context_tensor(context_device.p, DType::BF16, {kHidden, width, batch});
    Tensor positions_tensor(positions_device.p, DType::I32, {width, batch});
    Tensor counts_tensor(counts_device.p, DType::I32, {batch});
    Tensor slots_tensor(slots_device.p, DType::I32, {batch});
    const int minimum = *std::min_element(counts.begin(), counts.end());
    const int maximum = *std::max_element(counts.begin(), counts.end());
    ops::context_kv_materialize(
        context_tensor, positions_tensor, counts_tensor, slots_tensor, fixture.views,
        {static_cast<std::uint32_t>(minimum), static_cast<std::uint32_t>(maximum)}, workspace,
        nullptr);
    cuda_synchronize();

    int failures = verify_state_effect(label, fixture, positions, counts, state_slots, width);
    failures +=
        verify_numeric_samples(label, fixture, context, positions, counts, state_slots, width);
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "context_kv_materialize: SKIP (CUDA unavailable)\n";
            return 77;
        }

        Fixture fixture;
        int failures = 0;
        failures += run_case(fixture, "context_kv_materialize decode", 8, 3, {0, 3, 8}, {2, 0, 1},
                             {-31, -30, -29, -28, -27,  -26,  -25,  -24,  2046, 2047, 2048, 0,
                              0,   0,   0,   0,   4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099},
                             0x701U);

        fixture.reset_cache();
        failures += run_case(fixture, "context_kv_materialize prefill", 5, 1, {5}, {2},
                             {6142, 6143, 6144, 6145, 6146}, 0x702U);

        if (failures != 0) {
            std::cerr << "context_kv_materialize failures=" << failures << '\n';
            return 1;
        }
        std::cout << "context_kv_materialize: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "context_kv_materialize: " << error.what() << '\n';
        return 1;
    }
}
