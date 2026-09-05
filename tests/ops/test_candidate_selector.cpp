#include "ninfer/ops/candidate_selector.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kCandidates   = 16;
constexpr std::int32_t kSteps        = 7;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;
constexpr std::int32_t kMaxBatch     = 8;

constexpr PointwiseCriterion kProbabilityCriterion{/*absolute=*/2.0e-6,
                                                   /*relative=*/2.0e-5};

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float represented_pattern(std::uint32_t first, std::uint32_t second, std::uint32_t seed,
                          float scale) {
    const std::uint32_t mixed = mix32(first * 0x9e3779b9U ^ second * 0x85ebca6bU ^ seed);
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) { centered = ((first + second) & 1U) == 0 ? 1 : -1; }
    return bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * scale));
}

float predecessor_value(std::int32_t token, std::int32_t rank) {
    return represented_pattern(static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(rank),
                               101U, 1.0F / 256.0F);
}

float successor_value(std::int32_t token, std::int32_t rank) {
    return represented_pattern(static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(rank),
                               211U, 1.0F / 256.0F);
}

std::size_t candidate_offset(std::int32_t batch, std::int32_t step, std::int32_t candidate) {
    return (static_cast<std::size_t>(batch) * kSteps + step) * kCandidates + candidate;
}

std::size_t lattice_offset(std::int32_t batch, std::int32_t step, std::int32_t predecessor_rank,
                           std::int32_t candidate) {
    return ((static_cast<std::size_t>(batch) * kSteps + step) * kCandidates + predecessor_rank) *
               kCandidates +
           candidate;
}

std::vector<std::int32_t> make_candidate_ids() {
    std::vector<std::int32_t> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                const std::int32_t token = 1000 + batch * 20000 + step * 257 + candidate;
                if (token >= kTokenDomain) { throw std::logic_error("candidate fixture overflow"); }
                result[candidate_offset(batch, step, candidate)] = token;
            }
        }
    }
    return result;
}

std::vector<float> make_unary_scores() {
    std::vector<float> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                const std::uint32_t mixed = mix32(static_cast<std::uint32_t>(
                    (batch * kSteps + step) * kCandidates + candidate + 307));
                const float variation =
                    static_cast<float>(static_cast<int>((mixed >> 10) & 0xffU) - 128) / 512.0F;
                result[candidate_offset(batch, step, candidate)] =
                    variation + static_cast<float>(kCandidates - candidate) * 0.0078125F;
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_projected_hidden() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kRank);
    for (std::int32_t column = 0; column < kMaxBatch * kSteps; ++column) {
        for (std::int32_t rank = 0; rank < kRank; ++rank) {
            result[static_cast<std::size_t>(column) * kRank + rank] = f32_to_bf16(
                represented_pattern(static_cast<std::uint32_t>(column),
                                    static_cast<std::uint32_t>(rank), 401U, 1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<std::int32_t> make_anchors() {
    std::vector<std::int32_t> result(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        result[batch] = 220000 + batch * 101;
    }
    return result;
}

std::vector<std::int32_t> make_base_positions() {
    std::vector<std::int32_t> result(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) { result[batch] = 1000 + batch * 97; }
    return result;
}

std::vector<std::int32_t> accessed_tokens(const std::vector<std::int32_t>& candidate_ids,
                                          const std::vector<std::int32_t>& anchors) {
    std::set<std::int32_t> tokens(anchors.begin(), anchors.end());
    tokens.insert(candidate_ids.begin(), candidate_ids.end());
    return {tokens.begin(), tokens.end()};
}

void populate_codebook(DeviceBuffer& device, std::span<const std::int32_t> tokens,
                       bool predecessor) {
    std::vector<std::uint16_t> row(kRank);
    for (const std::int32_t token : tokens) {
        for (std::int32_t rank = 0; rank < kRank; ++rank) {
            const float value =
                predecessor ? predecessor_value(token, rank) : successor_value(token, rank);
            row[rank] = f32_to_bf16(value);
        }
        device.copy_from_host(row.data(), row.size() * sizeof(std::uint16_t),
                              static_cast<std::size_t>(token) * kRank * sizeof(std::uint16_t));
    }
}

std::vector<double> build_lattice(const std::vector<std::int32_t>& candidate_ids,
                                  const std::vector<float>& unary_scores,
                                  const std::vector<std::uint16_t>& projected_hidden,
                                  const std::vector<std::int32_t>& anchors) {
    std::vector<double> lattice(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates *
                                kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t predecessor_rank = 0; predecessor_rank < kCandidates;
                 ++predecessor_rank) {
                const std::int32_t predecessor =
                    step == 0 ? anchors[batch]
                              : candidate_ids[candidate_offset(batch, step - 1, predecessor_rank)];
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    const std::int32_t successor =
                        candidate_ids[candidate_offset(batch, step, candidate)];
                    double edge = unary_scores[candidate_offset(batch, step, candidate)];
                    const std::size_t hidden_base =
                        static_cast<std::size_t>(batch * kSteps + step) * kRank;
                    for (std::int32_t rank = 0; rank < kRank; ++rank) {
                        edge +=
                            static_cast<double>(predecessor_value(predecessor, rank)) *
                            static_cast<double>(bf16_to_f32(projected_hidden[hidden_base + rank])) *
                            static_cast<double>(successor_value(successor, rank));
                    }
                    lattice[lattice_offset(batch, step, predecessor_rank, candidate)] = edge;
                }
            }
        }
    }
    return lattice;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float oracle_uniform(std::uint64_t seed, std::int32_t position) {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                            0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(ops::kSamplePurposeDFlash2Proposal) << 21));
    const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);
    return static_cast<float>(bits) * (1.0F / 16777216.0F);
}

struct WalkResult {
    std::vector<std::int32_t> drafts;
    std::vector<double> probabilities;
    std::array<double, kMaxBatch> minimum_decision_margin;
};

WalkResult walk_oracle(const std::vector<std::int32_t>& candidate_ids,
                       const std::vector<double>& lattice,
                       const std::vector<ops::SamplingConfig>& configs,
                       const std::vector<std::int32_t>& base_positions) {
    WalkResult result;
    result.drafts.resize(static_cast<std::size_t>(kMaxBatch) * kSteps);
    result.probabilities.resize(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    result.minimum_decision_margin.fill(std::numeric_limits<double>::infinity());
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        std::int32_t predecessor_rank = 0;
        for (std::int32_t step = 0; step < kSteps; ++step) {
            const std::size_t edge_base = lattice_offset(batch, step, predecessor_rank, 0);
            std::int32_t selected       = 0;
            if (configs[batch].temperature > 0.0F) {
                double maximum = lattice[edge_base];
                for (std::int32_t candidate = 1; candidate < kCandidates; ++candidate) {
                    maximum = std::max(maximum, lattice[edge_base + candidate]);
                }
                double sum = 0.0;
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    const double probability = std::exp((lattice[edge_base + candidate] - maximum) /
                                                        configs[batch].temperature);
                    result.probabilities[candidate_offset(batch, step, candidate)] = probability;
                    sum += probability;
                }
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    result.probabilities[candidate_offset(batch, step, candidate)] /= sum;
                }
                const double uniform =
                    oracle_uniform(configs[batch].seed, base_positions[batch] + step);
                double lower      = 0.0;
                double cumulative = 0.0;
                selected          = kCandidates - 1;
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    cumulative += result.probabilities[candidate_offset(batch, step, candidate)];
                    if (uniform < cumulative) {
                        selected = candidate;
                        result.minimum_decision_margin[batch] =
                            std::min(result.minimum_decision_margin[batch],
                                     std::min(uniform - lower, cumulative - uniform));
                        break;
                    }
                    lower = cumulative;
                }
            } else {
                double best   = lattice[edge_base];
                double second = -std::numeric_limits<double>::infinity();
                for (std::int32_t candidate = 1; candidate < kCandidates; ++candidate) {
                    const double edge = lattice[edge_base + candidate];
                    if (edge > best) {
                        second   = best;
                        best     = edge;
                        selected = candidate;
                    } else {
                        second = std::max(second, edge);
                    }
                }
                result.minimum_decision_margin[batch] =
                    std::min(result.minimum_decision_margin[batch], best - second);
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    result.probabilities[candidate_offset(batch, step, candidate)] =
                        candidate == selected ? 1.0 : 0.0;
                }
            }
            result.drafts[static_cast<std::size_t>(batch) * kSteps + step] =
                candidate_ids[candidate_offset(batch, step, selected)];
            predecessor_rank = selected;
        }
    }
    return result;
}

std::vector<ops::SamplingConfig> choose_configs(const std::vector<std::int32_t>& candidate_ids,
                                                const std::vector<double>& lattice,
                                                const std::vector<std::int32_t>& base_positions) {
    std::vector<ops::SamplingConfig> configs(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        if ((batch & 1) != 0) {
            configs[batch].temperature = 0.0F;
            configs[batch].seed        = 1000 + batch;
            continue;
        }
        configs[batch].temperature = 0.75F + 0.1F * static_cast<float>(batch % 3);
        bool found                 = false;
        for (std::uint64_t seed = 1; seed < 100000; ++seed) {
            configs[batch].seed = seed;
            const WalkResult candidate =
                walk_oracle(candidate_ids, lattice, configs, base_positions);
            if (candidate.minimum_decision_margin[batch] > 5.0e-3) {
                found = true;
                break;
            }
        }
        if (!found) { throw std::runtime_error("failed to choose selector RNG fixture"); }
    }
    const WalkResult final = walk_oracle(candidate_ids, lattice, configs, base_positions);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        const double minimum_margin = configs[batch].temperature > 0.0F ? 5.0e-3 : 5.0e-4;
        if (final.minimum_decision_margin[batch] <= minimum_margin) {
            throw std::runtime_error(
                "selector fixture has an unstable decision boundary at B=" + std::to_string(batch) +
                ": " + std::to_string(final.minimum_decision_margin[batch]));
        }
    }
    return configs;
}

int run() {
    const std::vector<std::int32_t> candidate_ids     = make_candidate_ids();
    const std::vector<float> unary_scores             = make_unary_scores();
    const std::vector<std::uint16_t> projected_hidden = make_projected_hidden();
    const std::vector<std::int32_t> anchors           = make_anchors();
    const std::vector<std::int32_t> base_positions    = make_base_positions();
    const std::vector<double> lattice =
        build_lattice(candidate_ids, unary_scores, projected_hidden, anchors);
    const std::vector<ops::SamplingConfig> configs =
        choose_configs(candidate_ids, lattice, base_positions);
    const WalkResult expected = walk_oracle(candidate_ids, lattice, configs, base_positions);

    DeviceBuffer candidate_device = to_device(candidate_ids);
    DeviceBuffer unary_device     = to_device(unary_scores);
    DeviceBuffer hidden_device    = to_device(projected_hidden);
    DeviceBuffer anchor_device    = to_device(anchors);
    DeviceBuffer position_device  = to_device(base_positions);
    DeviceBuffer config_device    = to_device(configs);
    DeviceBuffer predecessor_device(static_cast<std::size_t>(kRank) * kCodebookRows *
                                    sizeof(std::uint16_t));
    DeviceBuffer successor_device(static_cast<std::size_t>(kRank) * kCodebookRows *
                                  sizeof(std::uint16_t));
    predecessor_device.fill();
    successor_device.fill();
    const std::vector<std::int32_t> tokens = accessed_tokens(candidate_ids, anchors);
    populate_codebook(predecessor_device, tokens, true);
    populate_codebook(successor_device, tokens, false);

    Tensor predecessor(predecessor_device.p, DType::BF16, {kRank, kCodebookRows});
    Tensor successor(successor_device.p, DType::BF16, {kRank, kCodebookRows});

    int failures = 0;
    for (const std::int32_t batch_size : std::array<std::int32_t, 2>{1, kMaxBatch}) {
        const std::size_t draft_count = static_cast<std::size_t>(batch_size) * kSteps;
        const std::size_t q_count     = draft_count * kCandidates;
        GuardedDeviceBuffer draft_device(draft_count * sizeof(std::int32_t));
        GuardedDeviceBuffer q_device(q_count * sizeof(float));
        draft_device.fill(0xff);
        q_device.fill(0xff);
        Tensor ids(candidate_device.p, DType::I32, {kCandidates, kSteps, batch_size});
        Tensor unary(unary_device.p, DType::FP32, {kCandidates, kSteps, batch_size});
        Tensor hidden(hidden_device.p, DType::BF16, {kRank, kSteps, batch_size});
        Tensor anchor(anchor_device.p, DType::I32, {batch_size});
        Tensor positions(position_device.p, DType::I32, {batch_size});
        Tensor drafts(draft_device.data(), DType::I32, {kSteps, batch_size});
        Tensor q(q_device.data(), DType::FP32, {kCandidates, kSteps, batch_size});
        ops::candidate_selector_path(ids, unary, hidden, anchor, predecessor, successor, positions,
                                     static_cast<const ops::SamplingConfig*>(config_device.p),
                                     drafts, q, nullptr);
        cuda_synchronize();

        const std::string label = "candidate_selector_path B=" + std::to_string(batch_size);
        failures += verify_exact((label + " drafts").c_str(),
                                 from_device<std::int32_t>(draft_device.data(), draft_count),
                                 std::vector<std::int32_t>(expected.drafts.begin(),
                                                           expected.drafts.begin() + draft_count));
        const std::vector<float> actual_q = from_device<float>(q_device.data(), q_count);
        std::vector<double> actual_q_double(actual_q.begin(), actual_q.end());
        failures += verify_pointwise(
            label + " proposal_q", actual_q_double,
            std::span<const double>(expected.probabilities.data(), q_count), kProbabilityCriterion);
        failures += draft_device.verify_guards(label + " drafts");
        failures += q_device.verify_guards(label + " proposal_q");
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " candidate_selector_path\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "candidate_selector_path test: " << error.what() << '\n';
        return 1;
    }
}
