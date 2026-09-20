#include "ops/linear_add/linear_add_test_common.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int expect_schedule(std::int32_t rows, std::int32_t k, std::int32_t cols,
                    ninfer::ops::detail::Q5LinearAddScheduleId expected,
                    const char* label) {
    using namespace ninfer::ops::detail;

    const Q5LinearAddPlan plan =
        q5_linear_add_resolve_plan({rows, k, k, cols});

    if (plan.schedule != expected || plan.workspace_bytes != 0) {
        std::cerr << label
                  << " [" << rows << "," << k << "] T=" << cols
                  << ": expected "
                  << q5_linear_add_schedule_name(expected)
                  << ", got "
                  << q5_linear_add_schedule_name(plan.schedule)
                  << ", workspace=" << plan.workspace_bytes << '\n';
        return 1;
    }
    return 0;
}

int q5_route_policy() {
    using S = ninfer::ops::detail::Q5LinearAddScheduleId;

    int failures = 0;

    // 5120 x 6144: preserve fork C32/C64 boundaries and add only T=1
    // Split2 plus the 513+ tail schedule.
    for (const auto [t, expected] : std::array{
             std::pair{1, S::Split2ExactResidual},
             std::pair{13, S::Split2ExactResidual},
             std::pair{14, S::MmaResidualR64C16},
             std::pair{32, S::MmaResidualR64C16},
             std::pair{33, S::MmaResidualR64C24},
             std::pair{48, S::MmaResidualR64C24},
             std::pair{49, S::MmaResidualR64C32S4},
             std::pair{95, S::MmaResidualR64C32S4},
             std::pair{96, S::MmaResidualR64C64},
             std::pair{128, S::MmaResidualR64C64},
             std::pair{129, S::MmaResidualR64C128},
             std::pair{512, S::MmaResidualR64C128},
             std::pair{513, S::MmaResidualR64C128Tail},
             std::pair{621, S::MmaResidualR64C128Tail},
             std::pair{704, S::MmaResidualR64C128Tail},
             std::pair{705, S::MmaResidualR64C128Tail},
             std::pair{1025, S::MmaResidualR64C128Tail},
         }) {
        failures += expect_schedule(5120, 6144, t, expected, "Q5 route policy");
    }

    // 5120 x 17408: same principle with the fork's own C32S3/C64 crossover.
    for (const auto [t, expected] : std::array{
             std::pair{1, S::Split2ExactResidual},
             std::pair{16, S::Split2ExactResidual},
             std::pair{17, S::MmaResidualR64C16},
             std::pair{32, S::MmaResidualR64C16},
             std::pair{33, S::MmaResidualR64C24},
             std::pair{48, S::MmaResidualR64C24},
             std::pair{49, S::MmaResidualR64C32S3},
             std::pair{96, S::MmaResidualR64C32S3},
             std::pair{97, S::MmaResidualR64C64},
             std::pair{128, S::MmaResidualR64C64},
             std::pair{129, S::MmaResidualR64C128},
             std::pair{512, S::MmaResidualR64C128},
             std::pair{513, S::MmaResidualR64C128Tail},
             std::pair{621, S::MmaResidualR64C128Tail},
             std::pair{704, S::MmaResidualR64C128Tail},
             std::pair{705, S::MmaResidualR64C128Tail},
             std::pair{1025, S::MmaResidualR64C128Tail},
         }) {
        failures += expect_schedule(5120, 17408, t, expected, "Q5 route policy");
    }

    // Critical regression guard: both 4096 shapes retain their exact pre-port
    // policy. In particular T=1 MUST remain residual GEMV, and 513+ MUST remain
    // the ordinary C128 schedule rather than entering the new tail path.
    for (const std::int32_t k : {4096, 12288}) {
        for (const auto [t, expected] : std::array{
                 std::pair{1, S::GemvResidual},
                 std::pair{2, S::Split2ExactResidual},
                 std::pair{16, S::Split2ExactResidual},
                 std::pair{17, S::MmaResidualR64C16},
                 std::pair{32, S::MmaResidualR64C16},
                 std::pair{33, S::MmaResidualR64C24},
                 std::pair{48, S::MmaResidualR64C24},
                 std::pair{49, S::MmaResidualR64C32S3},
                 std::pair{96, S::MmaResidualR64C32S3},
                 std::pair{97, S::MmaResidualR64C64},
                 std::pair{128, S::MmaResidualR64C64},
                 std::pair{129, S::MmaResidualR64C128},
                 std::pair{512, S::MmaResidualR64C128},
                 std::pair{513, S::MmaResidualR64C128},
                 std::pair{1025, S::MmaResidualR64C128},
             }) {
            failures += expect_schedule(4096, k, t, expected, "Q5 4096 route policy");
        }
    }

    return failures;
}

int q5_a16_conformance() {
    // Route starts exercise b-1/b/b+1. Interiors cover the narrow-tail
    // thresholds, the observed production remainder T=621, fallback tails and
    // ordinary 896-column prefill geometry.
    constexpr std::array<std::int32_t, 6> kK6144RouteStarts{
        14, 33, 49, 96, 129, 513
    };
    constexpr std::array<std::int32_t, 13> kK6144RouteInteriors{
        1, 8, 24, 40, 64, 112, 256, 621, 640, 704, 705, 768, 896
    };

    constexpr std::array<std::int32_t, 6> kK17408RouteStarts{
        17, 33, 49, 97, 129, 513
    };
    constexpr std::array<std::int32_t, 13> kK17408RouteInteriors{
        1, 8, 24, 40, 64, 112, 256, 621, 640, 704, 705, 768, 896
    };

    // 4096 shapes are deliberately kept to their existing route boundaries.
    constexpr std::array<std::int32_t, 6> kK4096RouteStarts{
        2, 17, 33, 49, 97, 129
    };
    constexpr std::array<std::int32_t, 7> kK4096RouteInteriors{
        1, 8, 24, 40, 64, 112, 256
    };

    int failures = q5_route_policy();

    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kK6144RouteInteriors});

    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kK17408RouteInteriors});

    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd 4096 preserve", WeightFormat::Q5G64F16S,
        ShapeCase{4096, 4096, 421U, kK4096RouteStarts, kK4096RouteInteriors});

    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd 4096 preserve", WeightFormat::Q5G64F16S,
        ShapeCase{4096, 12288, 431U, kK4096RouteStarts, kK4096RouteInteriors});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
