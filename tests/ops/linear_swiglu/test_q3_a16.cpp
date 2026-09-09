#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 1> kTokenCases{
            4,
        };

        const int failures = run_profile(
            "LinearSwiGLU Q3_A16",
            {
                QType::Q3G64_F16S,
                34816,
                5120,
                17408,
                1403U,
                ActivationCompute::A16
            },
            kTokenCases);

        std::cout
            << (failures == 0 ? "OK" : "FAIL")
            << " LinearSwiGLU Q3_A16 correctness\n";

        return failures == 0 ? 0 : 1;

    } catch (const std::exception& error) {

        std::cerr
            << "LinearSwiGLU Q3_A16 test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
