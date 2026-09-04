#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // One public numerical case begins each materially distinct W8 implementation interval;
        // selected endpoints exercise exact-T and predicated tails without inspecting selectors.
        constexpr std::array<std::int32_t, 25> kTokenCases{
            1,   2,   6,   32,  33,  40,  41,  48,  49,  65,  81,  97,  129,
            193, 241, 256, 257, 265, 289, 321, 385, 449, 513, 560, 561,
        };
        int failures = run_profile(
            "LinearSwiGLU W8_A16",
            {QType::W8G32_F16S, 12288, 2048, 6144, 1601U, ActivationCompute::A16}, kTokenCases);

        // The eight DFlash2 decode extents cover every direct-small-T specialization used by the
        // product. The remaining cases cover each distinct non-small execution family.
        constexpr std::array<std::int32_t, 12> kDFlash2TokenCases{
            8, 16, 24, 32, 40, 48, 56, 64, 65, 96, 97, 1024,
        };
        failures +=
            run_profile("LinearSwiGLU W8_A16 DFlash2",
                        {QType::W8G32_F16S, 34816, 5120, 17408, 1603U, ActivationCompute::A16},
                        kDFlash2TokenCases);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU W8_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU W8_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
