#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Programmatic dependent launch requires compute capability 9.0+. On older
// devices the launch attribute is dropped, stream order fully serializes the
// producer/consumer pair, and the device helpers below become no-ops:
// correctness is preserved, only the launch-overlap optimization is lost.
inline bool device_supports_pdl() {
    static const bool supported = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return false; }
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess) {
            return false;
        }
        return major >= 9;
    }();
    return supported;
}

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = nullptr;
    config.numAttrs         = 0;

    cudaLaunchAttribute attribute{};
    if (device_supports_pdl()) {
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs    = &attribute;
        config.numAttrs = 1;
    }

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void trigger_dependents() { cudaTriggerProgrammaticLaunchCompletion(); }

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() { cudaGridDependencySynchronize(); }
#else
__device__ __forceinline__ void trigger_dependents() {}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {}
#endif

} // namespace ninfer::pdl
