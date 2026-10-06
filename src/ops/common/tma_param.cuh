#pragma once

// Kernel-parameter carrier for a CUtensorMap.
//
// CUDA declares CUtensorMap with alignas(128). MSVC refuses by-value function parameters whose
// alignment exceeds 64 bytes (C2719), and nvcc's host stub passes __grid_constant__ kernel
// parameters by value, so a CUtensorMap cannot be a kernel parameter on Windows. PTX only requires
// the tensor-map address to be 64-byte aligned in param, const or global space, so the kernels
// carry this 64-byte-aligned byte copy and reinterpret its address when issuing TMA loads.

#include <cuda.h>

#include <cstring>

namespace ninfer::ops::detail {

struct alignas(64) TmaMapParam {
    unsigned long long opaque[sizeof(CUtensorMap) / sizeof(unsigned long long)];

    TmaMapParam() = default;
    TmaMapParam(const CUtensorMap& map) noexcept { // NOLINT(google-explicit-constructor)
        std::memcpy(opaque, &map, sizeof(CUtensorMap));
    }

    [[nodiscard]] __host__ __device__ const CUtensorMap* map() const noexcept {
        return reinterpret_cast<const CUtensorMap*>(this);
    }
};

static_assert(sizeof(TmaMapParam) == sizeof(CUtensorMap));
static_assert(alignof(TmaMapParam) == 64);

} // namespace ninfer::ops::detail
