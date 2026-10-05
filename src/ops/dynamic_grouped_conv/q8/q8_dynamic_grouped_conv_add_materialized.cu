#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_schedule.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

using Launch = Q8Launch;

template <int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry    = Q8LinearGeometry<kRows, InputRows>;
    using Schedule    = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                                Q8ScaleAccess::Shared, Activation>;
    const int columns = x.ne[1];
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&tiled_projection<C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto attention = make_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp       = make_launchers<17408>(std::make_index_sequence<11>{});

// Q4 projection into the same materialized BF16 plane (draft Q4 experiment). The sliced-K
// schedules are K-static; the MMA schedules tile a runtime K for wide column counts.
template <int Capacity, int StaticK>
void q4_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, StaticK, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kRows}, LinearIdentityEpilogue{},
        stream);
}

template <class Schedule>
void q4_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma<Schedule>(q4_linear_operands(x, weight),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kRows},
                                LinearIdentityEpilogue{}, stream);
}

template <int StaticK>
void q4_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using MmaR32T32  = Q4A16MmaSchedule<32, 32, 64, 16, 16, 3, 2, Q4MmaFragmentPipeline::Serial,
                                        Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
    using MmaR32T64  = Q4A16MmaSchedule<32, 64, 64, 16, 32, 3, 2, Q4MmaFragmentPipeline::Serial,
                                        Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
    using MmaR64T128 = Q4A16MmaSchedule<64, 128, 64, 64, 32, 2, 1, Q4MmaFragmentPipeline::Serial,
                                        Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
    const int tokens = x.ne[1];
    if (tokens <= 4) return q4_sliced<4, StaticK>(x, weight, out, stream);
    if (tokens <= 8) return q4_sliced<8, StaticK>(x, weight, out, stream);
    if (tokens <= 16) return q4_sliced<16, StaticK>(x, weight, out, stream);
    if (tokens <= 24) return q4_sliced<24, StaticK>(x, weight, out, stream);
    if (tokens <= 32) return q4_sliced<32, StaticK>(x, weight, out, stream);
    if (tokens <= 96) return q4_mma<MmaR32T32>(x, weight, out, stream);
    if (tokens <= 192) return q4_mma<MmaR32T64>(x, weight, out, stream);
    return q4_mma<MmaR64T128>(x, weight, out, stream);
}

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    if (weight.qtype == QType::Q4_G64_FP16) {
        if (x.ne[0] == 4096) {
            q4_projection<4096>(flat, weight, result, stream);
        } else {
            q4_projection<17408>(flat, weight, result, stream);
        }
    } else {
        switch (schedule) {
        case Q8DynamicConvAddSchedule::TiledMma: {
            const auto& launchers = x.ne[0] == 4096 ? attention : mlp;
            launchers[(tokens - 1) / 8](flat, weight, result, stream);
            break;
        }
        case Q8DynamicConvAddSchedule::MmaK128:
            launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
            break;
        }
    }
    const dim3 grid((kRows + 255) / 256, tokens);
    finish_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                            static_cast<const __nv_bfloat16*>(base.data),
                                            static_cast<const __nv_bfloat16*>(delta.data),
                                            static_cast<__nv_bfloat16*>(residual.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
