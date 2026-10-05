#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N16384K5120;
// Same tile as the materialized A4 route for T <= 64 (nvfp4_gdn_input_a4.cu): 32 token rows, 64
// parent rows, 256 threads. Every token of a parent row lives in one CTA, so the causal
// convolution over tokens needs no second kernel.
using Schedule = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;

// Collective epilogue for the exact B=1, T=kNvfp4GdnRecordFusedTokens Record route. It reproduces
// the materialized route bit for bit: the represented projected value is the BF16 rounding of the
// scaled accumulator, the record plane receives that value for every token, and the convolution,
// SiLU, masking and Q/K/V/Z split come from the shared GdnConvOutput adapter.
template <int Tokens>
struct Nvfp4GdnConvRecordEpilogue {
    GdnConvOutput<Tokens, NoHistoryPublish> conv;
    __nv_bfloat16* record;

    template <class S>
    static constexpr int kSharedBytes = S::kBlockTokens * (S::kBlockRows + 8) * 2;

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(Output, unsigned char* scratch, float (&acc)[S::kMmaTokens][S::kMmaRows][4],
                int row_begin, int token_begin, int, int token_end) const {
        static_assert(S::kBlockTokens >= Tokens);
        constexpr int stride = S::kBlockRows + 8;
        auto* tile           = reinterpret_cast<__nv_bfloat16*>(scratch);
        const int lane       = static_cast<int>(threadIdx.x) & 31;
        const int warp       = static_cast<int>(threadIdx.x) >> 5;
        const int warp_m     = warp / S::kWarpsRows;
        const int warp_n     = warp - warp_m * S::kWarpsRows;
        const int ar         = lane >> 2;
        const int ac         = 2 * (lane & 3);

        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int t0 = warp_m * S::kWarpTokens + mt * 16 + ar;
            const int t1 = t0 + 8;
#pragma unroll
            for (int mr = 0; mr < S::kMmaRows; ++mr) {
                const int local = warp_n * S::kWarpRows + mr * 8 + ac;
                *reinterpret_cast<__nv_bfloat162*>(tile + t0 * stride + local) =
                    __floats2bfloat162_rn(acc[mt][mr][0], acc[mt][mr][1]);
                *reinterpret_cast<__nv_bfloat162*>(tile + t1 * stride + local) =
                    __floats2bfloat162_rn(acc[mt][mr][2], acc[mt][mr][3]);
            }
        }
        __syncthreads();

        // The launcher guarantees one token tile holding exactly Tokens tokens.
        (void)token_begin;
        (void)token_end;
        for (int local = static_cast<int>(threadIdx.x); local < S::kBlockRows;
             local += S::kThreads) {
            const int row = row_begin + local;
            float projected[Tokens];
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const __nv_bfloat16 word = tile[token * stride + local];
                projected[token]         = __bfloat162float(word);
                if (row < kGdnChannels) {
                    record[static_cast<std::int64_t>(token) * kGdnChannels + row] = word;
                }
            }
            conv.store_row(row, projected);
        }
    }
};

} // namespace

void nvfp4_gdn_record_fused_a4_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, Nvfp4A4Workspace workspace,
                                      cudaStream_t stream) {
    constexpr int kTokens     = kNvfp4GdnRecordFusedTokens;
    const std::int32_t tokens = x.ne[1];
    if (tokens != kTokens || x.ne[2] != 1 || weight.n != Geometry::kOutputRows ||
        weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 fused gdn record: route requires B=1 and T=8");
    }
    launch_nvfp4_a4_quantize(x, weight, workspace, Nvfp4ScaleLayout::RowMajor, stream);
    const Nvfp4GdnConvRecordEpilogue<kTokens> epilogue{
        make_gdn_conv_output<kTokens, NoHistoryPublish>(conv_weight, conv_states, valid_columns,
                                                        initial_slot, query, key, value, z,
                                                        NoHistoryPublish{}),
        static_cast<__nv_bfloat16*>(conv_record.data)};
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(conv_record.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        epilogue, stream);
}

} // namespace ninfer::ops::detail
