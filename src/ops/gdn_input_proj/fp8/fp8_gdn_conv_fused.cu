#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int Tokens, class Publish>
struct Fp8GdnConvEpilogue {
    [[maybe_unused]] static constexpr int kRowTokens = Tokens;

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};

using Geometry = Fp8N16384K5120;

// Four neighboring lanes own a row's eight projected tokens. Exchange their
// rounded BF16 values before applying the four-tap convolution, retaining the
// materialized route's casts and MMA reduction order.
template <class Publish>
struct Fp8GdnConvMmaOutput {
    GdnConvOutput<8, Publish> output;

    __device__ __forceinline__ void store_pair(int row, int token, float even, float odd) const {
        const auto a_bits = __float2bfloat16_rn(even);
        const auto b_bits = __float2bfloat16_rn(odd);
        if (row >= kGdnChannels) {
            output.z[static_cast<std::int64_t>(token) * kGdnZRows + row - kGdnChannels] = a_bits;
            output.z[static_cast<std::int64_t>(token + 1) * kGdnZRows + row - kGdnChannels] =
                b_bits;
            return;
        }
        const auto& c = output.conv;
        const float a = __bfloat162float(a_bits), b = __bfloat162float(b_bits);
        const int lane_in_row  = threadIdx.x & 3;
        const auto initial     = static_cast<std::int64_t>(c.initial_slots[0]) * kGdnChannels * 3;
        const float s0         = __bfloat162float(c.state_read[initial + row]);
        const float s1         = __bfloat162float(c.state_read[initial + kGdnChannels + row]);
        const float s2         = __bfloat162float(c.state_read[initial + 2LL * kGdnChannels + row]);
        const float prior_even = __shfl_up_sync(0xffffffffU, a, 1, 4);
        const float prior_odd  = __shfl_up_sync(0xffffffffU, b, 1, 4);
        const float older_odd  = __shfl_up_sync(0xffffffffU, b, 2, 4);
        const float h0         = lane_in_row == 0 ? s0 : lane_in_row == 1 ? s2 : older_odd;
        const float h1         = lane_in_row == 0 ? s1 : prior_even;
        const float h2         = lane_in_row == 0 ? s2 : prior_odd;
        const float w0         = __bfloat162float(c.conv_weight[row]);
        const float w1         = __bfloat162float(c.conv_weight[kGdnChannels + row]);
        const float w2         = __bfloat162float(c.conv_weight[2LL * kGdnChannels + row]);
        const float w3         = __bfloat162float(c.conv_weight[3LL * kGdnChannels + row]);
        int valid              = c.valid_columns == nullptr ? 8 : c.valid_columns[0];
        valid                  = valid < 0 ? 0 : valid > 8 ? 8 : valid;
        // A snapshot destination may alias this row's initial history slot.
        // Every lane must have loaded that history before any lane publishes.
        __syncwarp();
        if constexpr (std::is_same_v<Publish, RecordColumnPublish>) {
            // Preserve the materialized route's complete physical record plane.
            c.publish.publish(token, 0, row, h1, h2, a);
            c.publish.publish(token + 1, 0, row, h2, a, b);
        }
        auto* destination   = row < kGdnQueryRows                 ? c.query
                              : row < kGdnQueryRows + kGdnKeyRows ? c.key
                                                                  : c.value;
        const int rows      = row < kGdnQueryRows                 ? kGdnQueryRows
                              : row < kGdnQueryRows + kGdnKeyRows ? kGdnKeyRows
                                                                  : kGdnValueRows;
        const int local_row = row < kGdnQueryRows ? row
                              : row < kGdnQueryRows + kGdnKeyRows
                                  ? row - kGdnQueryRows
                                  : row - kGdnQueryRows - kGdnKeyRows;
        const auto convolve = [&](float p0, float p1, float p2, float p3) {
            float sum = fmaf(w0, p0, 0.0F);
            sum       = fmaf(w1, p1, sum);
            sum       = fmaf(w2, p2, sum);
            sum       = fmaf(w3, p3, sum);
            return __float2bfloat16_rn(silu(sum));
        };
        destination[static_cast<std::int64_t>(token) * rows + local_row] =
            token < valid ? convolve(h0, h1, h2, a) : __float2bfloat16_rn(0.0F);
        destination[static_cast<std::int64_t>(token + 1) * rows + local_row] =
            token + 1 < valid ? convolve(h1, h2, a, b) : __float2bfloat16_rn(0.0F);
        if constexpr (std::is_same_v<Publish, SnapshotHistoryPublish>) {
            if (token < valid) c.publish.publish(token, 0, row, h1, h2, a);
            if (token + 1 < valid) c.publish.publish(token + 1, 0, row, h2, a, b);
        }
    }

    __device__ __forceinline__ void store_fragment(int row_a, int row_b, int token,
                                                   float4 values) const {
        if (token >= 8) return;
        store_pair(row_a, token, values.x, values.y);
        store_pair(row_b, token, values.z, values.w);
    }
};

template <class Publish>
void launch_mma8(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                 const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
                 Tensor& query, Tensor& key, Tensor& value, Tensor& z, Publish publish,
                 cudaStream_t stream) {
    using Schedule =
        Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
    const auto output = make_gdn_conv_output<8>(conv_weight, conv_states, valid_columns,
                                                initial_slot, query, key, value, z, publish);
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, 5120>>(
        fp8_a16_operands(x, weight), Fp8GdnConvMmaOutput<Publish>{output}, LinearIdentityEpilogue{},
        stream);
}

using SnapshotLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using RecordLaunch   = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, cudaStream_t);

template <int ActiveTokens, class Publish>
void launch_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                    const Tensor& conv_states, const Tensor& valid_columns,
                    const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                    Tensor& z, Publish publish, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 2, 16, ActiveTokens, 1, Fp8SimtActivationAccess::SharedPhase,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    static_assert(Schedule::kBlockTokens == ActiveTokens);
    using Output = GdnConvOutput<ActiveTokens, Publish>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        fp8_a16_operands(x, weight),
        make_gdn_conv_output<ActiveTokens>(conv_weight, conv_states, valid_columns, initial_slot,
                                           query, key, value, z, publish),
        Fp8GdnConvEpilogue<ActiveTokens, Publish>{}, stream);
}

template <int ActiveTokens>
void launch_snapshot_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             cudaStream_t stream) {
    launch_small_t<ActiveTokens>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               kGdnChannels},
        stream);
}

template <int ActiveTokens>
void launch_record_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_small_t<ActiveTokens>(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                                 query, key, value, z,
                                 RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data),
                                                     kGdnChannels, ActiveTokens},
                                 stream);
}

void launch_snapshot_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                            Tensor& conv_states, const Tensor& valid_columns,
                            const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight),
        make_gdn_conv_output<1>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kGdnChannels}),
        Fp8GdnConvEpilogue<1, SnapshotHistoryPublish>{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLaunch, sizeof...(Offsets)>{
        &launch_snapshot_small_t<2 + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        &launch_record_small_t<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSnapshotLaunchers = make_snapshot_launchers(std::make_index_sequence<3 - 2 + 1>{});
constexpr auto kRecordLaunchers   = make_record_launchers(std::make_index_sequence<3 - 2 + 1>{});

} // namespace

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) {
    if (x.ne[2] == 1 && x.ne[1] == 8) {
        launch_mma8(
            x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kGdnChannels},
            stream);
        return;
    }
    if (x.ne[2] != 1 || x.ne[1] <= 0 || x.ne[1] > 3) {
        throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
    }
    if (x.ne[1] == 1) {
        launch_snapshot_decode(x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                               snapshot_base_slot, query, key, value, z, stream);
        return;
    }
    kSnapshotLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot, query,
        key, value, z, stream);
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[2] == 1 && x.ne[1] == 8) {
        launch_mma8(
            x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), kGdnChannels, 8},
            stream);
        return;
    }
    if (x.ne[2] != 1 || x.ne[1] < 2 || x.ne[1] > 3) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    kRecordLaunchers[static_cast<std::size_t>(x.ne[1] - 2)](
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record, query, key,
        value, z, stream);
}

} // namespace ninfer::ops::detail
