#pragma once

// Portable 128-bit helpers for the few host-side computations that need headroom beyond 64 bits.
// GCC/Clang provide unsigned __int128; MSVC does not, so this header implements the handful of
// operations NInfer uses (saturating products, cross-multiplication compare, exact decimal
// fraction scaling) on a two-limb representation.

#include <cstdint>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#endif

namespace ninfer::wide {

struct Uint128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    [[nodiscard]] friend constexpr bool operator==(Uint128, Uint128) noexcept = default;
};

[[nodiscard]] constexpr bool less(Uint128 left, Uint128 right) noexcept {
    return left.hi < right.hi || (left.hi == right.hi && left.lo < right.lo);
}

[[nodiscard]] inline Uint128 mul(std::uint64_t left, std::uint64_t right) noexcept {
#if defined(__SIZEOF_INT128__)
    const unsigned __int128 product = static_cast<unsigned __int128>(left) * right;
    return {static_cast<std::uint64_t>(product), static_cast<std::uint64_t>(product >> 64U)};
#elif defined(_MSC_VER)
    Uint128 result;
    result.lo = _umul128(left, right, &result.hi);
    return result;
#else
    const std::uint64_t a_lo = left & 0xffffffffULL, a_hi = left >> 32U;
    const std::uint64_t b_lo = right & 0xffffffffULL, b_hi = right >> 32U;
    const std::uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
    const std::uint64_t middle = (p0 >> 32U) + (p1 & 0xffffffffULL) + (p2 & 0xffffffffULL);
    Uint128 result;
    result.lo = (middle << 32U) | (p0 & 0xffffffffULL);
    result.hi = p3 + (p1 >> 32U) + (p2 >> 32U) + (middle >> 32U);
    return result;
#endif
}

// Low 128 bits of value * factor; callers bound the operands so the product fits.
[[nodiscard]] constexpr Uint128 mul_small(Uint128 value, std::uint32_t factor) noexcept {
    const std::uint64_t lo_lo = (value.lo & 0xffffffffULL) * factor;
    const std::uint64_t lo_hi = (value.lo >> 32U) * factor + (lo_lo >> 32U);
    Uint128 result;
    result.lo = (lo_hi << 32U) | (lo_lo & 0xffffffffULL);
    result.hi = value.hi * factor + (lo_hi >> 32U);
    return result;
}

[[nodiscard]] constexpr Uint128 add_small(Uint128 value, std::uint64_t addend) noexcept {
    Uint128 result;
    result.lo = value.lo + addend;
    result.hi = value.hi + (result.lo < value.lo ? 1U : 0U);
    return result;
}

[[nodiscard]] constexpr Uint128 shift_left(Uint128 value, unsigned bits) noexcept {
    if (bits == 0) { return value; }
    if (bits >= 64) { return {0, value.lo << (bits - 64U)}; }
    return {value.lo << bits, (value.hi << bits) | (value.lo >> (64U - bits))};
}

[[nodiscard]] constexpr Uint128 shift_right(Uint128 value, unsigned bits) noexcept {
    if (bits == 0) { return value; }
    if (bits >= 64) { return {value.hi >> (bits - 64U), 0}; }
    return {(value.lo >> bits) | (value.hi << (64U - bits)), value.hi >> bits};
}

struct DivisionResult {
    Uint128 quotient;
    std::uint64_t remainder = 0;
};

// Restoring long division by a 64-bit divisor; used only in configuration parsing.
[[nodiscard]] constexpr DivisionResult divide(Uint128 value, std::uint64_t divisor) noexcept {
    DivisionResult result;
    std::uint64_t remainder = 0;
    for (int bit = 127; bit >= 0; --bit) {
        const bool overflow = (remainder >> 63U) != 0;
        remainder <<= 1U;
        const std::uint64_t source = bit >= 64 ? value.hi : value.lo;
        remainder |= (source >> (bit & 63)) & 1U;
        if (overflow || remainder >= divisor) {
            remainder -= divisor;
            if (bit >= 64) {
                result.quotient.hi |= std::uint64_t{1} << (bit - 64);
            } else {
                result.quotient.lo |= std::uint64_t{1} << bit;
            }
        }
    }
    result.remainder = remainder;
    return result;
}

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

[[nodiscard]] inline std::uint64_t saturating_mul(std::uint64_t left,
                                                  std::uint64_t right) noexcept {
    const Uint128 product = mul(left, right);
    return product.hi != 0 ? std::numeric_limits<std::uint64_t>::max() : product.lo;
}

// left_a * left_b compared with right_a * right_b without overflow: <0, 0, >0.
[[nodiscard]] inline int compare_products(std::uint64_t left_a, std::uint64_t left_b,
                                          std::uint64_t right_a, std::uint64_t right_b) noexcept {
    const Uint128 left  = mul(left_a, left_b);
    const Uint128 right = mul(right_a, right_b);
    if (less(left, right)) { return -1; }
    if (less(right, left)) { return 1; }
    return 0;
}

} // namespace ninfer::wide
