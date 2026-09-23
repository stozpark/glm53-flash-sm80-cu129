#!/usr/bin/env python3
"""CPU-only reference validation for the SM80 E4M3FN software encoder."""

from __future__ import annotations

import math
import random
import struct


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bits_from_f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def to_f32(x: float) -> float:
    return f32_from_bits(bits_from_f32(x))


def decode_positive_e4m3fn(code: int) -> float:
    assert 0 <= code <= 0x7E
    exp = (code >> 3) & 0xF
    mant = code & 0x7
    if exp == 0:
        return mant * (2.0 ** -9)
    return (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


POSITIVE = [(code, decode_positive_e4m3fn(code)) for code in range(0x7F)]


def reference_encode(x: float) -> int:
    """Independent nearest-value RNE + satfinite reference."""
    b = bits_from_f32(x)
    sign = (b >> 24) & 0x80
    magbits = b & 0x7FFFFFFF
    if magbits > 0x7F800000:
        return sign | 0x7F
    if magbits == 0x7F800000:
        return sign | 0x7E

    mag = abs(f32_from_bits(magbits))
    if mag >= 448.0:
        return sign | 0x7E

    best_code = 0
    best_dist = float("inf")
    for code, value in POSITIVE:
        d = abs(mag - value)
        if d < best_dist:
            best_dist = d
            best_code = code
        elif d == best_dist:
            # Round-to-nearest-even: the retained low significand bit is the
            # encoding LSB, including exponent-boundary carry cases.
            if (code & 1) == 0 and (best_code & 1) != 0:
                best_code = code
    return sign | best_code


def ported_triton_encoder(x: float) -> int:
    """Integer-equivalent translation of fp8_sm80._f32_to_e4m3fn_u8."""
    fbits = bits_from_f32(x)
    sign8 = (fbits >> 24) & 0x80
    mag = fbits & 0x7FFFFFFF
    is_nan = mag > 0x7F800000

    exp = (mag >> 23) - 127
    mant = (mag & 0x7FFFFF) | 0x800000

    shift = min(20 + max(-6 - exp, 0), 30)
    keep = mant >> shift
    rest = mant & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    round_up = rest > half or (rest == half and (keep & 1) == 1)
    if round_up:
        keep += 1

    val_normal = ((exp + 6) << 3) + keep
    val = val_normal if exp >= -6 else keep
    val = min(val, 0x7E)
    if is_nan:
        val = 0x7F
    return (val & 0x7F) | sign8


def next_f32_bits(bits: int, direction: int) -> int:
    assert 0 <= bits < 0x7F800000
    out = bits + direction
    assert 0 <= out <= 0x7F800000
    return out


def main() -> None:
    samples: set[int] = set()

    # Exact E4M3 values plus every rounding boundary and the adjacent float32s.
    vals = [v for _, v in POSITIVE]
    for v in vals:
        samples.add(bits_from_f32(v))
    for a, b in zip(vals, vals[1:]):
        midpoint = to_f32((a + b) / 2.0)
        mb = bits_from_f32(midpoint)
        samples.add(mb)
        if 0 < mb < 0x7F800000:
            samples.add(next_f32_bits(mb, -1))
            samples.add(next_f32_bits(mb, +1))

    # Special/saturation/subnormal boundaries.
    for x in (
        0.0,
        -0.0,
        1e-45,
        -1e-45,
        2.0**-10,
        -(2.0**-10),
        448.0,
        -448.0,
        449.0,
        -449.0,
        float("inf"),
        -float("inf"),
        float("nan"),
    ):
        samples.add(bits_from_f32(x))

    # Deterministic broad sampling of finite float32 bit patterns.
    rng = random.Random(530080)
    for _ in range(100_000):
        b = rng.getrandbits(32)
        # Keep NaNs too; all payloads must map to canonical signed NaN.
        samples.add(b)

    for bits in samples:
        x = f32_from_bits(bits)
        got = ported_triton_encoder(x)
        want = reference_encode(x)
        assert got == want, (
            f"E4M3 mismatch bits=0x{bits:08x} x={x!r} "
            f"got=0x{got:02x} want=0x{want:02x}"
        )

    print(f"SM80_E4M3FN_REFERENCE=PASS samples={len(samples)}")


if __name__ == "__main__":
    main()
