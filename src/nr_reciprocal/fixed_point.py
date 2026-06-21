from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class QFormat:
    frac_bits: int
    def scale(self) -> int:
        return 1 << self.frac_bits
    def to_int_trunc(self, value: float) -> int:
        return int(value * self.scale())
    def to_int_nearest_even(self, value: float) -> int:
        scaled = value * self.scale()
        if scaled >= 0:
            return _round_half_even_positive(scaled)
        return -_round_half_even_positive(-scaled)
    def to_float(self, int_value: int) -> float:
        return int_value / self.scale()
def _round_half_even_positive(x: float) -> int:
    i = int(x)
    frac = x - i
    if frac > 0.5:
        return i + 1
    if frac < 0.5:
        return i
    return i + (i & 1)
def quantize(value: float, q: QFormat, rounding: str) -> int:
    if rounding == "trunc":
        return q.to_int_trunc(value)
    if rounding == "nearest_even":
        return q.to_int_nearest_even(value)
    raise ValueError(f"Unsupported rounding: {rounding}")
def mul_q(a: int, b: int, frac_bits: int, rounding: str) -> int:
    prod = a * b
    if rounding == "trunc":
        return prod >> frac_bits
    if rounding != "nearest_even":
        raise ValueError(f"Unsupported rounding: {rounding}")
    sign = -1 if prod < 0 else 1
    prod_abs = -prod if prod < 0 else prod
    mask = (1 << frac_bits) - 1
    r = prod_abs & mask
    q = prod_abs >> frac_bits
    half = 1 << (frac_bits - 1) if frac_bits > 0 else 0
    if frac_bits == 0:
        return sign * prod_abs
    if r > half:
        q += 1
    elif r == half:
        q += (q & 1)
    return sign * q
