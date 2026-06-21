from __future__ import annotations
import math
from typing import Literal
PrecisionPreset = Literal["fp32", "bf16", "fp8"]
PRECISION_THRESHOLDS: dict[str, float] = {
    "fp32": 2.0 ** (-24),
    "bf16": 2.0 ** (-7),
    "fp8": 2.0 ** (-3),
}
def threshold_for_precision(name: str) -> float:
    key = name.strip().lower()
    if key not in PRECISION_THRESHOLDS:
        raise ValueError(
            f"Unknown precision preset: {name}. Supported: {', '.join(sorted(PRECISION_THRESHOLDS))}"
        )
    return PRECISION_THRESHOLDS[key]
def goldschmidt_binomial_divide(
    a: float,
    b: float,
    *,
    threshold: float = 1e-12,
    precision: PrecisionPreset | None = None,
    max_iter: int = 50,
    order: int = 3,
) -> tuple[float, int, list[float]]:
    if not isinstance(a, (float, int)):
        raise TypeError("a must be a float")
    if not isinstance(b, (float, int)):
        raise TypeError("b must be a float")
    a = float(a)
    b = float(b)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("a and b must be finite")
    if a <= 0.0:
        raise ValueError("a must be > 0")
    if b <= 0.0:
        raise ValueError("b must be > 0")
    if precision is not None:
        threshold = threshold_for_precision(precision)
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")
    if max_iter < 0:
        raise ValueError("max_iter must be >= 0")
    if order not in (2, 3):
        raise ValueError("order must be 2 or 3")
    q_exact = a / b
    m, e = math.frexp(b)
    a_k = math.ldexp(a, -e)
    b_k = m
    errors: list[float] = []
    for _ in range(max_iter):
        e_k = 1.0 - b_k  # error term: چقدر b_k از 1 فاصله داره
        e2 = e_k * e_k
        if order == 2:
            # بسط دوجمله‌ای: 1/(1-e) ≈ 1 + e + e²
            inv = 1.0 + e_k + e2
        else:
            e3 = e2 * e_k
            inv = 1.0 + e_k + e2 + e3  # مرتبه ۳ - همگرایی سریع‌تر
        a_k = a_k * inv
        b_k = b_k * inv
        q = a_k
        err = abs(q - q_exact) / abs(q_exact)
        errors.append(err)
        if err < threshold:
            break
    return a_k, len(errors), errors
