from __future__ import annotations
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from typing import Literal, Optional
from .fixed_point import QFormat, mul_q, quantize
Mode = Literal["float", "fixed"]
Rounding = Literal["nearest_even", "trunc"]
X0Strategy = Literal["linear", "one", "magic", "magic_bf16", "magic_fp8"]
@dataclass(frozen=True)
class NewtonReciprocalConfig:
    mode: Mode = "float"
    max_iter: int = 10
    tol_abs: float = 0.0
    tol_rel: float = 0.0
    rounding: Rounding = "nearest_even"
    frac_bits: int = 24
    x0_strategy: X0Strategy = "linear"
    reference_decimal_prec: int = 80
@dataclass(frozen=True)
class SimulationPoint:
    k: int
    b: float
    b_norm: float
    scale_e: int
    x: float
    residual: float
    err_abs: float
    err_rel: float
    x_int: Optional[int] = None
def _as_decimal(b: float | int | str | Decimal) -> Decimal:
    if isinstance(b, Decimal):
        return b
    if isinstance(b, (int,)):
        return Decimal(b)
    if isinstance(b, float):
        return Decimal(repr(b))
    if isinstance(b, str):
        return Decimal(b)
    raise TypeError(f"Unsupported type for b: {type(b)}")
def _normalize_positive_b(b: float) -> tuple[float, int]:
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("b must be positive and finite for normalization")
    e = 0
    m = b
    while m >= 1.0:
        m *= 0.5
        e += 1
    while m < 0.5:
        m *= 2.0
        e -= 1
    return m, e
def _x0_linear(m: float) -> float:
    return (48.0 / 17.0) - (32.0 / 17.0) * m
_MAGIC_STRATEGIES = {"magic", "magic_bf16", "magic_fp8"}
_MAGIC_FMT_MAP: dict[str, str] = {
    "magic": "fp32",
    "magic_bf16": "bf16",
    "magic_fp8": "fp8_e4m3",
}
def _initial_x0(m: float, strategy: X0Strategy, *, b_abs: float | None = None) -> float:
    if strategy == "linear":
        return _x0_linear(m)
    if strategy == "one":
        return 1.0
    if strategy in _MAGIC_STRATEGIES:
        from .basic import magic_constant_reciprocal_seed
        if b_abs is None:
            raise ValueError("b_abs must be provided for magic strategies")
        fmt = _MAGIC_FMT_MAP[strategy]
        return magic_constant_reciprocal_seed(b_abs, fmt=fmt)
    raise ValueError(f"Unsupported x0_strategy: {strategy}")
def _reference_reciprocal_decimal(b_dec: Decimal, prec: int) -> Decimal:
    ctx = getcontext().copy()
    ctx.prec = prec
    with localcontext(ctx):
        return Decimal(1) / b_dec
def simulate_newton_reciprocal(
    b: float | int | str | Decimal,
    cfg: NewtonReciprocalConfig = NewtonReciprocalConfig(),
) -> list[SimulationPoint]:
    b_dec = _as_decimal(b)
    if b_dec == 0:
        raise ZeroDivisionError("b must be non-zero")
    b_float = float(b_dec)
    if not math.isfinite(b_float):
        raise ValueError("b must be finite")
    sign = -1.0 if b_float < 0 else 1.0
    b_abs = abs(b_float)
    m, e = _normalize_positive_b(b_abs)
    ref_dec = _reference_reciprocal_decimal(b_dec, cfg.reference_decimal_prec)
    ref = float(ref_dec)
    if cfg.x0_strategy in _MAGIC_STRATEGIES:
        x_recip_b = _initial_x0(m, cfg.x0_strategy, b_abs=b_abs)
        x = math.ldexp(x_recip_b, e)
    else:
        x = _initial_x0(m, cfg.x0_strategy)
    q: Optional[QFormat] = None
    m_q: Optional[int] = None
    x_q: Optional[int] = None
    if cfg.mode == "fixed":
        if cfg.frac_bits < 0:
            raise ValueError("frac_bits must be >= 0")
        q = QFormat(cfg.frac_bits)
        m_q = quantize(m, q, cfg.rounding)
        x_q = quantize(x, q, cfg.rounding)
    points: list[SimulationPoint] = []
    for k in range(cfg.max_iter + 1):
        if cfg.mode == "float":
            x_real_m = x
        else:
            assert q is not None and x_q is not None
            x_real_m = q.to_float(x_q)
        x_real_b = sign * math.ldexp(x_real_m, -e)
        ctx = getcontext().copy()
        ctx.prec = cfg.reference_decimal_prec
        with localcontext(ctx):
            if cfg.mode == "fixed":
                assert x_q is not None
                x_m_dec = Decimal(x_q) / Decimal(1 << cfg.frac_bits)
                if e >= 0:
                    scale_dec = Decimal(1) / Decimal(1 << e)
                else:
                    scale_dec = Decimal(1 << (-e))
                x_dec = (Decimal(-1) if sign < 0 else Decimal(1)) * x_m_dec * scale_dec
            else:
                x_dec = Decimal(repr(x_real_b))
            residual_dec = Decimal(1) - b_dec * x_dec
            err_abs_dec = abs(x_dec - ref_dec)
            err_rel_dec = err_abs_dec / abs(ref_dec) if ref_dec != 0 else Decimal("Infinity")
        residual = float(residual_dec)
        err_abs = float(err_abs_dec)
        err_rel = float(err_rel_dec)
        points.append(
            SimulationPoint(
                k=k,
                b=b_float,
                b_norm=m,
                scale_e=e,
                x=x_real_b,
                residual=residual,
                err_abs=err_abs,
                err_rel=err_rel,
                x_int=x_q if cfg.mode == "fixed" else None,
            )
        )
        if cfg.tol_abs > 0.0 and err_abs <= cfg.tol_abs:
            break
        if cfg.tol_rel > 0.0 and err_rel <= cfg.tol_rel:
            break
        if k == cfg.max_iter:
            break
        if cfg.mode == "float":
            x = x * (2.0 - m * x)
        else:
            assert q is not None and m_q is not None and x_q is not None
            two_q = 2 << cfg.frac_bits
            mx_q = mul_q(m_q, x_q, cfg.frac_bits, cfg.rounding)
            term_q = two_q - mx_q
            x_q = mul_q(x_q, term_q, cfg.frac_bits, cfg.rounding)
    return points
