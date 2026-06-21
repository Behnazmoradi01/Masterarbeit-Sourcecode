from __future__ import annotations
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from typing import Optional
from basic import float_quantize, hardware_initial_guess
@dataclass(frozen=True)
class NewtonReciprocalConfig:
    max_iter: int = 10
    tol_rel: float = 0.0
    x0_strategy: str = "linear"
    reference_decimal_prec: int = 80
    float_quantize_fmt: Optional[str] = None
@dataclass(frozen=True)
class SimulationPoint:
    k: int
    b: float
    x: float
    residual: float
    err_abs: float
    err_rel: float
def _as_decimal(b) -> Decimal:
    if isinstance(b, Decimal):
        return b
    if isinstance(b, int):
        return Decimal(b)
    if isinstance(b, float):
        return Decimal(repr(b))
    if isinstance(b, str):
        return Decimal(b)
    raise TypeError(f"Cannot convert {type(b)} to Decimal")
def simulate_newton_reciprocal(
    b,
    cfg: NewtonReciprocalConfig = NewtonReciprocalConfig(),
) -> list[SimulationPoint]:
    b_dec = _as_decimal(b)
    b_float = float(b_dec)
    if not math.isfinite(b_float):
        raise ValueError("b must be finite")
    if b_float <= 0.0:
        raise ValueError("b must be > 0")
    prec = cfg.reference_decimal_prec
    ctx = getcontext().copy()
    ctx.prec = prec
    with localcontext(ctx):
        ref_dec = Decimal(1) / b_dec
    x = hardware_initial_guess(b_float, strategy=cfg.x0_strategy)
    points: list[SimulationPoint] = []
    for k in range(cfg.max_iter + 1):
        with localcontext(ctx):
            x_dec = Decimal(repr(x))
            residual = float(Decimal(1) - b_dec * x_dec)
            err_abs = float(abs(x_dec - ref_dec))
            err_rel = (
                float(abs(x_dec - ref_dec) / abs(ref_dec))
                if ref_dec != 0 else float("inf")
            )
        points.append(SimulationPoint(
            k=k, b=b_float, x=x,
            residual=residual, err_abs=err_abs, err_rel=err_rel,
        ))
        if cfg.tol_rel > 0.0 and err_rel <= cfg.tol_rel:
            break
        if k == cfg.max_iter:
            break
        x = x * (2.0 - b_float * x)
        if cfg.float_quantize_fmt is not None:
            x = float_quantize(x, cfg.float_quantize_fmt)
    return points
