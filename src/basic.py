from __future__ import annotations
import math
import struct
from dataclasses import dataclass
MAGIC_CONST_FP32: int = 0x7EF311C3
MAGIC_CONST_BF16: int = 0x7EF3
MAGIC_CONST_FP8_E4M3: int = 0x70
PRECISION_THRESHOLDS: dict[str, float] = {
    "fp32": 2.0 ** (-24),
    "bf16": 2.0 ** (-7),
    "fp8":  2.0 ** (-3),
}
def threshold_for_precision(name: str) -> float:
    key = name.strip().lower()
    if key not in PRECISION_THRESHOLDS:
        raise ValueError(
            f"Unknown precision: {name!r}.  "
            f"Supported: {', '.join(sorted(PRECISION_THRESHOLDS))}"
        )
    return PRECISION_THRESHOLDS[key]
def _float32_to_uint32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]
def _uint32_to_float32(n: int) -> float:
    return struct.unpack("<f", struct.pack("<I", n & 0xFFFFFFFF))[0]
def _float_to_bf16_int(f: float) -> int:
    return (_float32_to_uint32(f) >> 16) & 0xFFFF
def _bf16_int_to_float(n: int) -> float:
    return _uint32_to_float32((n & 0xFFFF) << 16)
def _float_to_fp8e4m3_int(f: float) -> int:
    if f <= 0.0:
        return 0
    if f < 2.0 ** (-6):
        return 0x08
    if f > 240.0:
        return 0x7F
    e_unbiased = math.floor(math.log2(f))
    e_biased = max(1, min(14, e_unbiased + 7))
    significand = f / (2.0 ** (e_biased - 7))
    frac = max(0, min(7, int((significand - 1.0) * 8)))
    return ((e_biased & 0xF) << 3) | (frac & 0x7)
def _fp8e4m3_int_to_float(n: int) -> float:
    e_biased = (n >> 3) & 0xF
    frac = n & 0x7
    if e_biased == 0:
        return 0.0
    return (2.0 ** (e_biased - 7)) * (1.0 + frac / 8.0)
def float_quantize(x: float, fmt: str) -> float:
    key = fmt.strip().lower()
    if key == "fp32":
        return struct.unpack("<f", struct.pack("<f", x))[0]
    if key == "bf16":
        return _bf16_int_to_float(_float_to_bf16_int(x))
    if key in ("fp8", "fp8_e4m3", "fp8e4m3"):
        return _fp8e4m3_int_to_float(_float_to_fp8e4m3_int(x))
    raise ValueError(f"Unsupported quantization format: {fmt!r}")
def magic_constant_reciprocal_seed(b: float, fmt: str = "fp32") -> float:
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("b must be positive and finite")
    key = fmt.strip().lower()
    if key == "fp32":
        b_int = _float32_to_uint32(struct.unpack("<f", struct.pack("<f", b))[0])
        return float(_uint32_to_float32((MAGIC_CONST_FP32 - b_int) & 0xFFFFFFFF))
    if key == "bf16":
        b_int = _float_to_bf16_int(b)
        return float(_bf16_int_to_float((MAGIC_CONST_BF16 - b_int) & 0xFFFF))
    if key in ("fp8", "fp8_e4m3", "fp8e4m3"):
        b_int = _float_to_fp8e4m3_int(b)
        return float(_fp8e4m3_int_to_float((MAGIC_CONST_FP8_E4M3 - b_int) & 0xFF))
    raise ValueError(f"Unsupported format: {fmt!r}")
def _x0_linear(m: float) -> float:
    return 2.823529411764706 - 1.8823529411764706 * m
def hardware_initial_guess(b: float, *, strategy: str = "linear") -> float:
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("b must be > 0 and finite")
    key = strategy.strip().lower()
    if key == "magic":
        return magic_constant_reciprocal_seed(b, "fp32")
    if key == "magic_bf16":
        return magic_constant_reciprocal_seed(b, "bf16")
    if key in ("magic_fp8", "magic_fp8_e4m3"):
        return magic_constant_reciprocal_seed(b, "fp8_e4m3")
    m, e = math.frexp(b)
    if key == "linear":
        x_m = _x0_linear(m)
    elif key in ("const", "constant", "one"):
        x_m = 1.0
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}.  "
            "Use 'linear', 'const', 'magic', 'magic_bf16', or 'magic_fp8'."
        )
    return math.ldexp(x_m, -e)
@dataclass(frozen=True)
class NewtonResult:
    x: float
    iterations: int
    errors: list[float]
    converged: bool
def newton_reciprocal_result(
    b: float,
    *,
    threshold: float = 1e-12,
    max_iter: int = 50,
    x0: float | None = None,
    precision: str | None = None,
    x0_strategy: str = "linear",
    float_quantize_fmt: str | None = None,
) -> NewtonResult:
    if not isinstance(b, (float, int)):
        raise TypeError("b must be numeric")
    b = float(b)
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("b must be > 0 and finite")
    if precision is not None:
        threshold = threshold_for_precision(precision)
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")
    if max_iter < 0:
        raise ValueError("max_iter must be >= 0")
    x_exact = 1.0 / b
    x = hardware_initial_guess(b, strategy=x0_strategy) if x0 is None else float(x0)
    errors: list[float] = []
    for _ in range(max_iter):
        x = x * (2.0 - b * x)
        if float_quantize_fmt is not None:
            x = float_quantize(x, float_quantize_fmt)
        err = abs(x - x_exact) / abs(x_exact)
        errors.append(err)
        if err < threshold:
            break
    return NewtonResult(
        x=x,
        iterations=len(errors),
        errors=errors,
        converged=bool(errors) and errors[-1] < threshold,
    )
def newton_reciprocal(
    b: float,
    *,
    threshold: float = 1e-12,
    max_iter: int = 50,
    x0: float | None = None,
    precision: str | None = None,
    x0_strategy: str = "linear",
    float_quantize_fmt: str | None = None,
) -> tuple[float, int, list[float]]:
    r = newton_reciprocal_result(
        b, threshold=threshold, max_iter=max_iter, x0=x0,
        precision=precision, x0_strategy=x0_strategy,
        float_quantize_fmt=float_quantize_fmt,
    )
    return r.x, r.iterations, r.errors
