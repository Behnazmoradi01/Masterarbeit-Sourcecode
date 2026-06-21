from __future__ import annotations
import math
import struct
from dataclasses import dataclass
X0Strategy = str

# ثابت جادویی برای تخمین اولیه 1/b از روی IEEE 754 bit representation
# ایده: log2(b) ~ (int_bits(b) - bias) / 2^23  →  int_bits(1/b) ≈ 2*bias*2^23 - int_bits(b)
# این constant از بهینه‌سازی worst-case relative error روی [2^-126, 2^127] بدست اومده
MAGIC_CONST_FP32: int = 0x7EF311C3
# BF16 همون ۱۶ بیت بالای FP32 constant هست (mantissa کوتاه‌تره)
MAGIC_CONST_BF16: int = 0x7EF3
MAGIC_CONST_FP8_E4M3: int = 0x70  # TODO: با spec واقعی FP8 hardware مقایسه بشه
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
def _x0_linear_for_mantissa(m: float) -> float:
    # linear fit روی بازه [0.5, 1): a - c*m
    # ضرایب از minimize کردن max|x0 - 1/m| روی همین بازه حساب شدن
    a = 2.823529411764706
    c = 1.8823529411764706
    return a - c * m
def _x0_constant_for_mantissa(_: float) -> float:
    return 1.0
def _float32_to_uint32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]
def _uint32_to_float32(n: int) -> float:
    return struct.unpack("<f", struct.pack("<I", n & 0xFFFFFFFF))[0]
def _float_to_bf16_int(f: float) -> int:
    f32_int = _float32_to_uint32(f)
    return (f32_int >> 16) & 0xFFFF
def _bf16_int_to_float(n: int) -> float:
    return _uint32_to_float32((n & 0xFFFF) << 16)
def _float_to_fp8e4m3_int(f: float) -> int:
    if f <= 0.0:
        return 0
    if f < 2.0 ** (-6):
        return 0x08
    if f > 240.0:
        return 0x7F
    m, e = math.frexp(f)
    e_unbiased = e - 1
    e_biased = e_unbiased + 7
    if e_biased < 1:
        e_biased = 1
        e_unbiased = e_biased - 7
    if e_biased > 14:
        e_biased = 14
        e_unbiased = e_biased - 7
    significand = f / (2.0 ** e_unbiased)
    frac = int((significand - 1.0) * 8)
    frac = max(0, min(7, frac))
    return ((e_biased & 0xF) << 3) | (frac & 0x7)
def _fp8e4m3_int_to_float(n: int) -> float:
    e_biased = (n >> 3) & 0xF
    frac = n & 0x7
    if e_biased == 0:
        return 0.0
    return (2.0 ** (e_biased - 7)) * (1.0 + frac / 8.0)
def magic_constant_reciprocal_seed(
    b: float,
    fmt: str = "fp32",
) -> float:
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("magic_constant_reciprocal_seed requires b > 0 and finite")
    key = fmt.strip().lower()
    if key == "fp32":
        b_f32 = struct.unpack("<f", struct.pack("<f", b))[0]
        b_int = _float32_to_uint32(b_f32)
        seed_int = (MAGIC_CONST_FP32 - b_int) & 0xFFFFFFFF
        return float(_uint32_to_float32(seed_int))
    if key == "bf16":
        b_int = _float_to_bf16_int(b)
        seed_int = (MAGIC_CONST_BF16 - b_int) & 0xFFFF
        return float(_bf16_int_to_float(seed_int))
    if key in ("fp8", "fp8_e4m3", "fp8e4m3"):
        b_int = _float_to_fp8e4m3_int(b)
        seed_int = (MAGIC_CONST_FP8_E4M3 - b_int) & 0xFF
        return float(_fp8e4m3_int_to_float(seed_int))
    raise ValueError(
        f"Unsupported format: {fmt}. "
        f"Supported: 'fp32', 'bf16', 'fp8_e4m3'"
    )
def search_magic_constant(
    fmt: str = "fp32",
    *,
    b_samples: list[float] | None = None,
    objective: str = "worst",
) -> tuple[int, float]:
    if b_samples is None:
        b_samples = [
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
            7.0, 10.0, 42.0, 100.0, 1000.0,
        ]
    key = fmt.strip().lower()
    if key == "bf16":
        encode = _float_to_bf16_int
        decode = _bf16_int_to_float
        max_const = 0xFFFF
        mask = 0xFFFF
    elif key in ("fp8", "fp8_e4m3", "fp8e4m3"):
        encode = _float_to_fp8e4m3_int
        decode = _fp8e4m3_int_to_float
        max_const = 0xFF
        mask = 0xFF
    else:
        raise ValueError(
            f"Brute-force search not practical for {fmt}. "
            f"Use 'bf16' or 'fp8_e4m3'."
        )
    best_const = 0
    best_score = float("inf")
    for candidate in range(max_const + 1):
        errors = []
        valid = True
        for bv in b_samples:
            exact = 1.0 / bv
            b_int = encode(bv)
            seed_int = (candidate - b_int) & mask
            x0 = decode(seed_int)
            if x0 <= 0.0:
                valid = False
                break
            errors.append(abs(x0 / exact - 1.0))
        if not valid:
            continue
        score = max(errors) if objective == "worst" else sum(errors) / len(errors)
        if score < best_score:
            best_score = score
            best_const = candidate
    return best_const, best_score
def hardware_initial_guess(b: float, *, strategy: str = "linear") -> float:
    if b <= 0.0 or not math.isfinite(b):
        raise ValueError("b must be > 0 and finite")
    key = strategy.strip().lower()
    if key == "magic":
        return magic_constant_reciprocal_seed(b, fmt="fp32")
    if key == "magic_bf16":
        return magic_constant_reciprocal_seed(b, fmt="bf16")
    if key in {"magic_fp8", "magic_fp8_e4m3"}:
        return magic_constant_reciprocal_seed(b, fmt="fp8_e4m3")
    # frexp: b = m * 2^e  با  0.5 <= m < 1
    # x0 رو برای مانتیس نرمالایز‌شده حساب می‌کنیم، بعد با ldexp scale می‌زنیم
    m, e = math.frexp(b)
    if key == "linear":
        x_m = _x0_linear_for_mantissa(m)
    elif key in {"const", "constant", "one"}:
        x_m = _x0_constant_for_mantissa(m)
    else:
        raise ValueError(
            "x0_strategy must be 'linear', 'const', 'magic', "
            "'magic_bf16', or 'magic_fp8'"
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
) -> NewtonResult:
    if not isinstance(b, (float, int)):
        raise TypeError("b must be a float")
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
        x = x * (2.0 - b * x)  # NR iteration: x_{n+1} = x_n * (2 - b*x_n)
        err = abs(x - x_exact) / abs(x_exact)
        errors.append(err)
        if err < threshold:
            break
    converged = bool(errors) and errors[-1] < threshold
    return NewtonResult(x=x, iterations=len(errors), errors=errors, converged=converged)
def newton_reciprocal(
    b: float,
    *,
    threshold: float = 1e-12,
    max_iter: int = 50,
    x0: float | None = None,
    precision: str | None = None,
    x0_strategy: str = "linear",
) -> tuple[float, int, list[float]]:
    res = newton_reciprocal_result(
        b,
        threshold=threshold,
        max_iter=max_iter,
        x0=x0,
        precision=precision,
        x0_strategy=x0_strategy,
    )
    return res.x, res.iterations, res.errors
