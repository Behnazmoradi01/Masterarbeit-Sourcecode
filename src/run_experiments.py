from __future__ import annotations
import csv
import math
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basic import (
    newton_reciprocal_result,
    threshold_for_precision,
)
from simulate import (
    NewtonReciprocalConfig,
    simulate_newton_reciprocal,
)
STRATEGIES = ["linear", "magic", "magic_bf16"]
PRECISIONS = ["fp32", "bf16"]
TEST_BS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.14, 5.0, 10.0, 42.0, 100.0]
N_RANDOM = 1000
RNG_SEED = 42
def _log_uniform(n: int, lo: float, hi: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    ll, lh = math.log(lo), math.log(hi)
    return [math.exp(rng.uniform(ll, lh)) for _ in range(n)]
def table_iteration_counts(precision: str):
    thr = threshold_for_precision(precision)
    print(f"\n{'─' * 72}")
    print(f"  Iteration counts  │  precision = {precision}  "
          f"(threshold = {thr:.2e})")
    print(f"{'─' * 72}")
    hdr = f"  {'b':>10s}"
    for s in STRATEGIES:
        hdr += f"  {s:>12s}"
    print(hdr)
    print("  " + "─" * (10 + 14 * len(STRATEGIES)))
    for b in TEST_BS:
        line = f"  {b:10.4f}"
        for strat in STRATEGIES:
            r = newton_reciprocal_result(
                b, threshold=thr, max_iter=50, x0_strategy=strat,
            )
            line += f"  {r.iterations:12d}"
        print(line)
def table_statistics():
    samples = _log_uniform(N_RANDOM, 0.01, 1000.0, RNG_SEED)
    print(f"\n{'═' * 72}")
    print(f"  Statistical Summary  │  {N_RANDOM} log-uniform samples "
          f"in [0.01, 1000]")
    print(f"{'═' * 72}")
    for prec in PRECISIONS:
        thr = threshold_for_precision(prec)
        print(f"\n  ── {prec.upper()} (threshold = {thr:.2e}) ──")
        print(f"  {'strategy':>14s}  {'mean':>7s}  {'max':>4s}  "
              f"{'mean_err':>12s}  {'max_err':>12s}")
        print("  " + "─" * 55)
        for strat in STRATEGIES:
            iters: list[int] = []
            errs: list[float] = []
            for b in samples:
                r = newton_reciprocal_result(
                    b, threshold=thr, max_iter=50, x0_strategy=strat,
                )
                iters.append(r.iterations)
                if r.errors:
                    errs.append(r.errors[-1])
            m_it = sum(iters) / len(iters)
            mx_it = max(iters)
            m_err = sum(errs) / len(errs) if errs else float("nan")
            mx_err = max(errs) if errs else float("nan")
            print(f"  {strat:>14s}  {m_it:7.3f}  {mx_it:4d}  "
                  f"{m_err:12.3e}  {mx_err:12.3e}")
def table_convergence(b: float = 3.14, strategy: str = "linear"):
    cfg = NewtonReciprocalConfig(
        max_iter=12, tol_rel=1e-16, x0_strategy=strategy,
    )
    pts = simulate_newton_reciprocal(b, cfg)
    print(f"\n{'─' * 72}")
    print(f"  Convergence trace  │  b = {b},  seed = {strategy}")
    print(f"{'─' * 72}")
    print(f"  {'k':>3s}  {'x (≈ 1/b)':>22s}  {'residual':>14s}  "
          f"{'rel_error':>14s}")
    print("  " + "─" * 58)
    for p in pts:
        print(f"  {p.k:3d}  {p.x:22.15e}  {p.residual:14.6e}  "
              f"{p.err_rel:14.6e}")
def table_quantization_effect():
    samples = _log_uniform(200, 0.1, 100.0, RNG_SEED)
    quant_configs = [
        ("none (f64)", None),
        ("fp32",       "fp32"),
        ("bf16",       "bf16"),
        ("fp8_e4m3",   "fp8_e4m3"),
    ]
    print(f"\n{'═' * 72}")
    print(f"  Per-Iteration Quantization Effect  │  200 samples in [0.1, 100]")
    print(f"{'═' * 72}")
    thr32 = threshold_for_precision("fp32")
    print(f"\n  ── All targeting FP32 precision (threshold = {thr32:.2e}) ──")
    print(f"  {'quantize':>12s}  {'mean_it':>8s}  {'max_it':>6s}  "
          f"{'converged':>10s}")
    print("  " + "─" * 42)
    for label, fmt in quant_configs:
        iters: list[int] = []
        conv = 0
        for b in samples:
            r = newton_reciprocal_result(
                b, threshold=thr32, max_iter=50,
                x0_strategy="linear", float_quantize_fmt=fmt,
            )
            iters.append(r.iterations)
            conv += int(r.converged)
        print(f"  {label:>12s}  "
              f"{sum(iters)/len(iters):8.3f}  "
              f"{max(iters):6d}  "
              f"{100 * conv / len(samples):9.1f}%")
    match_cfg = [
        ("fp32",     "fp32",     "fp32"),
        ("bf16",     "bf16",     "bf16"),
        ("fp8_e4m3", "fp8_e4m3", "fp8"),
    ]
    print(f"\n  ── Each format at its matching precision threshold ──")
    print(f"  {'quantize':>12s}  {'threshold':>12s}  {'mean_it':>8s}  "
          f"{'max_it':>6s}  {'converged':>10s}")
    print("  " + "─" * 56)
    for label, fmt, prec in match_cfg:
        thr = threshold_for_precision(prec)
        iters = []
        conv = 0
        for b in samples:
            r = newton_reciprocal_result(
                b, threshold=thr, max_iter=50,
                x0_strategy="linear", float_quantize_fmt=fmt,
            )
            iters.append(r.iterations)
            conv += int(r.converged)
        print(f"  {label:>12s}  {thr:12.2e}  "
              f"{sum(iters)/len(iters):8.3f}  "
              f"{max(iters):6d}  "
              f"{100 * conv / len(samples):9.1f}%")
def table_fp8_exploratory():
    samples = _log_uniform(200, 0.1, 100.0, RNG_SEED)
    thr = threshold_for_precision("fp8")
    print(f"\n{'═' * 72}")
    print(f"  FP8 Magic Constant — EXPLORATORY  (threshold = {thr:.2e})")
    print(f"{'═' * 72}")
    for strat in ["linear", "magic_fp8"]:
        iters: list[int] = []
        conv = 0
        errs: list[float] = []
        for b in samples:
            r = newton_reciprocal_result(
                b, threshold=thr, max_iter=50, x0_strategy=strat,
            )
            iters.append(r.iterations)
            conv += int(r.converged)
            if r.errors:
                errs.append(r.errors[-1])
        m_it = sum(iters) / len(iters)
        mx_it = max(iters)
        m_err = sum(e for e in errs if math.isfinite(e)) / max(1, len(errs))
        print(f"  {strat:>12s}  mean_it={m_it:.2f}  max_it={mx_it}  "
              f"converged={100*conv/len(samples):.0f}%  "
              f"mean_err={m_err:.3e}")
    print("  NOTE: magic_fp8 is EXPERIMENTAL — not included in main tables.")
def export_csv(path: str):
    samples = _log_uniform(N_RANDOM, 0.01, 1000.0, RNG_SEED)
    rows = []
    for prec in PRECISIONS:
        thr = threshold_for_precision(prec)
        for strat in STRATEGIES:
            iters: list[int] = []
            errs: list[float] = []
            for b in samples:
                r = newton_reciprocal_result(
                    b, threshold=thr, max_iter=50, x0_strategy=strat,
                )
                iters.append(r.iterations)
                if r.errors:
                    errs.append(r.errors[-1])
            rows.append({
                "precision": prec,
                "strategy": strat,
                "mean_iters": f"{sum(iters)/len(iters):.4f}",
                "max_iters": max(iters),
                "mean_final_err": (
                    f"{sum(errs)/len(errs):.6e}" if errs else "NaN"
                ),
                "max_final_err": (
                    f"{max(errs):.6e}" if errs else "NaN"
                ),
            })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV exported → {path}")
def main():
    print("═" * 72)
    print("  Newton–Raphson Reciprocal (1/b) — Experiment Results")
    print("═" * 72)
    for prec in PRECISIONS:
        table_iteration_counts(prec)
    table_statistics()
    table_convergence(3.14, "linear")
    table_convergence(3.14, "magic")
    table_quantization_effect()
    table_fp8_exploratory()
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "experiment_summary.csv",
    )
    export_csv(out)
    print("\n" + "═" * 72)
    print("  Done.")
    print("═" * 72)
if __name__ == "__main__":
    main()
