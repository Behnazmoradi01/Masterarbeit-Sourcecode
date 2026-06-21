from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from nr_reciprocal import NewtonReciprocalConfig, simulate_newton_reciprocal
from nr_reciprocal import PRECISION_THRESHOLDS, threshold_for_precision
def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]
def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep b values and record convergence/iteration count")
    ap.add_argument("--b-min", type=float, required=True)
    ap.add_argument("--b-max", type=float, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--mode", choices=["float", "fixed"], default="fixed")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tol-abs", type=float, default=0.0)
    ap.add_argument("--tol-rel", type=float, default=1e-12)
    ap.add_argument(
        "--precision",
        choices=sorted(PRECISION_THRESHOLDS.keys()),
        default="",
        help="Simulate FP precision via relative-error threshold (overrides --tol-rel)",
    )
    ap.add_argument("--x0", choices=["linear", "one"], default="linear")
    ap.add_argument("--frac-bits", type=int, default=24)
    ap.add_argument("--rounding", choices=["nearest_even", "trunc"], default="nearest_even")
    ap.add_argument("--ref-prec", type=int, default=80)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    tol_rel = args.tol_rel
    if args.precision:
        tol_rel = threshold_for_precision(args.precision)
    cfg = NewtonReciprocalConfig(
        mode=args.mode,
        max_iter=args.max_iter,
        tol_abs=args.tol_abs,
        tol_rel=tol_rel,
        frac_bits=args.frac_bits,
        rounding=args.rounding,
        x0_strategy=args.x0,
        reference_decimal_prec=args.ref_prec,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bs = _linspace(args.b_min, args.b_max, args.count)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "b",
                "iters_recorded",
                "iters_to_tol",
                "final_x",
                "final_abs_err",
                "final_rel_err",
                "final_residual",
            ],
        )
        w.writeheader()
        for b in bs:
            if b == 0.0 or not math.isfinite(b):
                continue
            pts = simulate_newton_reciprocal(b, cfg)
            final = pts[-1]
            iters_to_tol = ""
            for p in pts:
                hit_abs = cfg.tol_abs > 0.0 and p.err_abs <= cfg.tol_abs
                hit_rel = cfg.tol_rel > 0.0 and p.err_rel <= cfg.tol_rel
                if hit_abs or hit_rel:
                    iters_to_tol = p.k
                    break
            w.writerow(
                {
                    "b": b,
                    "iters_recorded": len(pts) - 1,
                    "iters_to_tol": iters_to_tol,
                    "final_x": final.x,
                    "final_abs_err": final.err_abs,
                    "final_rel_err": final.err_rel,
                    "final_residual": final.residual,
                }
            )
    print(f"Wrote sweep CSV: {out_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
