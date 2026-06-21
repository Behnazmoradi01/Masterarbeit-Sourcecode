from __future__ import annotations
import argparse
import csv
from pathlib import Path
from nr_reciprocal import NewtonReciprocalConfig, simulate_newton_reciprocal
from nr_reciprocal import PRECISION_THRESHOLDS, threshold_for_precision
def _print_table(points):
    headers = [
        "k",
        "b_norm",
        "e",
        "x (approx 1/b)",
        "residual (1-bx)",
        "abs_err",
        "rel_err",
    ]
    print("\t".join(headers))
    for p in points:
        print(
            f"{p.k}\t"
            f"{p.b_norm:.12g}\t"
            f"{p.scale_e}\t"
            f"{p.x:.16g}\t"
            f"{p.residual:.3e}\t"
            f"{p.err_abs:.3e}\t"
            f"{p.err_rel:.3e}"
        )
def _write_csv(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "b",
                "b_norm",
                "scale_e",
                "x",
                "residual",
                "err_abs",
                "err_rel",
                "x_int",
            ],
        )
        w.writeheader()
        for p in points:
            w.writerow(
                {
                    "k": p.k,
                    "b": p.b,
                    "b_norm": p.b_norm,
                    "scale_e": p.scale_e,
                    "x": p.x,
                    "residual": p.residual,
                    "err_abs": p.err_abs,
                    "err_rel": p.err_rel,
                    "x_int": p.x_int,
                }
            )
def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate Newton–Raphson reciprocal for 1/b")
    ap.add_argument("--b", required=True, help="Divisor b (e.g. 10, 3.14, -7.5)")
    ap.add_argument("--mode", choices=["float", "fixed"], default="float")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tol-abs", type=float, default=0.0)
    ap.add_argument("--tol-rel", type=float, default=0.0)
    ap.add_argument(
        "--precision",
        choices=sorted(PRECISION_THRESHOLDS.keys()),
        default="",
        help="Simulate FP precision via relative-error threshold (overrides --tol-rel)",
    )
    ap.add_argument("--x0", choices=["linear", "one"], default="linear")
    ap.add_argument("--frac-bits", type=int, default=24)
    ap.add_argument("--rounding", choices=["nearest_even", "trunc"], default="nearest_even")
    ap.add_argument("--ref-prec", type=int, default=80, help="Decimal precision for reference 1/b")
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path")
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
    points = simulate_newton_reciprocal(args.b, cfg)
    _print_table(points)
    if args.out:
        _write_csv(Path(args.out), points)
        print(f"\nWrote CSV: {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
