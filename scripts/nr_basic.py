from __future__ import annotations
import argparse
import csv
from pathlib import Path
from nr_reciprocal import PRECISION_THRESHOLDS, newton_reciprocal, threshold_for_precision
def _write_csv(path: Path, b: float, x: float, iterations: int, errors: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "b", "x", "rel_error"])
        w.writeheader()
        for k, err in enumerate(errors, start=1):
            w.writerow({"k": k, "b": b, "x": x, "rel_error": err})
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Newton-Raphson reciprocal for 1/b."
    )
    ap.add_argument("--b", type=float, required=True, help="Divisor b (> 0)")
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=1e-12)
    ap.add_argument(
        "--precision",
        choices=sorted(PRECISION_THRESHOLDS.keys()),
        default="",
        help="Simulate FP precision via threshold (overrides --threshold)",
    )
    ap.add_argument(
        "--x0",
        type=float,
        default=float("nan"),
        help="Optional initial guess."
    )
    ap.add_argument(
        "--x0-strategy",
        choices=["linear", "const"],
        default="linear",
        help="Hardware-like x0 strategy used when --x0 is not provided",
    )
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path")
    args = ap.parse_args()
    x0 = None
    if args.x0 == args.x0:
        x0 = args.x0
    if args.precision:
        threshold = threshold_for_precision(args.precision)
    else:
        threshold = args.threshold
    x, iterations, errors = newton_reciprocal(
        args.b,
        threshold=threshold,
        max_iter=args.max_iter,
        x0=x0,
        x0_strategy=args.x0_strategy,
    )
    converged = bool(errors) and errors[-1] < threshold
    print(f"b={args.b}")
    print(f"threshold={threshold}")
    print(f"x={x:.17g}")
    print(f"iterations={iterations}")
    print(f"converged={converged}")
    if errors:
        print(f"final_rel_error={errors[-1]:.6e}")
    print("errors_per_iter=")
    for k, e in enumerate(errors, start=1):
        print(f"  {k}: {e:.6e}")
    if args.out:
        _write_csv(Path(args.out), args.b, x, iterations, errors)
        print(f"Wrote CSV: {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
