from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gs_binomial.basic import (
    PRECISION_THRESHOLDS,
    goldschmidt_binomial_divide,
    threshold_for_precision,
)
def _write_csv(path: Path, a: float, b: float, q: float, errors: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "a", "b", "q", "rel_error"])
        w.writeheader()
        for k, err in enumerate(errors, start=1):
            w.writerow({"k": k, "a": a, "b": b, "q": q, "rel_error": err})
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Goldschmidt division with binomial correction."
    )
    ap.add_argument("--a", type=float, required=True, help="Dividend a (>0, float)")
    ap.add_argument("--b", type=float, required=True, help="Divisor b (>0, float)")
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=1e-12)
    ap.add_argument(
        "--precision",
        choices=sorted(PRECISION_THRESHOLDS.keys()),
        default="",
        help="Simulate FP precision via threshold (overrides --threshold)",
    )
    ap.add_argument(
        "--order",
        type=int,
        choices=[2, 3],
        default=3,
        help="Binomial polynomial order: 2 uses 1+e+e^2, 3 uses 1+e+e^2+e^3",
    )
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path")
    args = ap.parse_args()
    if args.precision:
        threshold = threshold_for_precision(args.precision)
    else:
        threshold = args.threshold
    q, iters, errors = goldschmidt_binomial_divide(
        args.a,
        args.b,
        threshold=threshold,
        max_iter=args.max_iter,
        order=args.order,
    )
    converged = bool(errors) and errors[-1] < threshold
    print(f"a={args.a}")
    print(f"b={args.b}")
    print(f"order={args.order}")
    print(f"threshold={threshold}")
    print(f"q={q:.17g}")
    print(f"iterations={iters}")
    print(f"converged={converged}")
    if errors:
        print(f"final_rel_error={errors[-1]:.6e}")
    print("errors_per_iter=")
    for k, e in enumerate(errors, start=1):
        print(f"  {k}: {e:.6e}")
    if args.out:
        _write_csv(Path(args.out), args.a, args.b, q, errors)
        print(f"Wrote CSV: {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
