from __future__ import annotations
import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from nr_reciprocal import newton_reciprocal, threshold_for_precision
@dataclass(frozen=True)
class Stats:
    n: int
    min_iters: int
    max_iters: int
    mean_iters: float
    median_iters: float
    p95_iters: int
    p99_iters: int
    failure_rate: float
def _percentile_nearest_rank(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        raise ValueError("empty values")
    if p <= 0.0:
        return sorted_values[0]
    if p >= 1.0:
        return sorted_values[-1]
    n = len(sorted_values)
    k = int(math.ceil(p * n))
    idx = max(0, min(n - 1, k - 1))
    return sorted_values[idx]
def _median(sorted_values: list[int]) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
def _compute_stats(iters: list[int], failures: int) -> Stats:
    if not iters:
        raise ValueError("No samples")
    iters_sorted = sorted(iters)
    n = len(iters_sorted)
    mean = sum(iters_sorted) / n
    return Stats(
        n=n,
        min_iters=iters_sorted[0],
        max_iters=iters_sorted[-1],
        mean_iters=mean,
        median_iters=_median(iters_sorted),
        p95_iters=_percentile_nearest_rank(iters_sorted, 0.95),
        p99_iters=_percentile_nearest_rank(iters_sorted, 0.99),
        failure_rate=failures / n,
    )
def _sample_b_log_uniform(exp_min: float, exp_max: float) -> float:
    u = exp_min + (exp_max - exp_min) * random.random()
    return 2.0 ** u
def _write_summary_csv(path: Path, *, exp_min: float, exp_max: float, max_iter: int, rows: list[tuple[str, Stats]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "precision",
                "exp_min",
                "exp_max",
                "max_iter",
                "n",
                "min_iters",
                "max_iters",
                "mean_iters",
                "median_iters",
                "p95_iters",
                "p99_iters",
                "failure_rate",
            ],
        )
        w.writeheader()
        for precision, s in rows:
            w.writerow(
                {
                    "precision": precision,
                    "exp_min": exp_min,
                    "exp_max": exp_max,
                    "max_iter": max_iter,
                    "n": s.n,
                    "min_iters": s.min_iters,
                    "max_iters": s.max_iters,
                    "mean_iters": f"{s.mean_iters:.6f}",
                    "median_iters": f"{s.median_iters:.6f}",
                    "p95_iters": s.p95_iters,
                    "p99_iters": s.p99_iters,
                    "failure_rate": f"{s.failure_rate:.6f}",
                }
            )
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Random sweep for Newton–Raphson reciprocal convergence statistics. "
            "Samples b log-uniformly over [2**exp_min, 2**exp_max] and runs the "
            "threshold-only precision presets (fp8/bf16/fp32)."
        )
    )
    ap.add_argument("--n", type=int, default=100_000, help="Number of random b samples")
    ap.add_argument("--exp-min", type=float, default=-20.0, help="Minimum exponent (base-2) for b range")
    ap.add_argument("--exp-max", type=float, default=20.0, help="Maximum exponent (base-2) for b range")
    ap.add_argument("--max-iter", type=int, default=50, help="Maximum Newton iterations")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 uses system entropy)")
    ap.add_argument("--out", type=str, default="nr_random_sweep_summary.csv", help="Output summary CSV path")
    args = ap.parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.max_iter < 0:
        raise SystemExit("--max-iter must be >= 0")
    if args.exp_min >= args.exp_max:
        raise SystemExit("--exp-min must be < --exp-max")
    if args.seed != 0:
        random.seed(args.seed)
    bs = [_sample_b_log_uniform(args.exp_min, args.exp_max) for _ in range(args.n)]
    results: list[tuple[str, Stats]] = []
    for precision in ["fp8", "bf16", "fp32"]:
        thr = threshold_for_precision(precision)
        iters_list: list[int] = []
        failures = 0
        for b in bs:
            _, iters, errs = newton_reciprocal(b, precision=precision, max_iter=args.max_iter)
            iters_list.append(iters)
            final_err = errs[-1] if errs else float("inf")
            if not (final_err < thr):
                failures += 1
        stats = _compute_stats(iters_list, failures)
        results.append((precision, stats))
    print(f"N={args.n}")
    print(f"b range: [2**{args.exp_min}, 2**{args.exp_max}]")
    print(f"max_iter={args.max_iter}")
    print("precision\tmin\tmax\tmean\tmedian\tp95\tp99\tfailure_rate")
    for precision, s in results:
        print(
            f"{precision}\t{s.min_iters}\t{s.max_iters}\t{s.mean_iters:.3f}\t{s.median_iters:.3f}\t"
            f"{s.p95_iters}\t{s.p99_iters}\t{s.failure_rate:.6f}"
        )
    out_path = Path(args.out)
    _write_summary_csv(
        out_path,
        exp_min=args.exp_min,
        exp_max=args.exp_max,
        max_iter=args.max_iter,
        rows=results,
    )
    print(f"Wrote CSV: {out_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
