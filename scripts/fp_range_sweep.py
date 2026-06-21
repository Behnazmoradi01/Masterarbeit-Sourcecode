from __future__ import annotations
import csv
import random
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from nr_reciprocal import newton_reciprocal, threshold_for_precision as nr_thr
from gs_division.basic import goldschmidt_classic_divide, threshold_for_precision as gs_thr
from gs_binomial.basic import goldschmidt_binomial_divide, threshold_for_precision as gb_thr
N = 200_000
SEED = 42
MAX_ITER = 50
A = 1.0
RANGES = {
    "tiny":    (1e-6,  1e-2),
    "small":   (1e-2,  0.5),
    "near_1":  (0.5,   2.0),
    "medium":  (2.0,   100.0),
    "large":   (100.0, 1e6),
}
PRECISIONS = ["fp8", "bf16", "fp32"]
ALGORITHMS = ["Newton-Raphson", "Goldschmidt classic", "Goldschmidt binomial"]
def sample_uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * random.random()
def iters_until(errors: list[float], threshold: float) -> int:
    for i, e in enumerate(errors):
        if e < threshold:
            return i + 1
    return len(errors)
def run_algos(b: float):
    _, _, errs_nr = newton_reciprocal(b, precision="fp32", max_iter=MAX_ITER)
    _, _, errs_gs = goldschmidt_classic_divide(A, b, precision="fp32", max_iter=MAX_ITER)
    _, _, errs_gb = goldschmidt_binomial_divide(A, b, precision="fp32", max_iter=MAX_ITER, order=3)
    return {"Newton-Raphson": errs_nr,
            "Goldschmidt classic": errs_gs,
            "Goldschmidt binomial": errs_gb}
def thr_for(algo: str, prec: str) -> float:
    if algo == "Newton-Raphson":
        return nr_thr(prec)
    if algo == "Goldschmidt classic":
        return gs_thr(prec)
    return gb_thr(prec)
def main() -> None:
    random.seed(SEED)
    tracker: dict[str, dict[str, dict[str, int]]] = {}
    for rng in RANGES:
        tracker[rng] = {}
        for algo in ALGORITHMS:
            tracker[rng][algo] = {p: 0 for p in PRECISIONS}
    total = len(RANGES) * N
    done = 0
    for rng_name, (lo, hi) in RANGES.items():
        print(f"Range {rng_name} [{lo}, {hi}] ...")
        for _ in range(N):
            b = sample_uniform(lo, hi)
            all_errs = run_algos(b)
            for algo in ALGORITHMS:
                for prec in PRECISIONS:
                    it = iters_until(all_errs[algo], thr_for(algo, prec))
                    if it > tracker[rng_name][algo][prec]:
                        tracker[rng_name][algo][prec] = it
            done += 1
            if done % 100_000 == 0:
                print(f"  {done}/{total}")
    out = _ROOT / "results" / "stress" / "fp_range_worst_case.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rng_name, (lo, hi) in RANGES.items():
        for algo in ALGORITHMS:
            for prec in PRECISIONS:
                rows.append({
                    "range": rng_name,
                    "b_lo": lo,
                    "b_hi": hi,
                    "algorithm": algo,
                    "precision": prec,
                    "worst_case_iterations": tracker[rng_name][algo][prec],
                })
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {out}")
    print(f"\n{'range':<10} {'algorithm':<25} {'fp8':>4} {'bf16':>5} {'fp32':>5}")
    print("-" * 55)
    for rng_name in RANGES:
        for algo in ALGORITHMS:
            d = tracker[rng_name][algo]
            print(f"{rng_name:<10} {algo:<25} {d['fp8']:>4} {d['bf16']:>5} {d['fp32']:>5}")
        print()
if __name__ == "__main__":
    main()
