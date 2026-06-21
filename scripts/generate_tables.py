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
PRECISIONS = ["fp8", "bf16", "fp32"]
ALGORITHMS = ["Newton-Raphson", "Goldschmidt classic", "Goldschmidt binomial"]
PER_ITER_OPS = {
    "Newton-Raphson":       (2,   0,   1),
    "Goldschmidt classic":  (2,   0,   1),
    "Goldschmidt binomial": (4,   3,   1),
}
OVERHEAD = {
    "Newton-Raphson":       {"MUL": 1, "ADD": 0, "SUB": 1, "SHIFT": 2},
    "Goldschmidt classic":  {"MUL": 0, "ADD": 0, "SUB": 0, "SHIFT": 2},
    "Goldschmidt binomial": {"MUL": 0, "ADD": 0, "SUB": 0, "SHIFT": 2},
}
def iters_until(errors: list[float], threshold: float) -> int:
    for i, e in enumerate(errors):
        if e < threshold:
            return i + 1
    return len(errors)
def thr(algo: str, prec: str) -> float:
    if algo == "Newton-Raphson":
        return nr_thr(prec)
    if algo == "Goldschmidt classic":
        return gs_thr(prec)
    return gb_thr(prec)
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote: {path}")
def main() -> None:
    random.seed(SEED)
    out_dir = _ROOT / "results" / "stress"
    tracker: dict[str, dict[str, list[int]]] = {}
    for algo in ALGORITHMS:
        tracker[algo] = {p: [999, 0] for p in PRECISIONS}
    print(f"Sampling {N} random b values ...")
    for i in range(N):
        b = 2.0 ** (random.uniform(-20, 20))
        _, _, errs_nr = newton_reciprocal(b, precision="fp32", max_iter=MAX_ITER)
        _, _, errs_gs = goldschmidt_classic_divide(A, b, precision="fp32", max_iter=MAX_ITER)
        _, _, errs_gb = goldschmidt_binomial_divide(A, b, precision="fp32", max_iter=MAX_ITER, order=3)
        all_errs = {
            "Newton-Raphson": errs_nr,
            "Goldschmidt classic": errs_gs,
            "Goldschmidt binomial": errs_gb,
        }
        for algo in ALGORITHMS:
            for prec in PRECISIONS:
                it = iters_until(all_errs[algo], thr(algo, prec))
                if it < tracker[algo][prec][0]:
                    tracker[algo][prec][0] = it
                if it > tracker[algo][prec][1]:
                    tracker[algo][prec][1] = it
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1}/{N}")
    print("\n=== Table 1: min/max iterations ===")
    rows1 = []
    for algo in ALGORITHMS:
        for prec in PRECISIONS:
            mn, mx = tracker[algo][prec]
            rows1.append({
                "algorithm": algo,
                "precision": prec,
                "min_iterations": mn,
                "max_iterations": mx,
            })
    write_csv(out_dir / "table1_iterations.csv", rows1)
    print(f"{'algorithm':<25} {'prec':<6} {'min':>4} {'max':>4}")
    print("-" * 43)
    for r in rows1:
        print(f"{r['algorithm']:<25} {r['precision']:<6} {r['min_iterations']:>4} {r['max_iterations']:>4}")
    print("\n=== Table 2: per-iteration ops ===")
    rows2 = []
    for algo in ALGORITHMS:
        mul, add, sub = PER_ITER_OPS[algo]
        rows2.append({
            "algorithm": algo,
            "MUL_per_iter": mul,
            "ADD_per_iter": add,
            "SUB_per_iter": sub,
        })
    write_csv(out_dir / "table2_per_iter_ops.csv", rows2)
    print(f"{'algorithm':<25} {'MUL':>4} {'ADD':>4} {'SUB':>4}")
    print("-" * 41)
    for r in rows2:
        print(f"{r['algorithm']:<25} {r['MUL_per_iter']:>4} {r['ADD_per_iter']:>4} {r['SUB_per_iter']:>4}")
    print("\n=== Table 3: total ops (per_iter * max_iter + overhead) ===")
    rows3 = []
    for algo in ALGORITHMS:
        mul_pi, add_pi, sub_pi = PER_ITER_OPS[algo]
        ov = OVERHEAD[algo]
        for prec in PRECISIONS:
            mx = tracker[algo][prec][1]
            rows3.append({
                "algorithm": algo,
                "precision": prec,
                "max_iterations": mx,
                "total_MUL": mul_pi * mx + ov["MUL"],
                "total_ADD": add_pi * mx + ov["ADD"],
                "total_SUB": sub_pi * mx + ov["SUB"],
                "total_SHIFT": ov["SHIFT"],
            })
    write_csv(out_dir / "table3_total_ops.csv", rows3)
    print(f"{'algorithm':<25} {'prec':<6} {'iters':>5} {'MUL':>5} {'ADD':>5} {'SUB':>5} {'SHIFT':>6}")
    print("-" * 60)
    for r in rows3:
        print(f"{r['algorithm']:<25} {r['precision']:<6} {r['max_iterations']:>5} "
              f"{r['total_MUL']:>5} {r['total_ADD']:>5} {r['total_SUB']:>5} {r['total_SHIFT']:>6}")
    print("\nDone.")
if __name__ == "__main__":
    main()
