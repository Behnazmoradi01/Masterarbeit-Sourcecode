import csv
import math
import os
import random
import statistics
import sys
import time
from nr_reciprocal.basic import newton_reciprocal_result
N_SAMPLES = 10_000
B_MIN, B_MAX = 1e-3, 1e6
MAX_ITER = 50
SEED_RNG = 42
STRATEGIES = ["linear", "magic", "magic_bf16", "magic_fp8"]
PRECISIONS = {
    "fp32": 2.0 ** (-24),
    "bf16": 2.0 ** (-7),
    "fp8":  2.0 ** (-3),
}
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "seed_iteration_table.csv")
def log_uniform_samples(n, lo, hi, seed=42):
    rng = random.Random(seed)
    log_lo, log_hi = math.log(lo), math.log(hi)
    return [math.exp(rng.uniform(log_lo, log_hi)) for _ in range(n)]
def percentile(data, p):
    s = sorted(data)
    k = int(math.ceil(p / 100.0 * len(s))) - 1
    return s[max(0, k)]
def main():
    b_values = log_uniform_samples(N_SAMPLES, B_MIN, B_MAX, SEED_RNG)
    rows = []
    results = []
    total = len(STRATEGIES) * len(PRECISIONS)
    done = 0
    t0 = time.time()
    for strategy in STRATEGIES:
        for prec_name, threshold in PRECISIONS.items():
            iters_list = []
            fails = 0
            for b in b_values:
                try:
                    res = newton_reciprocal_result(
                        b,
                        x0_strategy=strategy,
                        threshold=threshold,
                        max_iter=MAX_ITER,
                    )
                    if res.converged:
                        iters_list.append(res.iterations)
                    else:
                        fails += 1
                except (ValueError, OverflowError):
                    fails += 1
            done += 1
            n_ok = len(iters_list)
            if n_ok > 0:
                mean_i = statistics.mean(iters_list)
                med_i = statistics.median(iters_list)
                p95_i = percentile(iters_list, 95)
                p99_i = percentile(iters_list, 99)
                max_i = max(iters_list)
            else:
                mean_i = med_i = p95_i = p99_i = max_i = float("nan")
            fail_pct = 100.0 * fails / N_SAMPLES
            row = {
                "strategy": strategy,
                "precision": prec_name,
                "threshold": f"{threshold:.2e}",
                "n_ok": n_ok,
                "mean": f"{mean_i:.2f}",
                "median": f"{med_i:.1f}",
                "p95": p95_i,
                "p99": p99_i,
                "max": max_i,
                "fail_pct": f"{fail_pct:.2f}",
            }
            rows.append(row)
            results.append((strategy, prec_name, mean_i, med_i, p95_i, p99_i, max_i, fail_pct))
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  [{done}/{total}] {strategy:12s} x {prec_name:4s}  ({elapsed:.1f}s)")
            sys.stdout.flush()
    print("\n")
    hdr = f"{'strategy':>12s}  {'prec':>5s}  {'mean':>6s}  {'med':>5s}  {'p95':>4s}  {'p99':>4s}  {'max':>4s}  {'fail%':>6s}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for (strat, prec, mean_i, med_i, p95_i, p99_i, max_i, fail_pct) in results:
        print(f"{strat:>12s}  {prec:>5s}  {mean_i:6.2f}  {med_i:5.1f}  {p95_i:4d}  {p99_i:4d}  {max_i:4d}  {fail_pct:5.1f}%")
    print(sep)
    print(f"  ({N_SAMPLES} log-uniform samples, b in [{B_MIN}, {B_MAX}], max_iter={MAX_ITER})")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved: {os.path.abspath(OUT_CSV)}")
if __name__ == "__main__":
    main()
