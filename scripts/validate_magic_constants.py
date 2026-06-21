import csv
import math
import os
import random
import statistics
import sys
import time
from nr_reciprocal.basic import (
    MAGIC_CONST_BF16,
    MAGIC_CONST_FP8_E4M3,
    _float_to_bf16_int,
    _bf16_int_to_float,
    _float_to_fp8e4m3_int,
    _fp8e4m3_int_to_float,
    search_magic_constant,
    newton_reciprocal_result,
)
N_RANDOM = 200
RNG_SEED = 2026
MAX_ITER = 50
THRESHOLD_FP32 = 2.0 ** (-24)
B_SAMPLES_BF16 = [
    0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
    7.0, 10.0, 42.0, 100.0, 1000.0,
]
B_SAMPLES_FP8 = [
    0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0,
    4.0, 5.0, 7.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0,
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FORMATS = {
    "bf16": {
        "label":     "BF16 (bfloat16)",
        "const":     MAGIC_CONST_BF16,
        "max_val":   0xFFFF,
        "mask":      0xFFFF,
        "encode":    _float_to_bf16_int,
        "decode":    _bf16_int_to_float,
        "strategy":  "magic_bf16",
        "hex_fmt":   "0x{:04X}",
        "b_samples": B_SAMPLES_BF16,
    },
    "fp8": {
        "label":     "FP8 E4M3",
        "const":     MAGIC_CONST_FP8_E4M3,
        "max_val":   0xFF,
        "mask":      0xFF,
        "encode":    _float_to_fp8e4m3_int,
        "decode":    _fp8e4m3_int_to_float,
        "strategy":  "magic_fp8",
        "hex_fmt":   "0x{:02X}",
        "b_samples": B_SAMPLES_FP8,
    },
}
def seed_errors_for_constant(fmt_info, constant, b_samples):
    encode = fmt_info["encode"]
    decode = fmt_info["decode"]
    mask = fmt_info["mask"]
    errs = []
    for bv in b_samples:
        exact = 1.0 / bv
        b_int = encode(bv)
        seed_int = (constant - b_int) & mask
        x0 = decode(seed_int)
        if x0 <= 0.0:
            errs.append(None)
        else:
            errs.append(abs(x0 / exact - 1.0))
    return errs
def nr_stats_for_constant(fmt_info, constant, b_samples, threshold):
    encode = fmt_info["encode"]
    decode = fmt_info["decode"]
    mask = fmt_info["mask"]
    iters = []
    fails = 0
    for bv in b_samples:
        b_int = encode(bv)
        seed_int = (constant - b_int) & mask
        x0 = decode(seed_int)
        if x0 <= 0.0:
            fails += 1
            continue
        try:
            res = newton_reciprocal_result(
                bv, x0=x0, threshold=threshold, max_iter=MAX_ITER
            )
            if res.converged:
                iters.append(res.iterations)
            else:
                fails += 1
        except (ValueError, OverflowError):
            fails += 1
    if iters:
        return statistics.mean(iters), max(iters), fails
    return float("nan"), 0, fails
def convergence_search_exhaustive(fmt_info, b_samples, threshold):
    max_val = fmt_info["max_val"]
    best = (None, float("inf"), 999, 999)
    for c in range(max_val + 1):
        mi, mx, f = nr_stats_for_constant(fmt_info, c, b_samples, threshold)
        if (f, mi, mx) < (best[3], best[1], best[2]):
            best = (c, mi, mx, f)
        if (c + 1) % 10000 == 0:
            sys.stdout.write(f"\r     ... {c+1}/{max_val+1} candidates tested")
            sys.stdout.flush()
    if max_val > 1000:
        print()
    return best
def validate_format(fmt_key, fmt_info):
    label = fmt_info["label"]
    our_const = fmt_info["const"]
    hex_fmt = fmt_info["hex_fmt"]
    max_val = fmt_info["max_val"]
    b_samples = fmt_info["b_samples"]
    rng = random.Random(RNG_SEED)
    print()
    print("=" * 76)
    print(f"  VALIDATION: {label}   (shipped constant = {hex_fmt.format(our_const)})")
    print("=" * 76)
    print(f"\n  -- OUTPUT 1a: Seed-error search")
    print(f"     b_samples = {b_samples}")
    t0 = time.time()
    c_worst, s_worst = search_magic_constant(
        fmt_key, b_samples=b_samples, objective="worst"
    )
    c_mean, s_mean = search_magic_constant(
        fmt_key, b_samples=b_samples, objective="mean"
    )
    dt = time.time() - t0
    print(f"     Search time: {dt:.2f}s")
    our_errs = seed_errors_for_constant(fmt_info, our_const, b_samples)
    our_valid = [e for e in our_errs if e is not None]
    our_worst_err = max(our_valid) if our_valid else float("inf")
    our_mean_err = statistics.mean(our_valid) if our_valid else float("inf")
    print()
    print(f"     {'Objective':>16s}  {'Best const':>12s}  {'Score':>10s}  {'Ours':>10s}  {'Match?':>7s}")
    print(f"     {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*7}")
    match_w = "YES" if c_worst == our_const else "NO"
    match_m = "YES" if c_mean == our_const else "NO"
    print(f"     {'worst-case':>16s}  {hex_fmt.format(c_worst):>12s}  {s_worst:10.4f}  {our_worst_err:10.4f}  {match_w:>7s}")
    print(f"     {'mean':>16s}  {hex_fmt.format(c_mean):>12s}  {s_mean:10.4f}  {our_mean_err:10.4f}  {match_m:>7s}")
    if max_val <= 0xFF:
        print(f"\n  -- OUTPUT 1b: NR convergence search (threshold={THRESHOLD_FP32:.2e})")
        t0 = time.time()
        conv_best, conv_mi, conv_mx, conv_f = convergence_search_exhaustive(
            fmt_info, b_samples, THRESHOLD_FP32
        )
        dt = time.time() - t0
        print(f"     Search time: {dt:.1f}s  (tested {max_val + 1} candidates)")
    else:
        print(f"\n  -- OUTPUT 1b: Grid search around seed-error best constant")
        lo = max(0, c_worst - 256)
        hi = min(max_val, c_worst + 256)
        best = (None, float("inf"), 999, 999)
        t0 = time.time()
        for c in range(lo, hi + 1):
            mi, mx, f = nr_stats_for_constant(fmt_info, c, b_samples, THRESHOLD_FP32)
            if (f, mi, mx) < (best[3], best[1], best[2]):
                best = (c, mi, mx, f)
        conv_best, conv_mi, conv_mx, conv_f = best
        dt = time.time() - t0
        print(f"     Tested {hi - lo + 1} candidates in {dt:.1f}s")
    our_mi, our_mx, our_f = nr_stats_for_constant(
        fmt_info, our_const, b_samples, THRESHOLD_FP32
    )
    match_c = "YES" if conv_best == our_const else "NO"
    print()
    print(f"     {'':>16s}  {'constant':>12s}  {'mean_iter':>10s}  {'max_iter':>10s}  {'fails':>6s}")
    print(f"     {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*6}")
    print(f"     {'BEST (search)':>16s}  {hex_fmt.format(conv_best):>12s}  {conv_mi:10.2f}  {conv_mx:10d}  {conv_f:6d}")
    print(f"     {'OURS (shipped)':>16s}  {hex_fmt.format(our_const):>12s}  {our_mi:10.2f}  {our_mx:10d}  {our_f:6d}")
    print(f"     Match? {match_c}")
    print(f"\n  -- Seed detail for {hex_fmt.format(our_const)}:")
    print(f"     {'b':>10s}  {'x0':>12s}  {'exact':>12s}  {'seed_err':>10s}  {'NR_iters':>9s}")
    print(f"     {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*9}")
    encode = fmt_info["encode"]
    decode = fmt_info["decode"]
    mask = fmt_info["mask"]
    for i, bv in enumerate(b_samples):
        exact = 1.0 / bv
        b_int = encode(bv)
        seed_int = (our_const - b_int) & mask
        x0 = decode(seed_int)
        if x0 <= 0:
            print(f"     {bv:10.4f}  {'INVALID':>12s}  {exact:12.6f}  {'---':>10s}  {'---':>9s}")
            continue
        err = abs(x0 / exact - 1.0)
        try:
            res = newton_reciprocal_result(bv, x0=x0, threshold=THRESHOLD_FP32, max_iter=MAX_ITER)
            it = str(res.iterations) if res.converged else "FAIL"
        except Exception:
            it = "ERR"
        print(f"     {bv:10.4f}  {x0:12.6f}  {exact:12.6f}  {err:10.4f}  {it:>9s}")
    print(f"\n  -- OUTPUT 2: Compare with {N_RANDOM} random constants")
    rand_data = []
    for _ in range(N_RANDOM):
        c = rng.randint(0, max_val)
        errs = seed_errors_for_constant(fmt_info, c, b_samples)
        valid = [e for e in errs if e is not None]
        if not valid:
            continue
        mi, mx, f = nr_stats_for_constant(fmt_info, c, b_samples, THRESHOLD_FP32)
        rand_data.append((c, max(valid), statistics.mean(valid), mi, mx, f))
    n_rand = len(rand_data)
    rand_worst_errs = [d[1] for d in rand_data]
    rand_mean_errs = [d[2] for d in rand_data]
    rand_mean_iters = [d[3] for d in rand_data if not math.isnan(d[3])]
    rand_fails = [d[5] for d in rand_data]
    rank_werr = sum(1 for w in rand_worst_errs if w < our_worst_err) + 1
    rank_merr = sum(1 for m in rand_mean_errs if m < our_mean_err) + 1
    rank_iters = sum(1 for m in rand_mean_iters if m < our_mi) + 1 if not math.isnan(our_mi) else n_rand
    rank_fails = sum(1 for f in rand_fails if f < our_f) + 1
    def safe_med(lst):
        return statistics.median(lst) if lst else float("nan")
    def safe_mean(lst):
        return statistics.mean(lst) if lst else float("nan")
    print(f"\n     {'Metric':>16s}  {'Ours':>10s}  {'Rand med':>10s}  {'Rand mean':>10s}  {'Rank':>12s}")
    print(f"     {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")
    print(f"     {'worst seed err':>16s}  {our_worst_err:10.4f}  {safe_med(rand_worst_errs):10.4f}  "
          f"{safe_mean(rand_worst_errs):10.4f}  {rank_werr:>4d}/{n_rand}")
    print(f"     {'mean seed err':>16s}  {our_mean_err:10.4f}  {safe_med(rand_mean_errs):10.4f}  "
          f"{safe_mean(rand_mean_errs):10.4f}  {rank_merr:>4d}/{n_rand}")
    if not math.isnan(our_mi):
        print(f"     {'mean NR iters':>16s}  {our_mi:10.2f}  {safe_med(rand_mean_iters):10.2f}  "
              f"{safe_mean(rand_mean_iters):10.2f}  {rank_iters:>4d}/{n_rand}")
    print(f"     {'NR fails':>16s}  {our_f:10d}  {safe_med(rand_fails):10.1f}  "
          f"{safe_mean(rand_fails):10.1f}  {rank_fails:>4d}/{n_rand}")
    n_iter_comparable = len(rand_mean_iters)
    if n_iter_comparable > 0 and not math.isnan(our_mi):
        better_pct = 100.0 * (1.0 - rank_iters / n_iter_comparable)
        print(f"\n     => Our constant is faster than {better_pct:.0f}% of random constants.")
    csv_rows = [
        {
            "type": "optimal_shipped",
            "constant": hex_fmt.format(our_const),
            "worst_seed_err": f"{our_worst_err:.6f}",
            "mean_seed_err": f"{our_mean_err:.6f}",
            "mean_iters": f"{our_mi:.2f}" if not math.isnan(our_mi) else "nan",
            "max_iters": our_mx,
            "fails": our_f,
        },
        {
            "type": "brute_force_best_convergence",
            "constant": hex_fmt.format(conv_best),
            "worst_seed_err": "",
            "mean_seed_err": "",
            "mean_iters": f"{conv_mi:.2f}",
            "max_iters": conv_mx,
            "fails": conv_f,
        },
    ]
    for c, w, m, mi, mx, f in rand_data[:30]:
        csv_rows.append({
            "type": "random",
            "constant": hex_fmt.format(c),
            "worst_seed_err": f"{w:.6f}",
            "mean_seed_err": f"{m:.6f}",
            "mean_iters": f"{mi:.2f}" if not math.isnan(mi) else "nan",
            "max_iters": mx,
            "fails": f,
        })
    csv_path = os.path.join(OUT_DIR, f"validate_magic_{fmt_key}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n     CSV saved: {os.path.abspath(csv_path)}")
    print()
    if conv_best == our_const:
        print(f"  VERDICT: {hex_fmt.format(our_const)} is the best constant (by NR convergence).")
    else:
        gap = our_mi - conv_mi if not (math.isnan(our_mi) or math.isnan(conv_mi)) else float("nan")
        print(f"  VERDICT: {hex_fmt.format(our_const)} vs best {hex_fmt.format(conv_best)}: "
              f"gap = {gap:+.2f} mean iters.")
def main():
    print("=" * 76)
    print("  BF16 / FP8 MAGIC CONSTANT CHECK")
    print("  " + "-" * 72)
    print(f"  Random baselines  = {N_RANDOM}")
    print(f"  NR threshold      = {THRESHOLD_FP32:.2e}  (fp32)")
    print(f"  NR max_iter       = {MAX_ITER}")
    print("=" * 76)
    for fmt_key, fmt_info in FORMATS.items():
        validate_format(fmt_key, fmt_info)
    print("\n" + "=" * 76)
    print("  DONE - all results saved to results/validate_magic_*.csv")
    print("=" * 76)
if __name__ == "__main__":
    main()
