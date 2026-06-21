from __future__ import annotations
import argparse
import csv
import random
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from long_division.basic import long_divide_unsigned
def _avg_ops(total_ops: dict[str, int], n: int) -> dict[str, float]:
    return {k: total_ops[k] / n for k in total_ops}
def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
def main() -> int:
    ap = argparse.ArgumentParser(description="Random sweep for unsigned long division ops/consistency")
    ap.add_argument("--n", type=int, default=50_000, help="Number of random samples per bit-width")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path")
    args = ap.parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    random.seed(args.seed)
    rows: list[dict[str, object]] = []
    for bits in (16, 32):
        mask = (1 << bits) - 1
        total_ops = {
            "shifts": 0,
            "comparisons": 0,
            "subtractions": 0,
            "additions_restores": 0,
            "assignments": 0,
        }
        wrong = 0
        wrong_iters = 0
        for _ in range(args.n):
            a = random.getrandbits(bits)
            b = random.getrandbits(bits) | 1
            q, r, iterations, ops = long_divide_unsigned(a, b, bits=bits)
            q_ref = (a & mask) // (b & mask)
            r_ref = (a & mask) % (b & mask)
            if q != q_ref or r != r_ref:
                wrong += 1
            if iterations != bits:
                wrong_iters += 1
            for k in total_ops:
                total_ops[k] += ops[k]
        avg = _avg_ops(total_ops, args.n)
        print(f"bits={bits}")
        print(f"  n={args.n}")
        print(f"  iterations_expected={bits}")
        print(f"  wrong_results={wrong}")
        print(f"  wrong_iterations={wrong_iters}")
        print("  avg_ops_per_div=")
        for k in sorted(avg.keys()):
            print(f"    {k}: {avg[k]:.3f}")
        rows.append(
            {
                "bits": bits,
                "n": args.n,
                "iterations": bits,
                "wrong_results": wrong,
                "wrong_iterations": wrong_iters,
                "avg_shifts": f"{avg['shifts']:.3f}",
                "avg_comparisons": f"{avg['comparisons']:.3f}",
                "avg_subtractions": f"{avg['subtractions']:.3f}",
                "avg_additions_restores": f"{avg['additions_restores']:.3f}",
                "avg_assignments": f"{avg['assignments']:.3f}",
            }
        )
    if args.out:
        _write_csv(Path(args.out), rows)
        print(f"Wrote CSV: {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
