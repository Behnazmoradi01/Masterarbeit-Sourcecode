from __future__ import annotations
import argparse
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from long_division.basic import long_divide_unsigned
def main() -> int:
    ap = argparse.ArgumentParser(description="Unsigned restoring binary long division (shift-subtract)")
    ap.add_argument("--a", type=int, required=True, help="Unsigned dividend (int)")
    ap.add_argument("--b", type=int, required=True, help="Unsigned divisor (int, must be > 0)")
    ap.add_argument("--bits", type=int, choices=[16, 32], required=True, help="Bit-width")
    args = ap.parse_args()
    q, r, iterations, ops = long_divide_unsigned(args.a, args.b, bits=args.bits)
    print(f"a={args.a} (masked to {args.bits} bits)")
    print(f"b={args.b} (masked to {args.bits} bits)")
    print(f"bits={args.bits}")
    print(f"q={q}")
    print(f"r={r}")
    print(f"iterations={iterations}")
    print("ops=")
    for k in sorted(ops.keys()):
        print(f"  {k}: {ops[k]}")
    mask = (1 << args.bits) - 1
    a_m = args.a & mask
    b_m = args.b & mask
    if b_m == 0:
        print("\nverify: skipped (b masked to 0)")
        return 0
    q_ref = a_m // b_m
    r_ref = a_m % b_m
    ok = (q == q_ref) and (r == r_ref) and (iterations == args.bits)
    print("\nverify=")
    print(f"  q_ref={q_ref}")
    print(f"  r_ref={r_ref}")
    print(f"  ok={ok}")
    return 0 if ok else 2
if __name__ == "__main__":
    raise SystemExit(main())
