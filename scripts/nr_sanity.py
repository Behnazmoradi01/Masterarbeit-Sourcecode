from __future__ import annotations
from nr_reciprocal import newton_reciprocal, threshold_for_precision
def main() -> int:
    bs = [0.5, 0.75, 1.0, 1.5, 2.0, 7.0, 10.0, 123.456]
    precs = ["fp32", "bf16", "fp8"]
    max_iter = 20
    print(f"max_iter={max_iter}")
    print("b\tprec\tthreshold\titers\tfinal_err\tconverged")
    for b in bs:
        for p in precs:
            thr = threshold_for_precision(p)
            _, iters, errs = newton_reciprocal(b, precision=p, max_iter=max_iter)
            final_err = errs[-1] if errs else float("nan")
            converged = bool(errs) and final_err < thr
            print(f"{b}\t{p}\t{thr:.3e}\t{iters}\t{final_err:.3e}\t{converged}")
    for b in [0.0, -1.0]:
        try:
            newton_reciprocal(b)
            print("FAIL: expected ValueError for b=", b)
        except ValueError:
            print("PASS: ValueError for b=", b)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
