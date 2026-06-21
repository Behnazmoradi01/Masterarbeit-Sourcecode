from __future__ import annotations
import math
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
try:
    from basic import float_quantize as _fq
    _HAS_QUANTIZE = True
except ImportError:
    _HAS_QUANTIZE = False
def _gs_classic_trace(
    a: float,
    b: float,
    *,
    max_iter: int = 50,
    quantize_fmt: str | None = None,
) -> list[dict]:
    q_exact = a / b
    m, e = math.frexp(b)
    a_k = math.ldexp(a, -e)
    b_k = m
    rows: list[dict] = []
    prev_e = None
    for k in range(1, max_iter + 1):
        f_k = 2.0 - b_k
        a_k = a_k * f_k
        b_k = b_k * f_k
        if quantize_fmt is not None and _HAS_QUANTIZE:
            a_k = _fq(a_k, quantize_fmt)
            b_k = _fq(b_k, quantize_fmt)
        e_k = abs(1.0 - b_k)
        rel_err = abs(a_k - q_exact) / abs(q_exact) if q_exact != 0 else float("inf")
        ratio2 = None
        if prev_e is not None and prev_e > 0:
            denom = prev_e ** 2
            ratio2 = e_k / denom if denom > 0 else None
        rows.append(dict(
            k=k, a_k=a_k, b_k=b_k, f_k=f_k,
            e_k=e_k, rel_err=rel_err, ratio2=ratio2,
        ))
        prev_e = e_k
        if rel_err < 1e-15:
            break
    return rows
def _gs_binomial_trace(
    a: float,
    b: float,
    *,
    order: int = 3,
    max_iter: int = 50,
    quantize_fmt: str | None = None,
) -> list[dict]:
    q_exact = a / b
    m, e = math.frexp(b)
    a_k = math.ldexp(a, -e)
    b_k = m
    rows: list[dict] = []
    prev_e = None
    for k in range(1, max_iter + 1):
        e_k_raw = 1.0 - b_k
        e2 = e_k_raw * e_k_raw
        if order == 3:
            e3 = e2 * e_k_raw
            inv_k = 1.0 + e_k_raw + e2 + e3
        else:
            inv_k = 1.0 + e_k_raw + e2
        a_k = a_k * inv_k
        b_k = b_k * inv_k
        if quantize_fmt is not None and _HAS_QUANTIZE:
            a_k = _fq(a_k, quantize_fmt)
            b_k = _fq(b_k, quantize_fmt)
        e_k = abs(1.0 - b_k)
        rel_err = abs(a_k - q_exact) / abs(q_exact) if q_exact != 0 else float("inf")
        p = order
        ratio_p = None
        if prev_e is not None and prev_e > 0:
            denom = prev_e ** p
            ratio_p = e_k / denom if denom > 0 else None
        rows.append(dict(
            k=k, a_k=a_k, b_k=b_k, inv_k=inv_k,
            e_k=e_k, rel_err=rel_err,
            ratio_p=ratio_p, p=p,
        ))
        prev_e = e_k
        if rel_err < 1e-15:
            break
    return rows
SEP = "-" * 100
def _print_classic_table(rows: list[dict], max_show: int = 6) -> None:
    header = f"{'k':>3} | {'b_k':>22} | {'f_k':>22} | {'e_k=|1-b_k|':>14} | {'rel_err':>12} | {'ratio2':>12}"
    print(header)
    print("-" * len(header))
    for r in rows[:max_show]:
        r2 = f"{r['ratio2']:.6f}" if r["ratio2"] is not None else "--"
        print(
            f"{r['k']:3d} | {r['b_k']:22.15e} | {r['f_k']:22.15e} | "
            f"{r['e_k']:14.6e} | {r['rel_err']:12.4e} | {r2:>12}"
        )
def _print_binomial_table(rows: list[dict], max_show: int = 6) -> None:
    p = rows[0]["p"] if rows else 3
    label = f"ratio{p}"
    header = f"{'k':>3} | {'b_k':>22} | {'inv_k':>22} | {'e_k=|1-b_k|':>14} | {'rel_err':>12} | {label:>12}"
    print(header)
    print("-" * len(header))
    for r in rows[:max_show]:
        rp = f"{r['ratio_p']:.6f}" if r["ratio_p"] is not None else "--"
        print(
            f"{r['k']:3d} | {r['b_k']:22.15e} | {r['inv_k']:22.15e} | "
            f"{r['e_k']:14.6e} | {r['rel_err']:12.4e} | {rp:>12}"
        )
def main() -> None:
    a = 1.0
    b_list = [
        1.1,
        0.9,
        1 + 2**-10,
        1 - 2**-10,
        2**20,
        2**-20,
    ]
    modes: list[tuple[str, str | None]] = [("float64", None)]
    if _HAS_QUANTIZE:
        modes.append(("quantized:bf16", "bf16"))
    else:
        print("WARNING: float_quantize not found. Skip quantized mode.")
        print("   TODO: Add support for optional floating-point quantization.")
        print("         (Use NR float_quantize_fmt as example).\n")
    classic_ratio2s: list[float] = []
    binomial_ratio3s: list[float] = []
    quantize_diffs: list[tuple[str, float, float]] = []
    for b in b_list:
        m_norm, e_norm = math.frexp(b)
        b_label = f"{b}" if abs(b) < 1e4 else f"2**{math.log2(b):.0f}"
        print(f"\n{'=' * 100}")
        print(f"  b = {b}  (mantissa = {m_norm:.15f}, exponent = {e_norm}, "
              f"b_0 is in [0.5, 1))")
        print(f"  a = {a},  q_exact = a/b = {a / b:.17g}")
        print(f"{'=' * 100}")
        float64_classic_final_err = None
        float64_binom_final_err = None
        for mode_name, qfmt in modes:
            print(f"\n  -- run mode: {mode_name} {'-' * 60}")
            print(f"\n  [Goldschmidt Classic]")
            rows_c = _gs_classic_trace(a, b, quantize_fmt=qfmt)
            _print_classic_table(rows_c)
            final_c = rows_c[-1] if rows_c else None
            if final_c:
                print(f"  => iters={len(rows_c)}, final_e={final_c['e_k']:.4e}, "
                      f"final_rel_err={final_c['rel_err']:.4e}")
            if qfmt is None and final_c:
                float64_classic_final_err = final_c["rel_err"]
            if qfmt is None:
                for r in rows_c[1:]:
                    if (r["ratio2"] is not None
                            and math.isfinite(r["ratio2"])
                            and r["e_k"] > 1e-14):
                        classic_ratio2s.append(r["ratio2"])
            print(f"\n  [Goldschmidt Binomial order=3]")
            rows_b = _gs_binomial_trace(a, b, order=3, quantize_fmt=qfmt)
            _print_binomial_table(rows_b)
            final_b = rows_b[-1] if rows_b else None
            if final_b:
                print(f"  => iters={len(rows_b)}, final_e={final_b['e_k']:.4e}, "
                      f"final_rel_err={final_b['rel_err']:.4e}")
            if qfmt is None and final_b:
                float64_binom_final_err = final_b["rel_err"]
            if qfmt is None:
                for r in rows_b[1:]:
                    if (r["ratio_p"] is not None
                            and math.isfinite(r["ratio_p"])
                            and r["e_k"] > 1e-14):
                        binomial_ratio3s.append(r["ratio_p"])
            if qfmt is not None and final_c and final_b:
                diff_c = final_c["rel_err"] - (float64_classic_final_err or 0)
                diff_b = final_b["rel_err"] - (float64_binom_final_err or 0)
                quantize_diffs.append((b_label, diff_c, diff_b))
    print(f"\n\n{'=' * 100}")
    print("  RESULT")
    print(f"{'=' * 100}")
    if classic_ratio2s:
        mid = classic_ratio2s[len(classic_ratio2s) // 4:]
        if mid:
            avg = sum(mid) / len(mid)
            spread = max(mid) - min(mid)
            stable = spread < 0.5
            print(f"\n  Classic ratio2 = e_{{k+1}} / e_k^2")
            print(f"    samples: {len(mid)}")
            print(f"    mean   = {avg:.6f}")
            print(f"    spread = {spread:.6e}")
            if stable:
                print(f"    [OK] ratio2 is stable -> order about 2")
            else:
                print(f"    [!!] ratio2 is not stable -- check table above")
    else:
        print("  (no ratio2 data — classic may finish in <=1 iter)")
    if binomial_ratio3s:
        mid = binomial_ratio3s[len(binomial_ratio3s) // 4:]
        if mid:
            avg = sum(mid) / len(mid)
            spread = max(mid) - min(mid)
            stable = spread < 0.5
            print(f"\n  Binomial ratio3 = e_{{k+1}} / e_k^3")
            print(f"    samples: {len(mid)}")
            print(f"    mean   = {avg:.6f}")
            print(f"    spread = {spread:.6e}")
            if stable:
                print(f"    [OK] ratio3 is stable -> order about 3")
            else:
                print(f"    [!!] ratio3 is not stable -- check table above")
    else:
        print("  (no ratio3 data — binomial may finish in <=1 iter)")
    if quantize_diffs:
        print(f"\n  Quantization effect (quantized - float64 final_rel_err):")
        any_nonzero = False
        for label, dc, db in quantize_diffs:
            tag_c = "SAME" if abs(dc) < 1e-16 else f"{dc:+.4e}"
            tag_b = "SAME" if abs(db) < 1e-16 else f"{db:+.4e}"
            if abs(dc) >= 1e-16 or abs(db) >= 1e-16:
                any_nonzero = True
            print(f"    b={label:>16s}  classic d={tag_c:>14s}   binomial d={tag_b:>14s}")
        if any_nonzero:
            print("    [OK] Quantization changes accuracy (expected for BF16).")
        else:
            print("    [!!] No difference found -- quantization may not work in the loop.")
   else:
    if not _HAS_QUANTIZE:
        print("\n  WARNING: Quantized mode did not run.")
        print("     GS modules currently have no per-iteration quantization.")
        print("     TODO: Add float_quantize_fmt to goldschmidt_classic_divide")
        print("           and to goldschmidt_binomial_divide, similar to NR.")
print()

if __name__ == "__main__":
    main()
