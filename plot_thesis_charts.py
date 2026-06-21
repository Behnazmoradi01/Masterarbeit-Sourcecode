import csv
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nr_reciprocal.basic import newton_reciprocal
from src.gs_division.basic import goldschmidt_classic_divide
from src.gs_binomial.basic import goldschmidt_binomial_divide

# تنظیمات کلی فونت
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

TABLE1_PATH = os.path.join(PROJECT_ROOT, "results", "stress", "table1_iterations.csv")
TABLE3_PATH = os.path.join(PROJECT_ROOT, "results", "stress", "table3_total_ops.csv")

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- نمودار 1: همگرایی ---
print("Running algorithms for b=3.7 ...")
b_test = 3.7
_, _, nr_errors = newton_reciprocal(b_test, max_iter=20, threshold=1e-16)
_, _, gs_errors = goldschmidt_classic_divide(1.0, b_test, max_iter=20, threshold=1e-16)
_, _, gb_errors = goldschmidt_binomial_divide(1.0, b_test, max_iter=20, threshold=1e-16, order=3)

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.semilogy(range(1, len(nr_errors)+1), nr_errors,
             "o-", linewidth=2, markersize=7, color="steelblue", label="Newton-Raphson")
ax1.semilogy(range(1, len(gs_errors)+1), gs_errors,
             "s--", linewidth=2, markersize=7, color="tomato", label="Goldschmidt (classic)")
ax1.semilogy(range(1, len(gb_errors)+1), gb_errors,
             "D-.", linewidth=2, markersize=7, color="seagreen", label="Goldschmidt (binomial, order 3)")
ax1.set_xlabel("Iteration number")
ax1.set_ylabel("Relative error")
ax1.set_title(f"Convergence for b = {b_test}")
ax1.legend(loc="upper right")
ax1.grid(True, which="both", linestyle=":", alpha=0.5)
max_k = max(len(nr_errors), len(gs_errors), len(gb_errors))
ax1.set_xticks(range(1, max_k + 1))
fig1.savefig("chart1_convergence.png", dpi=300, bbox_inches="tight")
print("Saved: chart1_convergence.png")


# --- نمودار 2: تعداد iteration در هر precision ---
print("Reading iteration data ...")
rows_t1 = read_csv(TABLE1_PATH)
iter_data = {}
for row in rows_t1:
    iter_data.setdefault(row["algorithm"], {})[row["precision"]] = int(row["max_iterations"])

algo_keys = ["Newton-Raphson", "Goldschmidt classic", "Goldschmidt binomial"]
precisions = ["fp8", "bf16", "fp32"]
prec_labels = ["FP8", "BF16", "FP32"]
colors = ["steelblue", "tomato", "seagreen"]
algo_display = ["Newton-Raphson", "Goldschmidt (classic)", "Goldschmidt (binomial)"]

fig2, ax2 = plt.subplots(figsize=(8, 5))
x = np.arange(len(prec_labels))
bar_width = 0.25
for i, akey in enumerate(algo_keys):
    vals = [iter_data[akey][p] for p in precisions]
    bars = ax2.bar(x + i*bar_width, vals, bar_width, label=algo_display[i], color=colors[i])
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(h),
                 ha="center", va="bottom", fontsize=9)

ax2.set_xlabel("Floating-point format")
ax2.set_ylabel("Worst-case iterations")
ax2.set_title("Maximum iterations per precision\n(stress test, N=200k random inputs)")
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(prec_labels)
ax2.legend(loc="upper left")
ax2.grid(axis="y", linestyle="-", alpha=0.4)
ax2.set_ylim(0, max(v for d in iter_data.values() for v in d.values()) + 2)
fig2.savefig("chart2_iterations.png", dpi=300, bbox_inches="tight")
print("Saved: chart2_iterations.png")


# --- نمودار 3: عملیات حسابی در FP32 ---
rows_t3 = read_csv(TABLE3_PATH)
fp32_rows = [r for r in rows_t3 if r["precision"] == "fp32"]

algo_names, mul_vals, addsub_vals, shift_vals = [], [], [], []
for r in fp32_rows:
    n = r["algorithm"]
    algo_names.append("Binomial\nGoldschmidt" if "binomial" in n.lower()
                      else "Goldschmidt\n(classic)" if "classic" in n.lower()
                      else "Newton-\nRaphson")
    mul_vals.append(int(r["total_MUL"]))
    addsub_vals.append(int(r["total_ADD"]) + int(r["total_SUB"]))
    shift_vals.append(int(r["total_SHIFT"]))

fig3, ax3 = plt.subplots(figsize=(7, 5))
x3 = np.arange(len(algo_names))
w = 0.22
ax3.bar(x3 - w, mul_vals,    w, label="MUL",       color="darkorange")
ax3.bar(x3,     addsub_vals, w, label="ADD + SUB",  color="slateblue")
ax3.bar(x3 + w, shift_vals,  w, label="SHIFT",      color="sienna")
ax3.set_xticks(x3)
ax3.set_xticklabels(algo_names)
ax3.set_ylabel("Number of operations")
ax3.set_title("Total operations at FP32 (worst case)")
ax3.legend()
ax3.grid(axis="y", alpha=0.4)
# TODO: maybe annotate bars with numbers later
fig3.savefig("chart3_cost.png", dpi=300, bbox_inches="tight")
print("Saved: chart3_cost.png")


# --- نمودار 4: دقت در برابر هزینه ---
_, _, nr_fp32 = newton_reciprocal(b_test, precision="fp32", max_iter=50)
_, _, gs_fp32 = goldschmidt_classic_divide(1.0, b_test, precision="fp32", max_iter=50)
_, _, gb_fp32 = goldschmidt_binomial_divide(1.0, b_test, precision="fp32", max_iter=50, order=3)

def neg_log10(e):
    return -math.log10(e) if e > 0 else 16.0

acc = [neg_log10(nr_fp32[-1]), neg_log10(gs_fp32[-1]), neg_log10(gb_fp32[-1])]
ops = [int(r["total_MUL"]) + int(r["total_ADD"]) + int(r["total_SUB"]) + int(r["total_SHIFT"])
       for r in fp32_rows]
eff_labels = ["Newton-Raphson", "Goldschmidt (classic)", "Goldschmidt (binomial)"]

fig4, ax4 = plt.subplots(figsize=(7, 5))
for i, (o, a, lab) in enumerate(zip(ops, acc, eff_labels)):
    ax4.scatter(o, a, s=150, marker=["o","s","D"][i], color=colors[i],
                edgecolors="black", linewidths=0.6, label=lab, zorder=3)

ax4.set_xlabel("Total operations (FP32, worst case)")
ax4.set_ylabel("Accuracy  (-log10 relative error)")
ax4.set_title(f"Efficiency: cost vs. accuracy  (b = {b_test})")
ax4.legend()
ax4.grid(True, alpha=0.4)
fig4.savefig("chart4_efficiency.png", dpi=300, bbox_inches="tight")
print("Saved: chart4_efficiency.png")

print("\nAll charts saved.")
if os.environ.get("NO_SHOW") != "1":
    plt.show()
