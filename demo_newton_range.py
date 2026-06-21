from nr_reciprocal.basic import newton_reciprocal_result, hardware_initial_guess
B_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 7.0, 10.0, 42.0, 100.0, 1e4, 1e8]
STRATEGIES = ["linear", "magic", "magic_bf16"]
THRESHOLD = 1e-12
print(f"{'b':>12s}  {'strategy':>12s}  {'x0':>14s}  {'b*x0':>10s}  "
      f"{'iters':>5s}  {'final_err':>12s}  {'1/b':>14s}  {'x_final':>14s}")
print("-" * 105)
for b in B_VALUES:
    for strat in STRATEGIES:
        x0 = hardware_initial_guess(b, strategy=strat)
        res = newton_reciprocal_result(
            b, threshold=THRESHOLD, max_iter=50, x0_strategy=strat,
        )
        print(
            f"{b:12.4g}  {strat:>12s}  {x0:14.6e}  {b * x0:10.6f}  "
            f"{res.iterations:5d}  {res.errors[-1]:12.4e}  "
            f"{1.0 / b:14.6e}  {res.x:14.6e}"
        )
    print()
