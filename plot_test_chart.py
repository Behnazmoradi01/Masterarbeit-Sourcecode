import matplotlib.pyplot as plt
iterations = [0, 1, 2, 3, 4, 5, 6]
nr_error = [1e0, 5e-1, 2e-2, 3e-4, 8e-8, 5e-15, 1e-16]
gs_error = [1e0, 6e-1, 5e-2, 2e-3, 6e-6, 4e-11, 2e-16]
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(iterations, nr_error, "o-", label="Newton-Raphson")
ax.semilogy(iterations, gs_error, "s--", label="Goldschmidt (classic)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Error")
ax.set_title("Convergence Comparison")
ax.legend()
ax.grid(True)
fig.savefig("convergence_test.png", dpi=300, bbox_inches="tight")
print("Saved: convergence_test.png")
plt.show()
