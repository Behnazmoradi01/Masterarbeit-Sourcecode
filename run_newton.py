from nr_reciprocal.basic import newton_reciprocal_result
res = newton_reciprocal_result(7.0, precision="fp32", x0_strategy="magic")
print(f"Final approximation : {res.x:.15e}")
print(f"Iterations          : {res.iterations}")
print(f"Final relative error: {res.errors[-1]:.4e}")
print(f"Converged           : {res.converged}")
