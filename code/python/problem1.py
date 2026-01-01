"""Bound-constrained least-squares demo from the project description."""

import numpy as np
import cvxpy as cp


# Random but repeatable test instance so results stay stable.
rng = np.random.default_rng(1234)

m, n = 4, 2  # m equations, n variables as in the homework statement
A = rng.standard_normal((m, n))
b = rng.standard_normal(m)
l = 0.1  # lower bounds ℓ
u = 3.14  # upper bounds u

x = cp.Variable(n)
objective = cp.Minimize(cp.norm2(A @ x - b))  # ‖Ax - b‖2 objective
constraints = [x >= l, x <= u]  # componentwise box constraints
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)

print("A =")
print(A)
print("\nb =")
print(b)
print("\nMin bound is")
print(l)
print("\nMax bound is")
print(u)
print("\nConstrained value of x")
print(x.value)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             