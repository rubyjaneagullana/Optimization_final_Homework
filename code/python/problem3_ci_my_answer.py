"""Worst-case sidelobe minimization with a 0°–40° mainlobe."""


import pickle

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


theta_l = -10
theta_u = 10
NUM_ANA = 40
FREQ = 2.4e9
WAVE_LEN = 3e8 / FREQ  # approximate lightspeed for quick calculations
ANTENNA_DIS = WAVE_LEN / 2

ANGLE_DES = (theta_l + theta_u) / 2

indices = np.arange(NUM_ANA)
steering_phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_DES)) / WAVE_LEN
Steering_des = np.exp(steering_phase * indices)


def steering_column(angle_deg: float) -> np.ndarray:
    phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(angle_deg)) / WAVE_LEN
    return np.exp(phase * indices)[:, None]


w = cp.Variable(NUM_ANA, complex=True)
t = cp.Variable()  # epigraph variable capturing the worst sidelobe power

# Enforce unity response toward ANGLE_DES to keep the desired mainlobe untouched.
constraints = [cp.sum(cp.multiply(np.conj(Steering_des), w)) == 1]

# Sidelobe suppression constraints: |wᴴa(θ)|² ≤ t for every interference angle.
for angle in range(-90, theta_l + 1):
    vec = steering_column(angle)
    P_matrix = vec @ vec.conj().T
    constraints.append(cp.quad_form(w, P_matrix) <= t)
for angle in range(theta_u, 91):
    vec = steering_column(angle)
    P_matrix = vec @ vec.conj().T
    constraints.append(cp.quad_form(w, P_matrix) <= t)

problem = cp.Problem(cp.Minimize(t), constraints)
try:
    # Clarabel usually succeeds, but fall back to any available solver if needed.
    problem.solve(solver=cp.MOSEK)
except cp.SolverError:
    problem.solve()

angles = np.arange(-90, 91,0.1)
steering_vec_plot = np.exp(
    -1j * np.outer(np.sin(np.deg2rad(angles)), indices) * 2 * np.pi * ANTENNA_DIS / WAVE_LEN
)
# Plot |wᴴa(θ)|² across all directions to validate the design.
pattern_db = 10 * np.log10(np.abs(np.conj(w.value) @ steering_vec_plot.T) ** 2 + 1e-12)

fig, ax = plt.subplots()
ax.plot(angles, pattern_db)
ax.set_title("Problem3(ci)Minimizing the Worst-case Sidelobe")
fmt = (
    rf"$M={NUM_ANA},\ \theta_d={ANGLE_DES},\ "
    rf"\theta_\ell={theta_l},\ \theta_u={theta_u}$"
    )
ax.grid(True)
ax.set_xlabel("Angle (degree)")
ax.set_ylabel("Angle Response (dB)")
ax.text(-80, -10, fmt)
ax.set_xlim([-90, 90])
ax.set_ylim([-120, 0])
fig.savefig("problem3_ci_my_answer.png", dpi=300, bbox_inches="tight")
with open("problem3_ci_my_answer.fig.pickle", "wb") as fig_file:
    pickle.dump(fig, fig_file)
plt.show()
plt.close(fig)


