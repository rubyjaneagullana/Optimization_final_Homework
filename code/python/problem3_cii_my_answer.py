"""Worst-case sidelobe minimization with a narrow 15°–25° mainlobe."""

import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


theta_l = 45
theta_u = 55
NUM_ANA = 40
FREQ = 2.4e9
WAVE_LEN = 3e8 / FREQ
ANTENNA_DIS = WAVE_LEN / 2

ANGLE_DES = (theta_l + theta_u) / 2

indices = np.arange(NUM_ANA)
steering_phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_DES)) / WAVE_LEN
Steering_des = np.exp(steering_phase * indices)


def steering_column(angle_deg: float) -> np.ndarray:
    """Helper that returns a(θ) for the uniform linear array."""
    phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(angle_deg)) / WAVE_LEN
    return np.exp(phase * indices)[:, None]


w = cp.Variable(NUM_ANA, complex=True)
t = cp.Variable()  # bounds the maximum sidelobe gain among all constrained angles

# Preserve unity gain toward ANGLE_DES to ensure the desired signal is passed without distortion.
constraints = [cp.sum(cp.multiply(np.conj(Steering_des), w)) == 1]

# Build one quadratic constraint per sidelobe angle so |wᴴa(θ)|² does not exceed t.
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
    problem.solve(solver=cp.MOSEK)
except cp.SolverError:
    problem.solve()

angles = np.arange(-90, 91,0.1)
steering_vec_plot = np.exp(
    -1j * np.outer(np.sin(np.deg2rad(angles)), indices) * 2 * np.pi * ANTENNA_DIS / WAVE_LEN
)
# Inspect the final beampattern to confirm the worst-case bound.
pattern_db = 10 * np.log10(np.abs(np.conj(w.value) @ steering_vec_plot.T) ** 2 + 1e-12)

fig, ax = plt.subplots()
ax.plot(angles, pattern_db)
ax.set_title("Problem3c(ii) Minimizing the Worst-case Sidelobe ")
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
fig.savefig("problem3_cii_my_answer.png", dpi=300, bbox_inches="tight")
with open("problem3_cii_my_answer.fig.pickle", "wb") as fig_file:
    pickle.dump(fig, fig_file)
plt.show()
plt.close(fig)
