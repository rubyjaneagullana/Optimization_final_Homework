"""Worst-case sidelobe minimization for a 20-element ULA."""

import pickle

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


NUM_ANA = 20
CARR_FREQ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVE_LEN = LIGHT_SPEED / CARR_FREQ
ANTENNA_DIS = WAVE_LEN / 2

theta_l = 10
theta_u = 30
ANGLE_DES = (theta_l + theta_u) / 2  # target look direction

indices = np.arange(NUM_ANA)
steering_phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_DES)) / WAVE_LEN
Steering_des = np.exp(steering_phase * indices)


def steering_column(angle_deg: float) -> np.ndarray:
    """Steering vector (column) evaluated at angle_deg degrees."""
    phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(angle_deg)) / WAVE_LEN
    return np.exp(phase * indices)[:, None]


w = cp.Variable(NUM_ANA, complex=True)
t = cp.Variable()  # proxy for the largest sidelobe level across all blocked angles

# Distortionless constraint: preserve amplitude/phase toward ANGLE_DES.
constraints = [cp.sum(cp.multiply(np.conj(Steering_des), w)) == 1]

# Enumerate each sidelobe direction and force its quadratic output ≤ t.
for angle in range(-90, theta_l + 1):
    vec = steering_column(angle)
    P_matrix = vec @ vec.conj().T
    constraints.append(cp.quad_form(w, P_matrix) <= t)
for angle in range(theta_u, 91):
    vec = steering_column(angle)
    P_matrix = vec @ vec.conj().T
    constraints.append(cp.quad_form(w, P_matrix) <= t)

# Worst-case design minimizes t such that every sidelobe angle obeys |wᴴa(θ)|² ≤ t.
objective = cp.Minimize(t)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL)

angles = np.arange(-90, 91)
steering_vec_plot = np.exp(
    -1j * np.outer(np.sin(np.deg2rad(angles)), indices) * 2 * np.pi * ANTENNA_DIS / WAVE_LEN
)
# Convert the complex array response into a power pattern in dB.
pattern_db = 10 * np.log10(np.abs(np.conj(w.value) @ steering_vec_plot.T) ** 2 + 1e-12)

fig, ax = plt.subplots()
ax.plot(angles, pattern_db)
ax.set_title("Minimizing the Worst-case Sidelobe")
ax.grid(True)
ax.set_xlabel("Angle (degree)")
ax.set_ylabel("Angle Response (dB)")
ax.set_xlim([-90, 90])
fig.savefig("problem3a.png", dpi=300, bbox_inches="tight")
with open("problem3a.fig.pickle", "wb") as fig_file:
    pickle.dump(fig, fig_file)
plt.show()
plt.close(fig)
