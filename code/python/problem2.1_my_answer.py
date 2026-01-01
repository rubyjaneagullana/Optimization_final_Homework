"""Average sidelobe suppression for a narrow mainlobe (5°15°)."""

import pickle

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


NUM_ANA = 40  # number of transmit antennas
CARR_FREQ = 2.4e9  # carrier frequency in Hz
LIGHT_SPEED = 299_792_458.0  # m/s
WAVE_LEN = LIGHT_SPEED / CARR_FREQ
ANTENNA_DIS = WAVE_LEN / 2  # standard half-wavelength spacing

theta_l = 5  # mainlobe lower edge
theta_u = 15  # mainlobe upper edge
ANGLE_DES = (theta_l + theta_u) / 2  # boresight direction

indices = np.arange(NUM_ANA)
# Steering_des matches the textbook ULA steering vector at ANGLE_DES.
steering_phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_DES)) / WAVE_LEN
Steering_des = np.exp(steering_phase * indices)
Beam_weight_MF = Steering_des / NUM_ANA


def steering_column(angle_deg: float) -> np.ndarray:
    """Compute a(θ) = [e^{-jkdsinθ·0}, …, e^{-jkdsinθ·(M-1)}]ᵀ for the given angle."""
    phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(angle_deg)) / WAVE_LEN
    return np.exp(phase * indices)[:, None]


# Build P by summing steering-vector outer products over sidelobe angles.
P_matrix = np.zeros((NUM_ANA, NUM_ANA), dtype=np.complex128)
for angle in range(-90, theta_l + 1):
    vec = steering_column(angle)
    P_matrix += vec @ vec.conj().T
for angle in range(theta_u, 91):
    vec = steering_column(angle)
    P_matrix += vec @ vec.conj().T

w = cp.Variable(NUM_ANA, complex=True)
objective = cp.Minimize(cp.quad_form(w, P_matrix))  # average sidelobe energy
# Unity gain constraint keeps the desired look direction undistorted.
constraints = [cp.sum(cp.multiply(np.conj(Steering_des), w)) == 1]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)

# Sweep ±90° in 0.01° steps to view the full beampattern.
angles = np.arange(-90, 91, 0.01)
angles_rad = np.deg2rad(angles)
steering_vec_plot = np.exp(
    -1j * np.outer(np.sin(angles_rad), indices) * 2 * np.pi * ANTENNA_DIS / WAVE_LEN
)
pattern_db = 10 * np.log10(np.abs(np.conj(w.value) @ steering_vec_plot.T) ** 2 + 1e-12)

fig, ax = plt.subplots()
ax.plot(angles, pattern_db)
ax.set_title("Minimizing the Average Sidelobe")
fmt = (
    rf"$M={NUM_ANA},\ \theta_d={ANGLE_DES},\ "
    rf"\theta_\ell={theta_l},\ \theta_u={theta_u}$"
)
ax.set_xlabel("Angle (degree)")
ax.set_ylabel("Angle Response (dB)")
ax.text(-80, -10, fmt)
ax.grid(True)
ax.set_xlim([-90, 90])
ax.set_ylim([-120, 0])
fig.savefig("problem2_1_my_answer.png", dpi=300, bbox_inches="tight")
with open("problem2_1.fig.pickle", "wb") as fig_file:
    pickle.dump(fig, fig_file)
plt.show()
plt.close(fig)
