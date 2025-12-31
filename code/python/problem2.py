"""Average sidelobe energy minimization for a 40-element ULA."""

import pickle

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


NUM_ANA = 40  # number of array elements
CARR_FREQ = 2.4e9  # Hz carrier frequency
LIGHT_SPEED = 299_792_458.0  # m/s
WAVE_LEN = LIGHT_SPEED / CARR_FREQ  # wavelength λ
ANTENNA_DIS = WAVE_LEN / 2  # λ/2 spacing

# Desired mainlobe occupies [theta_l, theta_u] degrees.
theta_l = 10
theta_u = 30
ANGLE_DES = (theta_l + theta_u) / 2  # look direction θ0

indices = np.arange(NUM_ANA)
# Steering_des implements a(θ0) = [1, e^{-jkd sin θ0}, …]ᵀ for a ULA with spacing d.
steering_phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_DES)) / WAVE_LEN
Steering_des = np.exp(steering_phase * indices)
Beam_weight_MF = Steering_des / NUM_ANA  # matched-filter weights for reference


def steering_column(angle_deg: float) -> np.ndarray:
    """Steering vector a(θ) for a ULA: exp(-j 2π d sinθ / λ · antenna_index)."""
    phase = -1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(angle_deg)) / WAVE_LEN
    return np.exp(phase * indices)[:, None]


# Build the sidelobe energy matrix P by integrating steering vectors
# outside the desired passband. Each outer product measures power received
# when scanning toward an interferer angle, so summing gives the average sidelobe energy.
P_matrix = np.zeros((NUM_ANA, NUM_ANA), dtype=np.complex128)
for angle in range(-90, theta_l + 1):
    vec = steering_column(angle)
    P_matrix += vec @ vec.conj().T
for angle in range(theta_u, 91):
    vec = steering_column(angle)
    P_matrix += vec @ vec.conj().T

w = cp.Variable(NUM_ANA, complex=True)  # beamforming weights
objective = cp.Minimize(cp.quad_form(w, P_matrix))  # wᴴPw objective
# Force unit gain toward the look direction so the beampattern peaks at ANGLE_DES.
constraints = [cp.sum(cp.multiply(np.conj(Steering_des), w)) == 1]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL)

# Dense sweep to visualize the beampattern across the visible region.
angles = np.arange(-90, 91)
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
ax.text(-80, -10, fmt)  # annotate with design parameters
ax.grid(True)
ax.set_xlim([-90, 90])
ax.set_ylim([-120, 0])
fig.savefig("problem2.png", dpi=300, bbox_inches="tight")
with open("problem2.fig.pickle", "wb") as fig_file:
    pickle.dump(fig, fig_file)
plt.show()
plt.close(fig)
