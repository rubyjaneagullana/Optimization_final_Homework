"""Downlink beamforming with SINR constraints for two users."""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


NUM_ANA = 8  # BS antennas
NUM_MS = 2  # users
CARR_FREQ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVE_LEN = LIGHT_SPEED / CARR_FREQ
ANTENNA_DIS = WAVE_LEN / 2

CHAN_GAIN = np.ones(NUM_MS, dtype=np.complex128)  # flat channels
ANGLE_MS = np.array([-33, 10])  # angular locations of the two MSs
sigma_i = np.sqrt(0.01) * np.ones(NUM_MS)  # noise std dev per user
gamma_dB = 10  # SINR target in dB
gamma_0 = np.sqrt(10 ** (gamma_dB / 10))

indices = np.arange(NUM_ANA)
h = np.zeros((NUM_ANA, NUM_MS), dtype=np.complex128)
for i in range(NUM_MS):
    phase = 1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_MS[i])) / WAVE_LEN
    h[:, i] = CHAN_GAIN[i] * np.exp(phase * indices)

w = cp.Variable((NUM_ANA, NUM_MS), complex=True)
t = cp.Variable()

# Convenience matrices used for the SOC constraints.
H = h.T  # rows correspond to MS steering vectors
Hw = H @ w  # NUM_MS x NUM_MS transfer matrix between users
diag_mask = np.eye(NUM_MS)
Hw_offdiag = Hw - cp.multiply(diag_mask, Hw)
H2 = cp.hstack([Hw_offdiag, cp.Constant(sigma_i.reshape(-1, 1))])

# Total transmit power is ‖w‖_F^2; bounding it by t makes the problem convex.
constraints = [cp.norm(w, "fro") <= t]
for i in range(NUM_MS):
    hwi = h[:, i].T @ w[:, i]
    constraints += [
        cp.real(hwi) >= 0,  # phase-align the useful signal
        cp.imag(hwi) == 0,
        # SOC encoding of SINR constraint: ‖[interference; noise]‖₂ ≤ (1/γ)·signal
        cp.norm(H2[i, :], 2) <= (1 / gamma_0) * cp.real(hwi),
    ]

objective = cp.Minimize(t)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL)

angles = np.arange(-90, 91)
steering_vec_plot = np.exp(
    -1j * np.outer(np.sin(np.deg2rad(angles)), indices) * 2 * np.pi * ANTENNA_DIS / WAVE_LEN
)
pattern_db = 10 * np.log10(np.abs(np.conj(w.value).T @ steering_vec_plot.T) ** 2 + 1e-12)

fig, ax = plt.subplots()
ax.plot(angles, pattern_db.T)
ax.set_title("Minimizing the total transmit power")
ax.legend([r"MS at $-33^\circ$", r"MS at $10^\circ$"])
ax.grid(True)
ax.set_xlabel("Angle (degree)")
ax.set_ylabel("Angle Response (dB)")
ax.set_xlim([-90, 90])
fig.savefig("problem4a.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
