"""Downlink SINR-constrained beamforming for four users."""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


NUM_ANA = 32
NUM_MS = 5
CARR_FREQ = 2.4e9
LIGHT_SPEED = 299_792_458.0
WAVE_LEN = LIGHT_SPEED / CARR_FREQ
ANTENNA_DIS = WAVE_LEN / 2

CHAN_GAIN = np.ones(NUM_MS, dtype=np.complex128)
ANGLE_MS = np.array([-33, -10, 15, 45,55])
sigma_dbm = -70
sigma_db = 10**(sigma_dbm / 10)
sigma_lin = sigma_db / 1000
sigma_k = np.sqrt(sigma_lin) * np.ones(NUM_MS)
gamma_dB = 10
gamma_0 = np.sqrt(10 ** (gamma_dB / 10))

indices = np.arange(NUM_ANA)
h = np.zeros((NUM_ANA, NUM_MS), dtype=np.complex128)
for i in range(NUM_MS):
    phase = 1j * 2 * np.pi * ANTENNA_DIS * np.sin(np.deg2rad(ANGLE_MS[i])) / WAVE_LEN
    h[:, i] = CHAN_GAIN[i] * np.exp(phase * indices)

w = cp.Variable((NUM_ANA, NUM_MS), complex=True)
t = cp.Variable()

H = h.T
Hw = H @ w
diag_mask = np.eye(NUM_MS)
Hw_offdiag = Hw - cp.multiply(diag_mask, Hw)
sigma_column = cp.Constant(sigma_k.reshape(-1, 1))
H2 = cp.hstack([Hw_offdiag, sigma_column])

constraints = [cp.norm(w, "fro") <= t]
for i in range(NUM_MS):
    hwi = h[:, i].T @ w[:, i]
    constraints += [
        cp.real(hwi) >= 0,  # align the useful signal's phase
        cp.imag(hwi) == 0,
        # SOC SINR constraint: interference-plus-noise norm ≤ (1/γ) * desired signal
        cp.norm(H2[i, :], 2) <= (1 / gamma_0) * cp.real(hwi),
    ]

objective = cp.Minimize(t)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)

angles = np.arange(-90, 91,0.01)
steering_vec_plot = np.exp(
    -1j
    * np.outer(np.sin(np.deg2rad(angles)), indices)
    * 2
    * np.pi
    * ANTENNA_DIS
    / WAVE_LEN
)
power = np.abs(np.conj(w.value).T @ steering_vec_plot.T) ** 2
pattern_db = 10 * np.log10(np.maximum(power, 1e-12))

fig, ax = plt.subplots()
for i, label in enumerate(
    [r"MS at $-33^\circ$", r"MS at $-10^\circ$", r"MS at $15^\circ$", r"MS at $45^\circ$",r"MS at $55^\circ$"]
):
    ax.plot(angles, pattern_db[i, :], label=label)
ax.set_title("Minimizing the total transmit power")
ax.grid(True)
ax.set_xlabel("Angle (degree)")
ax.set_ylabel("Angle Response (dB)")
ax.set_xlim([-90, 90])
ax.legend(loc="lower right")
fig.savefig("problem4b_my_answer.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
