"""Plot the beam patterns from problem2, problem2_1, and problem2_2."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(fig_path: Path):
    """Extract x/y samples from a pickled Matplotlib figure."""
    with fig_path.open("rb") as fig_file:
        fig = pickle.load(fig_file)
    lines = [line for ax in fig.axes for line in ax.get_lines()]
    if not lines:
        raise ValueError(f"No line data in {fig_path}")
    xdata = lines[0].get_xdata()
    ydata = lines[0].get_ydata()
    plt.close(fig)
    return xdata, ydata


problem_files = [
    ("problem2.fig.pickle", r"$10^\circ$-$30^\circ$ Beam", "b"),
    ("problem2_1.fig.pickle", r"$15^\circ$-$25^\circ$ Beam", "r"),
    ("problem2_2.fig.pickle", r"$0^\circ$-$40^\circ$ Beam", "g"),
]

fig, ax = plt.subplots()
for fig_name, label, color in problem_files:
    x, y = load_curve(Path(fig_name))
    ax.plot(x, y, color=color, label=label)

ax.grid(True)
ax.set_xlim([-90, 90])
ax.set_ylim([-140, 0])
ax.set_xlabel("Normal Angle (deg)")
ax.set_ylabel("Array Response (dB)")
ax.legend()
fig.savefig("problem2together.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
