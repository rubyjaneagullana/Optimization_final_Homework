"""Plot beam patterns from problem3a, problem3b, and problem2."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(fig_path: Path):
    """Return x/y data from the first line of a stored Matplotlib figure."""
    with fig_path.open("rb") as fig_file:
        fig = pickle.load(fig_file)
    lines = [line for ax in fig.axes for line in ax.get_lines()]
    if not lines:
        raise ValueError(f"No line data in {fig_path}")
    xdata = lines[0].get_xdata()
    ydata = lines[0].get_ydata()
    plt.close(fig)
    return xdata, ydata


# (pickle filename, color, legend label)
fig_specs = [
    ("problem3a.fig.pickle", "b", "20 antenna Epigraph (problem-3a)"),
    ("problem3b.fig.pickle", "r", "40 antenna Epigraph (problem-3b)"),
    ("problem2.fig.pickle", "m", "40 antenna (problem-2a)"),
]

fig, ax = plt.subplots()
for path, color, label in fig_specs:
    x, y = load_curve(Path(path))
    ax.plot(x, y, color=color, label=label)

ax.grid(True)
ax.set_xlim([-90, 90])
ax.set_ylim([-140, 0])
ax.set_xlabel("Normal Angle (deg)")
ax.set_ylabel("Array Response (dB)")
ax.legend(loc="lower right")
fig.savefig("problem3b_1.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)
