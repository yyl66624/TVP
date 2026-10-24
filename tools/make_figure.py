#!/usr/bin/env python3
"""Figure: uncalibrated MVC vs calibrated (all-fit MVC) R@1 per benchmark,
sorted by uncalibrated R@1.  The dumbbell length is the calibration gain; the
ordering shows that the gain shrinks as the raw descriptor gets stronger.
Writes figures/gain_vs_raw.{png,pdf}."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = [
    # short label, raw R@1, all-fit R@1
    ("Freiburg 12w", 18.7, 35.2),
    ("Freiburg 15w", 29.4, 47.8),
    ("Nordland", 31.0, 67.7),
    ("MSLS-val", 39.6, 69.3),
    ("RobotCar sun→night", 54.8, 56.6),
    ("RobotCar overcast→night", 71.9, 84.9),
    ("ITLP indoor", 73.2, 81.8),
    ("ITLP winter→summer", 73.5, 75.6),
    ("Eynsham", 79.2, 90.2),
    ("CoVPR indoor", 81.7, 84.8),
    ("NYC indoor", 83.4, 82.1),
    ("RobotCar overcast→snow", 86.2, 91.7),
    ("st_lucia", 89.6, 98.0),
    ("ITLP day→night 07/08", 89.9, 93.5),
    ("ITLP day→night 05/06", 90.1, 91.9),
]

BLUE = "#2a78d6"
GRAY = "#8d8c86"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def main():
    os.makedirs("figures", exist_ok=True)
    rows = sorted(DATA, key=lambda d: d[1])  # weakest raw at the top
    n = len(rows)
    ys = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for y, (name, r, a) in zip(ys, rows):
        ax.plot([r, a], [y, y], color=GRAY, lw=1.2, solid_capstyle="round", zorder=1)
        ax.scatter(r, y, s=18, color=GRAY, edgecolor="white", linewidth=0.5, zorder=2)
        ax.scatter(a, y, s=20, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
        gain = a - r
        ax.annotate(
            f"{gain:+.1f}", (max(r, a), y), xytext=(4, 0), textcoords="offset points",
            ha="left", va="center", fontsize=5.6,
            color=INK2 if gain >= 0 else "#c0392b", zorder=4,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in rows], fontsize=5.8, color=INK)
    ax.set_xlabel("R@1 (%)", fontsize=7, color=INK)
    ax.set_xlim(14, 111)
    ax.set_ylim(-0.8, n - 0.2)
    ax.tick_params(axis="x", labelsize=6, colors=INK2, length=2, width=0.6)
    ax.tick_params(axis="y", length=0, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # legend (two series, so identity is never colour-alone; also direct marks)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
               markeredgecolor="white", markersize=3.6, label="uncalibrated MVC"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=3.8, label="calibrated (all-fit)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.6, frameon=False,
              handletextpad=0.2, borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/gain_vs_raw.{ext}", facecolor="white")
    print("wrote figures/gain_vs_raw.png")


if __name__ == "__main__":
    main()
