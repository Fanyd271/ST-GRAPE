"""Plotting implementation transcribed from the final paper notebook.

Run through stgrape.analysis; all outputs are directed outside the data archive.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RESULT_ROOT = Path(globals()["DATA_ROOT"]) / "supplementary/strength_scan"
FIGURE_DIRECTORY = Path(globals()["OUTPUT_DIR"])
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
PULSE_SEED = 0
SAVE_FIGURES = True
PLOT_FLOOR = 1e-12  # Display-only floor for a logarithmic axis.


def read_scan(task):
    path = RESULT_ROOT / f"{task}_seed{PULSE_SEED}.npz"
    with np.load(path, allow_pickle=False) as saved:
        strengths = saved["strengths"].copy()
        labels = saved["labels"].tolist()
        infidelities = saved["infidelities"].copy()
    assert infidelities.shape == (3, len(strengths), infidelities.shape[-1])
    assert labels == ["initial", "r1", "r2"]
    return strengths, infidelities


scans = {task: read_scan(task) for task in ("ghz", "qft")}
for task, (strengths, infidelities) in scans.items():
    print(f"{task.upper()}: {len(strengths)} strengths, {infidelities.shape[-1]} paired samples")


palette = {"initial": "#738087", "r1": "#b76c27", "r2": "#08748d"}
legend_text = {"initial": "Initial", "r1": r"ST-GRAPE, $r=1$", "r2": r"ST-GRAPE, $r=2$"}
labels = ("initial", "r1", "r2")
zoom_ranges = {"ghz": (0.05, 0.5), "qft": (0.0, 0.3)}
zoom_ticks = {"ghz": (0.05, 0.15, 0.2, 0.5), "qft": (0.0, 0.05, 0.1, 0.2, 0.3)}
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True, constrained_layout=True)
for ax, task, title in zip(
    axes,
    ("ghz", "qft"),
    ("(a) GHZ state preparation", "(b) QFT gate synthesis"),
    strict=True,
):
    strengths, infidelities = scans[task]
    means = infidelities.mean(axis=-1)
    for index, label in enumerate(labels):
        ax.semilogy(
            strengths,
            np.maximum(means[index], PLOT_FLOOR),
            color=palette[label],
            marker=("o", "s", "D")[index],
            markersize=3.2,
            markeredgewidth=0,
            linewidth=1.7,
            label=legend_text[label],
        )
    ax.axvline(1.0, color="#b8b8b8", linestyle="--", linewidth=0.8, zorder=0)
    ax.set(title=title, xlabel=r"Uncertainty-strength factor $s$", xlim=(0, float(strengths[-1])))
    ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Use the open lower part of the main log-scale panel for a local comparison.
    x_min, x_max = zoom_ranges[task]
    in_range = (strengths >= x_min) & (strengths <= x_max)
    robust_means = means[1:, in_range]
    y_min = max(PLOT_FLOOR, robust_means.min() / 1.4)
    y_max = robust_means.max() * 1.4
    zoom = ax.inset_axes((0.30, 0.10, 0.65, 0.39), zorder=5)
    for index, label in enumerate(labels[1:], start=1):
        zoom.plot(
            strengths[in_range],
            means[index, in_range],
            color=palette[label],
            marker=("o", "s", "D")[index],
            markersize=2.7,
            markeredgewidth=0,
            linewidth=1.4,
        )
    zoom.set_yscale("log")
    zoom.set(xlim=(max(0.0, x_min - 0.02), x_max + 0.02), ylim=(y_min, y_max))
    zoom.set_xticks(zoom_ticks[task])
    zoom.set_xticklabels([f"{value:g}" for value in zoom_ticks[task]])
    zoom.tick_params(axis="both", which="major", labelsize=6, pad=1, length=2)
    zoom.tick_params(axis="both", which="minor", length=0)
    zoom.grid(axis="y", which="major", color="#ededed", linewidth=0.45)
    zoom.set_facecolor("white")
    for spine in zoom.spines.values():
        spine.set(color="#969696", linewidth=0.7)
    zoom.text(0.04, 0.94, "zoom", transform=zoom.transAxes, va="top", fontsize=6.5, color="#666666")
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="#a0a0a0",
            linewidth=0.8,
            zorder=4,
        )
    )
axes[0].set_ylabel(r"Mean infidelity $\overline{1-F}$")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False
)
if SAVE_FIGURES:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = FIGURE_DIRECTORY / f"paired_strength_scan_seed{PULSE_SEED}.{suffix}"
        fig.savefig(target, bbox_inches="tight", dpi=300)
        print(f"Saved {target}")
plt.close("all")
