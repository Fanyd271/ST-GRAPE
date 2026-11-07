"""Runtime-scaling summary plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .style import PAPER_COLORS, apply_paper_style


def plot_scaling_summary(
    x: np.ndarray,
    grape_time: np.ndarray,
    st_time: np.ndarray,
    *,
    x_label: str,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Plot log-log cost scaling and the GRAPE/ST-GRAPE speedup.

    This is the reusable core of the existing dimension- and block-scaling
    notebooks.  Fitted exponents are shown in the legend.
    """

    apply_paper_style()
    x = np.asarray(x, dtype=float)
    grape = np.asarray(grape_time, dtype=float)
    st = np.asarray(st_time, dtype=float)
    if np.any(x <= 0) or np.any(grape <= 0) or np.any(st <= 0):
        raise ValueError("Scaling data must be strictly positive.")

    grape_fit = np.polyfit(np.log(x), np.log(grape), 1)
    st_fit = np.polyfit(np.log(x), np.log(st), 1)
    x_fit = np.geomspace(x.min(), x.max(), 200)

    figure, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    cost, speedup = axes
    cost.loglog(x, grape, "o", color=PAPER_COLORS["grape"], label="GRAPE")
    cost.loglog(x, st, "s", color=PAPER_COLORS["st_grape"], label="ST-GRAPE")
    cost.loglog(
        x_fit,
        np.exp(grape_fit[1]) * x_fit ** grape_fit[0],
        "--",
        color=PAPER_COLORS["grape"],
        label=rf"$x^{{{grape_fit[0]:.2f}}}$",
    )
    cost.loglog(
        x_fit,
        np.exp(st_fit[1]) * x_fit ** st_fit[0],
        "--",
        color=PAPER_COLORS["st_grape"],
        label=rf"$x^{{{st_fit[0]:.2f}}}$",
    )
    cost.set_ylabel("Time / step (s)")
    cost.set_title(title)
    cost.legend(frameon=False, ncol=2)

    ratios = grape / st
    speedup.semilogx(x, ratios, "o-", color=PAPER_COLORS["trotter"])
    speedup.axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    for x_value, ratio in zip(x, ratios, strict=True):
        speedup.annotate(
            f"{ratio:.1f}x",
            (x_value, ratio),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
    speedup.set_xlabel(x_label)
    speedup.set_ylabel("Speedup")
    return figure, axes
