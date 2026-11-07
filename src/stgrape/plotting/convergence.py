"""Convergence plots matching the existing paper notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import PAPER_COLORS, apply_paper_style


def plot_convergence(
    *,
    baseline_series: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    grape_times: np.ndarray | None = None,
    grape_losses: np.ndarray | None = None,
    st_times: np.ndarray | None = None,
    st_losses: np.ndarray | None = None,
    exact_times: np.ndarray | None = None,
    exact_losses: np.ndarray | None = None,
    exact_iterations: np.ndarray | None = None,
    inset_iterations: Mapping[str, np.ndarray] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot ODE-GRAPE baselines, ST-GRAPE, and true-loss checks.

    The marker/color choices follow the original ``loss_curves.ipynb``.
    Losses must be positive because the vertical axis is logarithmic.
    ``grape_times``/``grape_losses`` remain a compact single-baseline input;
    benchmark plots should pass explicit ``baseline_series`` keys.
    """

    apply_paper_style()
    if ax is None:
        created_figure, created_ax = plt.subplots(figsize=(8, 6))
        figure = cast(Figure, created_figure)
        ax = cast(Axes, created_ax)
    else:
        figure = cast(Figure, ax.figure)

    if grape_times is not None and grape_losses is not None:
        ax.plot(
            grape_times,
            grape_losses,
            color=PAPER_COLORS["grape"],
            marker="x",
            markevery=max(1, len(grape_losses) // 30),
            linestyle="none",
            label=r"$1-J$ in GRAPE",
        )
    baseline_styles = {
        "dense_grape": (PAPER_COLORS["dense_ode"], "x", "Dense GRAPE"),
        "block_sparse_grape": (
            PAPER_COLORS["block_sparse_ode"],
            "s",
            "Block-sparse GRAPE",
        ),
        "dense": (PAPER_COLORS["dense_ode"], "x", "Dense GRAPE"),
        "block-sparse": (
            PAPER_COLORS["block_sparse_ode"],
            "s",
            "Block-sparse GRAPE",
        ),
    }
    if baseline_series is not None:
        for key, (times, losses) in baseline_series.items():
            color, marker, label = baseline_styles.get(key, (None, ".", key))
            ax.plot(
                times,
                losses,
                color=color,
                marker=marker,
                markevery=max(1, len(losses) // 30),
                linestyle="-",
                linewidth=1.0,
                markersize=4,
                label=label,
            )
    # ST-GRAPE is optimized against the approximate loss, but benchmark plots
    # emphasize its periodically sampled true loss.  The approximate trajectory
    # remains available in the saved result for diagnostic plots.
    if st_times is not None and st_losses is not None and exact_losses is None:
        ax.plot(
            st_times,
            st_losses,
            color=PAPER_COLORS["trotter"],
            marker="o",
            markevery=max(1, len(st_losses) // 30),
            linestyle="none",
            label=r"$1-\hat{J}$ in ST-GRAPE",
        )
    if exact_times is not None and exact_losses is not None and len(exact_losses):
        ax.plot(
            exact_times,
            exact_losses,
            color=PAPER_COLORS["st_grape"],
            marker="^",
            markevery=max(1, len(exact_losses) // 30),
            markersize=4,
            linestyle="-",
            label=r"$1-J$ in ST-GRAPE",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, loc="upper left" if inset_iterations else "best")

    if inset_iterations:
        inset = ax.inset_axes((0.16, 0.18, 0.4, 0.35))
        if "grape" in inset_iterations:
            inset.plot(inset_iterations["grape"], color=PAPER_COLORS["grape"])
        for key, losses in inset_iterations.items():
            if key in {"grape", "st_grape"}:
                continue
            color, _, _ = baseline_styles.get(key, (None, ".", key))
            inset.plot(losses, color=color)
        if exact_iterations is not None and exact_losses is not None and len(exact_losses):
            inset.plot(
                exact_iterations,
                exact_losses,
                color=PAPER_COLORS["st_grape"],
            )
        elif "st_grape" in inset_iterations:
            inset.plot(inset_iterations["st_grape"], color=PAPER_COLORS["trotter"])
        inset.set_yscale("log")
        inset.set_xlabel("Iterations", fontsize=8)
        inset.set_ylabel("Loss", fontsize=8)
        inset.tick_params(labelsize=8)
    return figure, ax
