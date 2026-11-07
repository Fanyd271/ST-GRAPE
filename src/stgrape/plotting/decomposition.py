"""Plots for decomposed robust-control objectives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import apply_paper_style


@dataclass(frozen=True)
class LossDecompositionSeries:
    """One GRAPE trajectory split into fidelity loss and robust penalty."""

    n_qubits: int
    times_s: np.ndarray
    fidelity_losses: np.ndarray
    robustness_penalties: np.ndarray
    time_limit_hours: float | None = None
    time_limit_reached: bool = False


def plot_loss_decomposition(
    series: Sequence[LossDecompositionSeries],
) -> tuple[Figure, np.ndarray]:
    """Plot the two objective contributions used in the paper's Fig. 7."""

    if not series:
        raise ValueError("At least one loss-decomposition series is required.")
    apply_paper_style()
    created_figure, created_axes = plt.subplots(
        len(series),
        1,
        figsize=(7.2, max(3.5, 3.2 * len(series))),
        squeeze=False,
    )
    figure = cast(Figure, created_figure)
    axes = np.asarray(created_axes, dtype=object).reshape(-1)

    for panel, (axis_object, values) in enumerate(zip(axes, series, strict=True)):
        axis = cast(Axes, axis_object)
        times = np.asarray(values.times_s, dtype=float)
        fidelity = np.asarray(values.fidelity_losses, dtype=float)
        penalty = np.asarray(values.robustness_penalties, dtype=float)
        if times.ndim != 1 or fidelity.shape != times.shape or penalty.shape != times.shape:
            raise ValueError("Times, fidelity losses, and penalties must be equal 1D arrays.")
        if not len(times):
            raise ValueError("Loss-decomposition series cannot be empty.")

        times_hours = times / 3600.0
        axis.plot(
            times_hours,
            np.maximum(fidelity, np.finfo(float).eps),
            color="#2166ac",
            linewidth=1.6,
            label=r"Fidelity loss $1-F$",
        )
        axis.plot(
            times_hours,
            np.maximum(penalty, np.finfo(float).eps),
            color="#d6604d",
            linewidth=1.6,
            linestyle="--",
            label="Robust penalty",
        )
        if values.time_limit_reached:
            limit = (
                float(values.time_limit_hours)
                if values.time_limit_hours is not None
                else float(times_hours[-1])
            )
            axis.axvline(
                limit,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
                label="Time limit" if panel == 0 else None,
            )
        panel_label = chr(ord("a") + panel)
        status = "time limit reached" if values.time_limit_reached else "converged"
        axis.set_title(
            rf"({panel_label}) $n={values.n_qubits}$ qubits ({status})",
            loc="left",
        )
        axis.set_yscale("log")
        axis.set_xlabel("Wall-clock Time (h)")
        axis.set_ylabel("Loss")
        axis.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        axis.legend(frameon=False, loc="best")

    figure.tight_layout()
    return figure, axes
