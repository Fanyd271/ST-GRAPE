"""Comparison plots for piecewise-constant control fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import PAPER_COLORS, apply_paper_style


def plot_control_comparison(
    controls: Mapping[str, np.ndarray],
    *,
    time_step_us: float,
    n_qubits: int,
) -> tuple[Figure, np.ndarray]:
    """Plot X/Y PWC fields for several controls on a shared axis grid.

    Control channels must follow the package convention
    ``(X_1, ..., X_n, Y_1, ..., Y_n)`` and use angular-frequency units.  The
    displayed amplitudes are converted to MHz by dividing by ``2*pi``.
    """

    if time_step_us <= 0:
        raise ValueError("time_step_us must be positive.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    if not controls:
        raise ValueError("controls cannot be empty.")

    arrays = {key: np.asarray(values, dtype=float) for key, values in controls.items()}
    shapes = {values.shape for values in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("All controls must have the same shape.")
    n_controls, n_time_steps = next(iter(shapes))
    if n_controls != 2 * n_qubits or n_time_steps < 1:
        raise ValueError(
            f"Expected controls with shape ({2 * n_qubits}, n_time_steps); "
            f"received {(n_controls, n_time_steps)}."
        )

    styles = {
        "initial": (PAPER_COLORS["initial"], "-", r"$u_{\rm init}$"),
        "grape": (PAPER_COLORS["grape"], "--", "GRAPE"),
        "dense_grape": (PAPER_COLORS["dense_ode"], "--", "Dense GRAPE"),
        "block_sparse_grape": (
            PAPER_COLORS["block_sparse_ode"],
            ":",
            "Block-sparse GRAPE",
        ),
        "st_grape": (PAPER_COLORS["st_grape"], "-.", "ST-GRAPE"),
        "dense": (PAPER_COLORS["dense_ode"], "--", "Dense GRAPE"),
        "block-sparse": (
            PAPER_COLORS["block_sparse_ode"],
            ":",
            "Block-sparse GRAPE",
        ),
        "ST": (PAPER_COLORS["st_grape"], "-.", "ST-GRAPE"),
    }
    apply_paper_style()
    created_figure, created_axes = plt.subplots(
        n_qubits,
        2,
        figsize=(10.0, max(3.6, 2.0 * n_qubits + 0.8)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    figure = cast(Figure, created_figure)
    axes = np.asarray(created_axes, dtype=object)
    times_ns = 1e3 * time_step_us * np.arange(n_time_steps + 1)

    for qubit in range(n_qubits):
        for quadrature, name in enumerate(("X control", "Y control")):
            axis = cast(Axes, axes[qubit, quadrature])
            channel = qubit + quadrature * n_qubits
            for key, values in arrays.items():
                color, line_style, label = styles.get(key, (None, "-", key))
                amplitudes_mhz = values[channel] / (2.0 * np.pi)
                axis.step(
                    times_ns,
                    np.append(amplitudes_mhz, amplitudes_mhz[-1]),
                    where="post",
                    color=color,
                    linestyle=line_style,
                    linewidth=1.1,
                    label=label,
                )
            axis.grid(alpha=0.2, linewidth=0.6)
            if qubit == 0:
                axis.set_title(name)
        cast(Axes, axes[qubit, 0]).set_ylabel(f"Qubit {qubit + 1}\nAmplitude (MHz)")

    for axis in axes[-1]:
        cast(Axes, axis).set_xlabel("Time (ns)")
    handles, labels = cast(Axes, axes[0, 0]).get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(arrays),
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure, axes
