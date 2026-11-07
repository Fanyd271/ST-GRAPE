"""Gate/state error distributions and cumulative distribution functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter

from .style import PAPER_COLORS, apply_paper_style

_NOISE_SOURCE_STYLES = {
    "static": ("#2166ac", "-", "Static"),
    "control": ("#d6604d", "--", "Control-amplitude drift"),
    "colored": ("#008837", "-.", "Non-Markovian"),
}

_METHOD_STYLES = {
    "initial": (PAPER_COLORS["initial"], "-", r"$u_{\rm init}$"),
    "ST": (PAPER_COLORS["st_grape"], "-.", "ST-GRAPE"),
    "dense": (PAPER_COLORS["dense_ode"], "--", "Dense GRAPE"),
    "block-sparse": (
        PAPER_COLORS["block_sparse_ode"],
        ":",
        "Block-sparse GRAPE",
    ),
}


def plot_error_cdf(
    fidelities: Mapping[str, np.ndarray],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot empirical CDFs of ``1 - fidelity`` on a logarithmic error axis."""

    apply_paper_style()
    if ax is None:
        created_figure, created_ax = plt.subplots(figsize=(6, 4.5))
        figure = cast(Figure, created_figure)
        ax = cast(Axes, created_ax)
    else:
        figure = cast(Figure, ax.figure)

    styles = {
        "initial": (PAPER_COLORS["initial"], "-", r"$u_{\rm init}$"),
        "grape": (PAPER_COLORS["grape"], "--", r"$u_{\rm gr}$"),
        "dense_grape": (
            PAPER_COLORS["dense_ode"],
            "--",
            "Dense GRAPE",
        ),
        "block_sparse_grape": (
            PAPER_COLORS["block_sparse_ode"],
            ":",
            "Block-sparse GRAPE",
        ),
        "st_grape": (PAPER_COLORS["st_grape"], "-.", r"$u_{\rm st}$"),
        "dense": (PAPER_COLORS["dense_ode"], "--", "Dense GRAPE"),
        "block-sparse": (
            PAPER_COLORS["block_sparse_ode"],
            ":",
            "Block-sparse GRAPE",
        ),
        "ST": (PAPER_COLORS["st_grape"], "-.", r"$u_{\rm st}$"),
    }
    plotted_errors: list[np.ndarray] = []
    for name, values in fidelities.items():
        fidelity = np.asarray(values, dtype=float)
        if np.any((fidelity < 0) | (fidelity > 1 + 1e-10)):
            raise ValueError(f"Fidelities for {name!r} lie outside [0, 1].")
        error = np.clip(1.0 - fidelity, np.finfo(float).tiny, None)
        plotted_errors.append(error)
        sorted_error = np.sort(error)
        probability = np.arange(1, len(error) + 1) / len(error)
        color, linestyle, label = styles.get(name, (None, "-", name))
        ax.plot(sorted_error, probability, color=color, linestyle=linestyle, label=label)
        ax.hist(error, bins=bins, density=True, alpha=0.08, color=color)

    ax.set_xscale("log")
    if plotted_errors:
        all_errors = np.concatenate(plotted_errors)
        minimum_error = float(np.min(all_errors))
        maximum_error = float(np.max(all_errors))
        if (
            minimum_error > np.finfo(float).tiny
            and maximum_error > minimum_error
            and maximum_error / minimum_error < 10.0
        ):
            # Matplotlib labels many minor log ticks when the visible interval
            # is narrower than one decade. Sparse numeric ticks remain legible
            # without changing the logarithmic data transform.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$1-F$")
    ax.set_ylabel("Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend(frameon=False)
    return figure, ax


def plot_fidelity_cdf(
    fidelities: Mapping[str, np.ndarray],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot empirical fidelity CDFs for independently applied noise sources."""

    if not fidelities:
        raise ValueError("At least one fidelity sample is required.")
    apply_paper_style()
    if ax is None:
        created_figure, created_ax = plt.subplots(figsize=(6, 4.5))
        figure = cast(Figure, created_figure)
        ax = cast(Axes, created_ax)
    else:
        figure = cast(Figure, ax.figure)

    plotted_fidelities: list[np.ndarray] = []
    for source, values in fidelities.items():
        fidelity = np.asarray(values, dtype=float)
        if fidelity.ndim != 1 or fidelity.size == 0:
            raise ValueError(f"Fidelities for {source!r} must be a non-empty 1D array.")
        if np.any((fidelity < 0) | (fidelity > 1 + 1e-10)):
            raise ValueError(f"Fidelities for {source!r} lie outside [0, 1].")
        sorted_fidelity = np.sort(np.clip(fidelity, 0.0, 1.0))
        plotted_fidelities.append(sorted_fidelity)
        probability = np.arange(1, sorted_fidelity.size + 1) / sorted_fidelity.size
        color, linestyle, label = _NOISE_SOURCE_STYLES.get(
            source,
            (None, "-", source),
        )
        ax.plot(
            sorted_fidelity,
            probability,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    all_fidelities = np.concatenate(plotted_fidelities)
    lower = float(np.min(all_fidelities))
    upper = float(np.max(all_fidelities))
    padding = max(0.05 * (upper - lower), 1e-6)
    ax.set_xlim(max(0.0, lower - padding), min(1.0, upper + padding))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Fidelity $F$")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Noise-source fidelity distributions")
    ax.legend(frameon=False)
    return figure, ax


def plot_method_fidelity_cdfs_by_noise(
    fidelities: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Figure, np.ndarray]:
    """Compare saved control methods in one CDF panel per noise family.

    The outer mapping is keyed by noise family (normally ``static``,
    ``control``, and ``colored``); each inner mapping associates a method with
    its independently sampled fidelities under only that noise family.
    """

    if not fidelities:
        raise ValueError("At least one noise-source sample is required.")
    apply_paper_style()
    source_order = tuple(
        source for source in ("static", "control", "colored") if source in fidelities
    ) + tuple(source for source in fidelities if source not in {"static", "control", "colored"})
    created_figure, created_axes = plt.subplots(
        1,
        len(source_order),
        figsize=(5.0 * len(source_order), 4.2),
        sharey=True,
        squeeze=False,
    )
    figure = cast(Figure, created_figure)
    axes = np.asarray(created_axes[0], dtype=object)

    for index, source in enumerate(source_order):
        axis = cast(Axes, axes[index])
        method_samples = fidelities[source]
        if not method_samples:
            raise ValueError(f"No method samples were supplied for {source!r}.")
        plotted: list[np.ndarray] = []
        for method, values in method_samples.items():
            fidelity = np.asarray(values, dtype=float)
            if fidelity.ndim != 1 or fidelity.size == 0:
                raise ValueError(
                    f"Fidelities for {source!r}/{method!r} must be a non-empty 1D array."
                )
            if np.any((fidelity < 0) | (fidelity > 1 + 1e-10)):
                raise ValueError(f"Fidelities for {source!r}/{method!r} lie outside [0, 1].")
            sorted_fidelity = np.sort(np.clip(fidelity, 0.0, 1.0))
            plotted.append(sorted_fidelity)
            probability = np.arange(1, sorted_fidelity.size + 1) / sorted_fidelity.size
            color, linestyle, label = _METHOD_STYLES.get(
                method,
                (None, "-", method),
            )
            axis.plot(
                sorted_fidelity,
                probability,
                color=color,
                linestyle=linestyle,
                label=label,
            )

        all_fidelities = np.concatenate(plotted)
        lower = float(np.min(all_fidelities))
        upper = float(np.max(all_fidelities))
        padding = max(0.05 * (upper - lower), 1e-6)
        axis.set_xlim(max(0.0, lower - padding), min(1.0, upper + padding))
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"Fidelity $F$")
        axis.set_title(_NOISE_SOURCE_STYLES.get(source, (None, None, source))[2])
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.legend(frameon=False)

    cast(Axes, axes[0]).set_ylabel("Cumulative probability")
    figure.tight_layout()
    return figure, axes


def plot_method_error_cdfs_by_noise(
    fidelities: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[Figure, np.ndarray]:
    """Compare methods using an infidelity CDF panel for each noise family."""

    if not fidelities:
        raise ValueError("At least one noise-source sample is required.")
    apply_paper_style()
    source_order = tuple(
        source for source in ("static", "control", "colored") if source in fidelities
    ) + tuple(source for source in fidelities if source not in {"static", "control", "colored"})
    created_figure, created_axes = plt.subplots(
        1,
        len(source_order),
        figsize=(5.0 * len(source_order), 4.2),
        sharey=True,
        squeeze=False,
    )
    figure = cast(Figure, created_figure)
    axes = np.asarray(created_axes[0], dtype=object)

    for index, source in enumerate(source_order):
        axis = cast(Axes, axes[index])
        method_samples = fidelities[source]
        if not method_samples:
            raise ValueError(f"No method samples were supplied for {source!r}.")
        for method, values in method_samples.items():
            fidelity = np.asarray(values, dtype=float)
            if fidelity.ndim != 1 or fidelity.size == 0:
                raise ValueError(
                    f"Fidelities for {source!r}/{method!r} must be a non-empty 1D array."
                )
            if np.any((fidelity < 0) | (fidelity > 1 + 1e-10)):
                raise ValueError(f"Fidelities for {source!r}/{method!r} lie outside [0, 1].")
            errors = np.sort(np.clip(1.0 - fidelity, np.finfo(float).tiny, None))
            probability = np.arange(1, errors.size + 1) / errors.size
            color, linestyle, label = _METHOD_STYLES.get(
                method,
                (None, "-", method),
            )
            axis.plot(
                errors,
                probability,
                color=color,
                linestyle=linestyle,
                label=label,
            )

        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"Error $1-F$")
        axis.set_title(_NOISE_SOURCE_STYLES.get(source, (None, None, source))[2])
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.legend(frameon=False)

    cast(Axes, axes[0]).set_ylabel("Cumulative probability")
    figure.tight_layout()
    return figure, axes
