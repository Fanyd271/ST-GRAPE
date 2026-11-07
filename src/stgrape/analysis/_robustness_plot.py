"""Plotting implementation transcribed from the final paper notebook.

Run through stgrape.analysis; all outputs are directed outside the data archive.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

TASK = globals().get("TASK", "ghz")
REPOSITORY_ROOT = Path(globals()["PROJECT_ROOT"])
RESULT_ROOT = (
    Path(globals()["DATA_ROOT"]) / "paper" / (TASK if TASK in ("ghz", "qft") else "stress")
)
FIGURE_DIRECTORY = Path(globals()["OUTPUT_DIR"])

EXPECTED_SEED_COUNT = 10
MONTE_CARLO_SAMPLES = 10_000
SAVE_FIGURES = True
N_QUBITS_OVERRIDE: int | None = None  # None: infer from the selected result directory.
ROBUSTNESS_ORDERS = (
    1,
    2,
)
METHODS = ("dense", "block-sparse", "ST")
METHOD_DRAW_ORDER = ("ST", "dense", "block-sparse")
METHOD_ZORDERS = {"ST": 3.0, "dense": 4.0, "block-sparse": 5.0}
METHOD_LINEWIDTH_FACTORS = {"ST": 0.86, "dense": 1.08, "block-sparse": 1.08}
METHOD_LABELS = {
    "dense": "Dense GRAPE",
    "block-sparse": "Block-sparse GRAPE",
    "ST": "ST-GRAPE",
}
COLORS = {
    "Initial": "#777777",
    "dense": "#D55E00",
    "block-sparse": "#009E73",
    "ST": "#0072B2",
}
ENDPOINT_MARKERS = {"dense": "o", "block-sparse": "s", "ST": "^"}
ENDPOINT_MARKER_SIZE = 2.7
ENDPOINT_MARKER_EDGEWIDTH = 0.65
ENDPOINT_MARKER_ALPHA = 0.78
ORDER_LINESTYLES = {1: "--", 2: "-"}
RANGE_ALPHA = 0.17
MEAN_LINEWIDTH = 1.08
CONVERGENCE_MEAN_LINEWIDTH = 0.88
SHOW_CONVERGENCE_MEAN = False
CONVERGENCE_RANGE_ALPHA = 0.13
SEED_LINEWIDTH = 0.56
SEED_ALPHA = 0.88
BOUND_LINEWIDTH = 0.68
BOUND_ALPHA = 0.76
LOG_Y_PADDING_DECADES = 0.16
INFIDELITY_FLOOR = 1e-14
CDF_DISPLAY_QUANTILES = (0.005, 0.995)
CDF_LOG_PADDING_DECADES = 0.24

if SAVE_FIGURES:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def infer_n_qubits(result_root: Path, override: int | None) -> int:
    """Resolve the qubit count and reject a stale manual override."""

    available = sorted(
        int(path.name[:-1])
        for path in result_root.iterdir()
        if path.is_dir() and path.name.endswith("q") and path.name[:-1].isdigit()
    )
    if override is not None:
        if override not in available:
            raise FileNotFoundError(
                f"Requested N_QUBITS_OVERRIDE={override}, but {result_root} contains "
                f"qubit directories {available}."
            )
        return override
    if len(available) != 1:
        raise ValueError(
            f"Cannot infer one qubit count from {result_root}; found {available}. "
            "Set N_QUBITS_OVERRIDE explicitly."
        )
    return available[0]


MANIFEST = json.loads((RESULT_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
CONFIGURATION = MANIFEST["configuration"]
N_QUBITS = infer_n_qubits(RESULT_ROOT, N_QUBITS_OVERRIDE)
TIME_STEP_US = float(CONFIGURATION["time_step_us"])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.linewidth": 0.75,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)

print(f"Repository root: {REPOSITORY_ROOT}")
print(f"Result root:     {RESULT_ROOT}")
print(f"Figure output:  {FIGURE_DIRECTORY}")


def method_directory(order: int, seed: int, method: str) -> Path:
    return RESULT_ROOT / f"{N_QUBITS}q" / f"r{order}" / f"seed{seed}" / method


def discover_complete_paired_seeds() -> tuple[int, ...]:
    """Return seeds complete for every method and both robustness orders."""

    seed_sets: list[set[int]] = []
    for order in ROBUSTNESS_ORDERS:
        order_root = RESULT_ROOT / f"{N_QUBITS}q" / f"r{order}"
        for method in METHODS:
            available = {
                int(path.name.removeprefix("seed"))
                for path in order_root.glob("seed*")
                if (path / method / "result.npz").is_file()
                and (path / method / "fidelities.npz").is_file()
                and (path / method / "method_summary.json").is_file()
            }
            seed_sets.append(available)
        seed_sets.append(
            {
                int(path.name.removeprefix("seed"))
                for path in order_root.glob("seed*")
                if (path / "initial_pulse.npz").is_file()
                and (path / "initial_fidelities.npz").is_file()
            }
        )
    shared = sorted(set.intersection(*seed_sets)) if seed_sets else []
    return tuple(shared[:EXPECTED_SEED_COUNT])


SEEDS = discover_complete_paired_seeds()
if not SEEDS:
    raise FileNotFoundError("No complete paired seeds were found.")
if len(SEEDS) < EXPECTED_SEED_COUNT:
    warnings.warn(
        f"Preview uses {len(SEEDS)} paired seeds {SEEDS}; "
        f"the final figure expects {EXPECTED_SEED_COUNT}.",
        stacklevel=2,
    )

# The same optimization seed must carry the same initial pulse and validation
# trajectories for r=1 and r=2.
for seed in SEEDS:
    first_case = RESULT_ROOT / f"{N_QUBITS}q" / "r1" / f"seed{seed}"
    second_case = RESULT_ROOT / f"{N_QUBITS}q" / "r2" / f"seed{seed}"
    with (
        np.load(first_case / "initial_pulse.npz") as first,
        np.load(second_case / "initial_pulse.npz") as second,
    ):
        np.testing.assert_allclose(first["coefficients"], second["coefficients"])
        np.testing.assert_allclose(
            first["control_amplitudes"],
            second["control_amplitudes"],
        )
    with (
        np.load(first_case / "initial_fidelities.npz") as first,
        np.load(second_case / "initial_fidelities.npz") as second,
    ):
        np.testing.assert_array_equal(first["values"], second["values"])

sampling_seeds: set[int] = set()
declared_sample_counts: set[int] = set()
for order in ROBUSTNESS_ORDERS:
    for seed in SEEDS:
        for method in METHODS:
            metadata = json.loads(
                (method_directory(order, seed, method) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            sampling_seeds.add(int(metadata["sampling_seed"]))
            declared_sample_counts.add(int(metadata["monte_carlo_samples"]))
if len(sampling_seeds) != 1:
    raise ValueError(f"Methods do not share one Monte Carlo sampling seed: {sampling_seeds}")
if len(declared_sample_counts) != 1:
    raise ValueError(f"Methods do not share one Monte Carlo sample count: {declared_sample_counts}")
if next(iter(declared_sample_counts)) < MONTE_CARLO_SAMPLES:
    raise ValueError(
        f"Saved runs declare only {next(iter(declared_sample_counts))} Monte Carlo "
        f"samples, but {MONTE_CARLO_SAMPLES} were requested."
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def load_summary(order: int, seed: int, method: str) -> dict[str, object]:
    return json.loads(
        (method_directory(order, seed, method) / "method_summary.json").read_text(encoding="utf-8")
    )


def true_loss_trajectory(
    order: int,
    seed: int,
    method: str,
    *,
    coordinate: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return historical-best exact loss on its actual evaluation domain."""

    saved = load_npz(method_directory(order, seed, method) / "result.npz")
    if method == "ST":
        losses = np.asarray(saved["exact_objective_history"], dtype=float)
        if coordinate == "iteration":
            x = np.asarray(saved["exact_evaluation_iterations"], dtype=float)
        elif coordinate == "time":
            x = np.asarray(saved["exact_evaluation_times"], dtype=float)
        else:
            raise ValueError("coordinate must be 'iteration' or 'time'.")
    else:
        history = np.asarray(saved["objective_history"], dtype=float)
        losses = np.r_[float(saved["initial_objective"]), history]
        if coordinate == "iteration":
            x = np.arange(losses.size, dtype=float)
        elif coordinate == "time":
            x = np.r_[0.0, np.asarray(saved["elapsed_time_history"], dtype=float)]
        else:
            raise ValueError("coordinate must be 'iteration' or 'time'.")

    if x.size != losses.size or x.size == 0:
        raise ValueError(f"Inconsistent loss history for r={order}, seed={seed}, method={method}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(losses)):
        raise ValueError("Loss histories must be finite.")
    if np.any(losses <= 0.0):
        raise ValueError("True-loss histories must remain positive for logarithmic plots.")

    order_index = np.argsort(x, kind="stable")
    x = x[order_index]
    losses = losses[order_index]
    return x, np.minimum.accumulate(losses)


def right_continuous_step(
    x: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    hold_terminal: bool = False,
) -> np.ndarray:
    """Evaluate right-continuous best-so-far values on a common grid."""

    indices = np.searchsorted(x, grid, side="right") - 1
    aligned = np.full(grid.shape, np.nan, dtype=float)
    valid = indices >= 0
    if not hold_terminal:
        valid &= grid <= x[-1]
    clipped_indices = np.clip(indices[valid], 0, values.size - 1)
    aligned[valid] = values[clipped_indices]
    return aligned


def common_coordinate_grid(
    trajectories: tuple[tuple[np.ndarray, np.ndarray], ...],
    coordinate: str,
) -> np.ndarray:
    """Return every observed event plus explicit seed-termination boundaries."""

    start = max(float(x[0]) for x, _ in trajectories)
    stop = max(float(x[-1]) for x, _ in trajectories)
    if stop <= start:
        raise ValueError("Paired loss histories do not span a positive domain.")
    # Evaluate bounds at every actual event. Subsampling the iteration axis
    # can miss a seed's decrease and let its trace escape the envelope.
    events = [x[(x >= start) & (x <= stop)] for x, _ in trajectories]
    # Make an early-terminated seed leave the active set immediately instead
    # of extending it to the next evaluation belonging to another seed.
    terminal_boundaries = [
        np.nextafter(float(x[-1]), np.inf) for x, _ in trajectories if float(x[-1]) < stop
    ]
    if terminal_boundaries:
        events.append(np.asarray(terminal_boundaries, dtype=float))
    return np.unique(np.concatenate(events))


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_convergence_panel(
    axis: plt.Axes,
    *,
    order: int,
    coordinate: str,
    label: str,
) -> np.ndarray:
    """Plot actual seed traces plus paired mean and observed min闂傚倸鍊搁崐鎼佸磹閹间礁纾归柣鎴ｅГ閸ゅ嫰鏌涢锝嗙缁炬儳顭烽弻鏇熺箾閻愵剚鐝旂紒鐐劤濞硷繝寮婚敍鍕勃閻犲洦褰冩慨搴ｇ磽娴ｅ搫校闁绘绮撻崺鈧い鎺嗗亾婵犫偓闁秮鈧箓宕奸妷銉ь槶?range."""

    trajectories = {
        (method, seed): true_loss_trajectory(
            order,
            seed,
            method,
            coordinate=coordinate,
        )
        for method in METHODS
        for seed in SEEDS
    }
    x_max = max(float(x[-1]) for x, _ in trajectories.values())
    plotted_values: list[np.ndarray] = []

    for method in METHOD_DRAW_ORDER:
        method_trajectories = tuple(trajectories[(method, seed)] for seed in SEEDS)
        for seed_index, (x, values) in enumerate(method_trajectories):
            # Alternate the dense/block-sparse foreground by paired seed so
            # neither baseline systematically masks the other.
            if method == "ST":
                seed_zorder = 5.0
            else:
                foreground_method = "dense" if seed_index % 2 == 0 else "block-sparse"
                seed_zorder = 7.0 if method == foreground_method else 6.0
            axis.plot(
                x,
                values,
                color=COLORS[method],
                linestyle="--",
                linewidth=SEED_LINEWIDTH,
                alpha=SEED_ALPHA,
                drawstyle="steps-post",
                zorder=seed_zorder,
            )
            axis.plot(
                x[-1],
                values[-1],
                marker=ENDPOINT_MARKERS[method],
                markersize=ENDPOINT_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=COLORS[method],
                markeredgewidth=ENDPOINT_MARKER_EDGEWIDTH,
                alpha=ENDPOINT_MARKER_ALPHA,
                linestyle="none",
                zorder=seed_zorder + 0.5,
                clip_on=False,
            )

        # At each coordinate, aggregate only seeds whose actual histories still
        # exist. A terminated seed is excluded rather than extended.
        grid = common_coordinate_grid(method_trajectories, coordinate)
        aligned = np.vstack(
            [right_continuous_step(x, values, grid) for x, values in method_trajectories]
        )
        active_columns = np.any(np.isfinite(aligned), axis=0)
        grid = grid[active_columns]
        aligned = aligned[:, active_columns]
        plotted_values.append(aligned[np.isfinite(aligned)])
        lower_bound = np.nanmin(aligned, axis=0)
        upper_bound = np.nanmax(aligned, axis=0)
        axis.fill_between(
            grid,
            lower_bound,
            upper_bound,
            color=COLORS[method],
            alpha=CONVERGENCE_RANGE_ALPHA,
            linewidth=0.0,
            step="post",
            zorder=METHOD_ZORDERS[method] - 0.8,
        )
        if SHOW_CONVERGENCE_MEAN:
            axis.plot(
                grid,
                np.nanmean(aligned, axis=0),
                color=COLORS[method],
                linestyle="-",
                linewidth=(CONVERGENCE_MEAN_LINEWIDTH * METHOD_LINEWIDTH_FACTORS[method]),
                zorder=METHOD_ZORDERS[method],
            )

    axis.set_yscale("log")
    if coordinate == "time":
        positive_coordinates = np.concatenate([x[x > 0.0] for x, _ in trajectories.values()])
        if positive_coordinates.size == 0:
            raise ValueError("Wall-clock histories contain no positive times.")
        axis.set_xscale("log")
        axis.set_xlim(float(np.min(positive_coordinates)), x_max)
    else:
        axis.set_xlim(0.0, x_max)
    axis.grid(True, which="major", axis="both", alpha=0.18, linewidth=0.45)
    axis.grid(False, which="minor", axis="y")
    axis.set_title(rf"$r={order}$")
    axis.set_xlabel("Iteration" if coordinate == "iteration" else "Wall-clock time (s)")
    panel_label(axis, label)
    return np.concatenate([values.ravel() for values in plotted_values])


def apply_loss_limits(
    axes: tuple[plt.Axes, ...],
    plotted_values: list[np.ndarray],
) -> None:
    """Apply one order-specific logarithmic range, independent of the other order."""

    positive = np.concatenate(plotted_values)
    log_min = float(np.log10(np.min(positive))) - LOG_Y_PADDING_DECADES
    log_max = float(np.log10(np.max(positive))) + LOG_Y_PADDING_DECADES
    for axis in axes:
        axis.set_ylim(10.0**log_min, 10.0**log_max)


def order_legend_handles() -> list[Line2D]:
    """Return the line-style key displayed inside the CDF panel."""

    return [
        Line2D([0], [0], color="black", linestyle="--", label=r"$r=1$"),
        Line2D([0], [0], color="black", linestyle="-", label=r"$r=2$"),
    ]


def legend_handles(*, include_initial: bool, include_orders: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    if include_initial:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["Initial"],
                linestyle=":",
                linewidth=MEAN_LINEWIDTH,
                label="Initial",
            )
        )
    handles.extend(
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linewidth=MEAN_LINEWIDTH,
            marker=ENDPOINT_MARKERS[method],
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=ENDPOINT_MARKER_EDGEWIDTH,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    )
    if include_orders:
        handles.extend(order_legend_handles())
    return handles


def save_figure(figure: plt.Figure, stem: str) -> None:
    if not SAVE_FIGURES:
        return
    figure.savefig(
        FIGURE_DIRECTORY / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    figure.savefig(
        FIGURE_DIRECTORY / f"{stem}.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def write_seed_summary_csv() -> None:
    """Write the plotted seeds' terminal loss and timing metadata."""

    if not SAVE_FIGURES:
        return
    summary_path = FIGURE_DIRECTORY / "figure_seed_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("seed", "order", "method", "true_loss", "wall_time_s"),
        )
        writer.writeheader()
        for order in ROBUSTNESS_ORDERS:
            for seed in SEEDS:
                for method in METHODS:
                    summary = load_summary(order, seed, method)
                    writer.writerow(
                        {
                            "seed": seed,
                            "order": order,
                            "method": method,
                            "true_loss": float(summary["true_loss"]),
                            "wall_time_s": float(summary["wall_time_s"]),
                        }
                    )


write_seed_summary_csv()

print(
    f"Using paired seeds {SEEDS} ({len(SEEDS)}/{EXPECTED_SEED_COUNT}); "
    f"Monte Carlo sampling seed={next(iter(sampling_seeds))}; "
    f"{MONTE_CARLO_SAMPLES} trajectories per seed will be plotted."
)

# (a), (b), (d), and (e): standalone convergence preview.
convergence_figure, convergence_axes = plt.subplots(
    2,
    2,
    figsize=(10.4, 6.0),
    sharex="col",
)
loss_a = plot_convergence_panel(
    convergence_axes[0, 0],
    order=1,
    coordinate="iteration",
    label="(a)",
)
loss_b = plot_convergence_panel(
    convergence_axes[0, 1],
    order=2,
    coordinate="iteration",
    label="(b)",
)
loss_d = plot_convergence_panel(
    convergence_axes[1, 0],
    order=1,
    coordinate="time",
    label="(d)",
)
loss_e = plot_convergence_panel(
    convergence_axes[1, 1],
    order=2,
    coordinate="time",
    label="(e)",
)

# Iteration and wall-clock panels share a range only within the same order.
apply_loss_limits((convergence_axes[0, 0], convergence_axes[1, 0]), [loss_a, loss_d])
apply_loss_limits((convergence_axes[0, 1], convergence_axes[1, 1]), [loss_b, loss_e])
for axis in convergence_axes.ravel():
    axis.set_ylabel("Historical-best true loss")
convergence_figure.legend(
    handles=legend_handles(include_initial=False, include_orders=False),
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.995),
)
convergence_figure.subplots_adjust(
    left=0.095,
    right=0.99,
    bottom=0.09,
    top=0.91,
    wspace=0.25,
    hspace=0.29,
)
save_figure(convergence_figure, "exp01_convergence_panels")
plt.close("all")


def load_seed_fidelities(
    *,
    seed: int,
    method: str | None = None,
    order: int = 1,
) -> np.ndarray:
    if method is None:
        path = RESULT_ROOT / f"{N_QUBITS}q" / f"r{order}" / f"seed{seed}" / "initial_fidelities.npz"
    else:
        path = method_directory(order, seed, method) / "fidelities.npz"
    values = np.asarray(load_npz(path)["values"], dtype=float)
    if values.size < MONTE_CARLO_SAMPLES:
        raise ValueError(
            f"{path} contains {values.size} samples; {MONTE_CARLO_SAMPLES} are required."
        )
    return values[:MONTE_CARLO_SAMPLES]


def empirical_cdf(sorted_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, grid, side="right") / sorted_values.size


def cdf_groups() -> list[tuple[str, str | None, int | None]]:
    # Draw ST first so the dense and block-sparse central curves remain visible.
    return [
        ("Initial", None, None),
        *(
            (f"{method}-r{order}", method, order)
            for method in METHOD_DRAW_ORDER
            for order in (1, 2)
        ),
    ]


def cdf_display_limits(errors: dict[tuple[str, int], np.ndarray]) -> tuple[float, float]:
    """Choose robust log limits without allowing extreme tails to waste the panel."""

    lower_quantile, upper_quantile = CDF_DISPLAY_QUANTILES
    lower_values = [float(np.quantile(values, lower_quantile)) for values in errors.values()]
    upper_values = [float(np.quantile(values, upper_quantile)) for values in errors.values()]
    lower = max(min(lower_values), INFIDELITY_FLOOR)
    upper = max(upper_values)
    if upper <= lower:
        upper = max(float(values[-1]) for values in errors.values())
    log_lower = np.log10(lower) - CDF_LOG_PADDING_DECADES
    log_upper = np.log10(upper) + CDF_LOG_PADDING_DECADES
    return 10.0**log_lower, 10.0**log_upper


def plot_infidelity_cdf(axis: plt.Axes, *, label: str) -> None:
    """Plot seed-level ECDFs and their pointwise mean and observed range."""

    errors: dict[tuple[str, int], np.ndarray] = {}
    for group, method, order in cdf_groups():
        for seed in SEEDS:
            fidelities = load_seed_fidelities(
                seed=seed,
                method=method,
                order=1 if order is None else order,
            )
            infidelity = np.maximum(1.0 - fidelities, INFIDELITY_FLOOR)
            errors[(group, seed)] = np.sort(infidelity)

    display_lower, display_upper = cdf_display_limits(errors)
    grid = np.geomspace(display_lower, display_upper, 900)

    for group, method, order in cdf_groups():
        mean_color = COLORS["Initial"] if method is None else COLORS[method]
        linestyle = ":" if order is None else ORDER_LINESTYLES[order]
        seed_cdfs = np.vstack([empirical_cdf(errors[(group, seed)], grid) for seed in SEEDS])
        lower_bound = np.min(seed_cdfs, axis=0)
        upper_bound = np.max(seed_cdfs, axis=0)
        axis.fill_between(
            grid,
            lower_bound,
            upper_bound,
            color=mean_color,
            alpha=RANGE_ALPHA,
            linewidth=0.0,
            zorder=0,
        )
        for bound in (lower_bound, upper_bound):
            axis.plot(
                grid,
                bound,
                color=mean_color,
                linestyle=":",
                linewidth=BOUND_LINEWIDTH,
                alpha=BOUND_ALPHA,
                zorder=2,
            )
        axis.plot(
            grid,
            np.mean(seed_cdfs, axis=0),
            color=mean_color,
            linestyle=linestyle,
            linewidth=MEAN_LINEWIDTH,
            zorder=3,
        )

    axis.set_xscale("log")
    axis.set_xlim(display_lower, display_upper)
    axis.set_ylim(0.0, 1.005)
    axis.set_xlabel(r"Infidelity $\varepsilon_F=1-F$")
    axis.set_ylabel(r"Infidelity CDF $\mathcal{C}(\varepsilon)$")
    axis.grid(True, which="major", axis="x", alpha=0.18, linewidth=0.45)
    axis.grid(True, which="major", axis="y", alpha=0.18, linewidth=0.45)
    axis.grid(False, which="minor", axis="y")
    panel_label(axis, label)


# (c): standalone CDF preview.
cdf_figure, cdf_axis = plt.subplots(figsize=(5.2, 3.7))
plot_infidelity_cdf(cdf_axis, label="(c)")
cdf_axis.legend(
    handles=order_legend_handles(),
    loc="upper left",
    ncol=1,
    frameon=False,
    handlelength=2.3,
    borderaxespad=0.55,
)
cdf_figure.legend(
    handles=legend_handles(include_initial=True, include_orders=False),
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
cdf_figure.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.83)
save_figure(cdf_figure, "exp01_infidelity_cdf")
plt.close("all")

print(f"CDF uses {MONTE_CARLO_SAMPLES} Monte Carlo samples for every seed and solution.")


def representative_seed() -> tuple[int, float]:
    """Choose the r=2 ST seed whose exact final loss is closest to the median."""

    losses = {seed: float(load_summary(2, seed, "ST")["true_loss"]) for seed in SEEDS}
    median_loss = float(np.median(list(losses.values())))
    selected = min(SEEDS, key=lambda seed: (abs(losses[seed] - median_loss), seed))
    return selected, median_loss


def load_waveforms(seed: int) -> list[tuple[str, np.ndarray]]:
    """Load only the Initial and final r=2 ST-GRAPE controls."""

    initial = np.asarray(
        load_npz(RESULT_ROOT / f"{N_QUBITS}q" / "r1" / f"seed{seed}" / "initial_pulse.npz")[
            "control_amplitudes"
        ],
        dtype=float,
    )
    r2 = np.asarray(
        load_npz(method_directory(2, seed, "ST") / "result.npz")["control_amplitudes"],
        dtype=float,
    )
    if initial.shape != r2.shape:
        raise ValueError("Representative waveforms do not share one control grid.")
    if initial.shape[0] != 2 * N_QUBITS:
        raise ValueError(
            f"Control channels must follow (X_1,...,X_n,Y_1,...,Y_n); received {initial.shape}."
        )
    return [
        ("Initial", initial),
        (r"$r=2$ ST-GRAPE", r2),
    ]


WAVEFORM_COLORS = {
    "Initial": "#4D4D4D",
    r"$r=2$ ST-GRAPE": "#0072B2",
}
QUADRATURE_LINESTYLES = {"X": "-", "Y": "--"}


def smooth_control_for_display(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a shape-preserving visual interpolation of saved PWC samples.

    The interpolation changes only the rendered curve. Optimization data and
    all numerical metrics continue to use the saved PWC amplitudes.
    """

    values = np.asarray(values, dtype=float)
    n_steps = values.size
    duration_ns = n_steps * TIME_STEP_US * 1e3
    centers_ns = (np.arange(n_steps, dtype=float) + 0.5) * TIME_STEP_US * 1e3
    knots_ns = np.r_[0.0, centers_ns, duration_ns]
    knot_values = np.r_[values[0], values, values[-1]]
    dense_times_ns = np.linspace(0.0, duration_ns, max(800, 8 * n_steps))
    smooth_values = PchipInterpolator(knots_ns, knot_values)(dense_times_ns)
    return dense_times_ns, np.asarray(smooth_values, dtype=float)


def waveform_legend_handles() -> list[Line2D]:
    """Describe only solution identity and X/Y quadratures."""

    return [
        Line2D(
            [0],
            [0],
            color=WAVEFORM_COLORS["Initial"],
            linewidth=1.1,
            label="Initial",
        ),
        Line2D(
            [0],
            [0],
            color=WAVEFORM_COLORS[r"$r=2$ ST-GRAPE"],
            linewidth=1.1,
            label=r"$r=2$ ST-GRAPE",
        ),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label="X"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="Y"),
    ]


def plot_qubit_waveform_strips(
    axes: list[plt.Axes] | np.ndarray,
    *,
    seed: int,
    label: str,
    compact: bool = False,
    legend_axis: plt.Axes | None = None,
) -> None:
    """Plot smooth Initial/r=2 X/Y controls in one strip per physical qubit."""

    axes = list(axes)
    if len(axes) != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS} waveform axes; received {len(axes)}.")
    waveforms = load_waveforms(seed)
    controls_mhz = {stage: controls / (2.0 * np.pi) for stage, controls in waveforms}
    amplitude_limit = 1.04 * max(float(np.max(np.abs(values))) for values in controls_mhz.values())
    n_steps = next(iter(controls_mhz.values())).shape[-1]
    duration_ns = n_steps * TIME_STEP_US * 1e3

    for qubit, axis in enumerate(axes):
        for stage, _ in waveforms:
            values = controls_mhz[stage]
            for quadrature, channel in (("X", qubit), ("Y", qubit + N_QUBITS)):
                times_ns, amplitudes = smooth_control_for_display(values[channel])
                axis.plot(
                    times_ns,
                    amplitudes,
                    color=WAVEFORM_COLORS[stage],
                    linestyle=QUADRATURE_LINESTYLES[quadrature],
                    linewidth=0.62 if compact else 0.92,
                    alpha=1.0 if stage == "Initial" else 0.92,
                )
        axis.set_xlim(0.0, duration_ns)
        axis.set_ylim(-amplitude_limit, amplitude_limit)
        axis.set_ylabel(
            f"Q{qubit + 1}\nMHz",
            rotation=0,
            ha="right",
            va="center",
            fontsize=5.8 if compact else 7.4,
            labelpad=3.0 if compact else 5.0,
        )
        axis.grid(True, alpha=0.14, linewidth=0.4)
        axis.set_yticks((-50.0, 0.0, 50.0))
        if qubit < N_QUBITS - 1:
            axis.tick_params(labelbottom=False)

    if legend_axis is None:
        legend_target = axes[0]
        legend_location = "upper center"
        legend_anchor = (0.54, 1.34)
    else:
        legend_target = legend_axis
        legend_target.set_axis_off()
        legend_location = "center"
        legend_anchor = (0.56, 0.48)
    legend_target.legend(
        handles=waveform_legend_handles(),
        loc=legend_location,
        bbox_to_anchor=legend_anchor,
        ncol=4,
        frameon=False,
        columnspacing=0.62 if compact else 1.15,
        handlelength=1.35 if compact else 1.95,
        handletextpad=0.38,
        labelspacing=0.10,
        fontsize=5.55 if compact else 7.0,
    )
    panel_label(legend_target, label)
    axes[-1].set_xlabel("Time (ns)")


REPRESENTATIVE_SEED, MEDIAN_R2_TRUE_LOSS = representative_seed()
print(
    f"Representative seed={REPRESENTATIVE_SEED}; median r=2 ST true loss={MEDIAN_R2_TRUE_LOSS:.6g}."
)

# (f): standalone control preview with one row per qubit and a dedicated legend row.
waveform_figure = plt.figure(figsize=(7.2, 5.35))
waveform_grid = waveform_figure.add_gridspec(
    N_QUBITS + 1,
    1,
    height_ratios=(0.42, *(1.0 for _ in range(N_QUBITS))),
    left=0.09,
    right=0.995,
    bottom=0.10,
    top=0.98,
    hspace=0.12,
)
waveform_legend_axis = waveform_figure.add_subplot(waveform_grid[0, 0])
waveform_axes = []
for qubit in range(N_QUBITS):
    shared_axis = waveform_axes[0] if waveform_axes else None
    waveform_axes.append(
        waveform_figure.add_subplot(
            waveform_grid[qubit + 1, 0],
            sharex=shared_axis,
            sharey=shared_axis,
        )
    )
plot_qubit_waveform_strips(
    waveform_axes,
    seed=REPRESENTATIVE_SEED,
    label="(f)",
    legend_axis=waveform_legend_axis,
)
save_figure(waveform_figure, "exp01_representative_controls")
plt.close("all")

# Final paper-format 2 x 3 composite figure.
paper_figure = plt.figure(figsize=(8.35, 5.22))
outer_grid = paper_figure.add_gridspec(
    2,
    3,
    width_ratios=(1.0, 1.0, 1.22),
    left=0.075,
    right=0.995,
    bottom=0.09,
    top=0.885,
    wspace=0.31,
    hspace=0.37,
)

axis_a = paper_figure.add_subplot(outer_grid[0, 0])
axis_b = paper_figure.add_subplot(outer_grid[0, 1])
axis_c = paper_figure.add_subplot(outer_grid[0, 2])
axis_d = paper_figure.add_subplot(outer_grid[1, 0], sharey=axis_a)
axis_e = paper_figure.add_subplot(outer_grid[1, 1], sharey=axis_b)

paper_loss_a = plot_convergence_panel(
    axis_a,
    order=1,
    coordinate="iteration",
    label="(a)",
)
paper_loss_b = plot_convergence_panel(
    axis_b,
    order=2,
    coordinate="iteration",
    label="(b)",
)
paper_loss_d = plot_convergence_panel(
    axis_d,
    order=1,
    coordinate="time",
    label="(d)",
)
paper_loss_e = plot_convergence_panel(
    axis_e,
    order=2,
    coordinate="time",
    label="(e)",
)
apply_loss_limits((axis_a, axis_d), [paper_loss_a, paper_loss_d])
apply_loss_limits((axis_b, axis_e), [paper_loss_b, paper_loss_e])
for axis in (axis_a, axis_b, axis_d, axis_e):
    axis.set_ylabel("Loss")

plot_infidelity_cdf(axis_c, label="(c)")
axis_c.legend(
    handles=order_legend_handles(),
    loc="upper left",
    ncol=1,
    frameon=False,
    handlelength=2.3,
    borderaxespad=0.55,
)

waveform_grid = outer_grid[1, 2].subgridspec(
    N_QUBITS + 1,
    1,
    height_ratios=(0.44, *(1.0 for _ in range(N_QUBITS))),
    hspace=0.08,
)
waveform_legend_axis = paper_figure.add_subplot(waveform_grid[0, 0])
waveform_axes = [paper_figure.add_subplot(waveform_grid[index + 1, 0]) for index in range(N_QUBITS)]
plot_qubit_waveform_strips(
    waveform_axes,
    seed=REPRESENTATIVE_SEED,
    label="(f)",
    compact=True,
    legend_axis=waveform_legend_axis,
)

paper_figure.legend(
    handles=legend_handles(include_initial=True, include_orders=False),
    loc="upper center",
    ncol=4,
    frameon=False,
    columnspacing=1.25,
    handlelength=2.6,
    bbox_to_anchor=(0.535, 0.988),
)

save_figure(paper_figure, "exp01_six_panel")
plt.close("all")

print(f"Saved final paper figure to {FIGURE_DIRECTORY / 'exp01_six_panel.pdf'}")
