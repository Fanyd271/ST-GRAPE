"""Plotting implementation transcribed from the final paper notebook.

Run through stgrape.analysis; all outputs are directed outside the data archive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

TASK = globals().get("TASK", "ghz")
REPOSITORY_ROOT = Path(globals()["PROJECT_ROOT"])
RESULT_ROOT = (
    Path(globals()["DATA_ROOT"]) / "paper" / (TASK if TASK in ("ghz", "qft") else "stress")
)
FIGURE_DIRECTORY = Path(globals()["OUTPUT_DIR"])

MONTE_CARLO_RUN = "samples-500_seed-42_dbe325b94dbf"
MONTE_CARLO_ROOT = RESULT_ROOT / "monte_carlo" / MONTE_CARLO_RUN
MONTE_CARLO_SAMPLES = 500
MONTE_CARLO_SAMPLING_SEED = 42
N_QUBITS_OVERRIDE: int | None = None
SEEDS_OVERRIDE: tuple[int, ...] | None = (0, 1, 2, 3, 4)
SAVE_FIGURES = True
SHOW_CONVERGENCE_MEAN = False
METHOD_DRAW_ORDER = ("ST", "dense", "block-sparse")
ENDPOINT_MARKERS = {"dense": "o", "block-sparse": "s", "ST": "^"}
ENDPOINT_MARKER_SIZE = 2.7
ENDPOINT_MARKER_ALPHA = 0.78
ENDPOINT_MARKER_EDGEWIDTH = 0.65
CONVERGENCE_RANGE_ALPHA = 0.13
SEED_LINEWIDTH = 0.56
SEED_ALPHA = 0.88
MEAN_LINEWIDTH = 1.08
BOUND_LINEWIDTH = 0.68
BOUND_ALPHA = 0.76
LOG_Y_PADDING_DECADES = 0.16
INFIDELITY_FLOOR = 1e-14
CDF_DISPLAY_QUANTILES = (0.005, 0.995)

METHOD_ORDER = ("dense", "block-sparse", "ST")
METHOD_LABELS = {
    "dense": "Dense GRAPE",
    "block-sparse": "Block-sparse GRAPE",
    "ST": "ST-GRAPE",
}
COLORS = {"Initial": "#777777", "dense": "#D55E00", "block-sparse": "#009E73", "ST": "#0072B2"}

if not (RESULT_ROOT / "run_manifest.json").is_file():
    raise FileNotFoundError(f"Select a completed stress-test run: {RESULT_ROOT}")
if not (MONTE_CARLO_ROOT / "monte_carlo_validation.json").is_file():
    raise FileNotFoundError(f"Select a completed Monte Carlo run: {MONTE_CARLO_ROOT}")
if SAVE_FIGURES:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

available_qubits = sorted(
    int(path.name[:-1])
    for path in RESULT_ROOT.glob("*q")
    if path.is_dir() and path.name[:-1].isdigit()
)
if N_QUBITS_OVERRIDE is None:
    if len(available_qubits) != 1:
        raise ValueError(f"Found qubit directories {available_qubits}; set N_QUBITS_OVERRIDE.")
    N_QUBITS = available_qubits[0]
elif N_QUBITS_OVERRIDE not in available_qubits:
    raise FileNotFoundError(f"{N_QUBITS_OVERRIDE}q is absent from {RESULT_ROOT}.")
else:
    N_QUBITS = N_QUBITS_OVERRIDE

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
print(f"Result root: {RESULT_ROOT}")
print(f"Figure output: {FIGURE_DIRECTORY}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def case_directory(seed: int) -> Path:
    return RESULT_ROOT / f"{N_QUBITS}q" / f"seed{seed}"


seed_sets: dict[str, set[int]] = {}
qubit_root = RESULT_ROOT / f"{N_QUBITS}q"
for method in METHOD_ORDER:
    seed_sets[method] = {
        int(path.name.removeprefix("seed"))
        for path in qubit_root.glob("seed*")
        if (path / method / "result.npz").is_file()
        and (path / method / "method_summary.json").is_file()
    }
AVAILABLE_METHODS = tuple(method for method in METHOD_ORDER if seed_sets[method])
if not AVAILABLE_METHODS:
    raise FileNotFoundError(f"No completed method results under {qubit_root}.")
paired_seeds = sorted(set.intersection(*(seed_sets[method] for method in AVAILABLE_METHODS)))
if SEEDS_OVERRIDE is not None:
    missing = sorted(set(SEEDS_OVERRIDE) - set(paired_seeds))
    if missing:
        raise FileNotFoundError(f"Seeds {missing} are not complete for {AVAILABLE_METHODS}.")
    SEEDS = tuple(SEEDS_OVERRIDE)
else:
    SEEDS = tuple(paired_seeds)
if not SEEDS:
    raise FileNotFoundError("No paired completed seeds were found.")

records: dict[tuple[int, str], dict[str, object]] = {}
for seed in SEEDS:
    for method in AVAILABLE_METHODS:
        directory = case_directory(seed) / method
        saved = load_npz(directory / "result.npz")
        summary = json.loads((directory / "method_summary.json").read_text(encoding="utf-8"))
        history = np.asarray(saved["objective_history"], dtype=float)
        elapsed = np.asarray(saved["elapsed_time_history"], dtype=float)
        record: dict[str, object] = {
            "directory": directory,
            "saved": saved,
            "summary": summary,
            "iteration": np.arange(history.size + 1, dtype=float),
            "time": np.r_[0.0, elapsed],
            "loss": np.r_[float(saved["initial_objective"]), history],
            "loss_kind": "Trotterized" if method == "ST" else "true",
        }
        fidelity_history = np.asarray(saved["fidelity_loss_history"], dtype=float)
        penalty_history = np.asarray(saved["robustness_penalty_history"], dtype=float)
        if (
            fidelity_history.size == history.size
            and penalty_history.size == history.size
            and np.isfinite(float(saved["initial_fidelity_loss"]))
            and np.isfinite(float(saved["initial_robustness_penalty"]))
        ):
            record["fidelity_loss"] = np.r_[
                float(saved["initial_fidelity_loss"]),
                fidelity_history,
            ]
            record["robustness_loss"] = np.r_[
                float(saved["initial_robustness_penalty"]),
                penalty_history,
            ]
        records[(seed, method)] = record

rows = []
for (seed, method), record in records.items():
    summary = record["summary"]
    rows.append(
        {
            "seed": seed,
            "method": method,
            "iterations": len(record["loss"]) - 1,
            "optimization_time_s": summary["wall_time_s"],
            "final_true_loss": summary["true_loss"],
            "final_nominal_fidelity": summary.get("final_nominal_fidelity"),
            "final_robustness_penalty": summary.get("final_robustness_penalty"),
        }
    )
if SAVE_FIGURES:
    with (FIGURE_DIRECTORY / "stress_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print(f"Paired seeds: {SEEDS}")
print(f"Available methods: {AVAILABLE_METHODS}")
for row in rows:
    print(row)


def save_figure(figure: plt.Figure, stem: str) -> None:
    if not SAVE_FIGURES:
        return
    figure.savefig(FIGURE_DIRECTORY / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(FIGURE_DIRECTORY / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.025)


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


def method_legend_handles(*, include_initial: bool = False) -> list[Line2D]:
    handles = []
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
        for method in AVAILABLE_METHODS
    )
    return handles


def convergence_trajectory(
    seed: int, method: str, coordinate: str
) -> tuple[np.ndarray, np.ndarray]:
    record = records[(seed, method)]
    x = np.asarray(record[coordinate], dtype=float)
    loss = np.asarray(record["loss"], dtype=float)
    if x.size != loss.size or not x.size or not np.all(np.isfinite(loss)) or np.any(loss <= 0):
        raise ValueError(f"Invalid loss history for seed={seed}, method={method}.")
    return x, np.minimum.accumulate(loss)


def common_coordinate_grid(trajectories: tuple[tuple[np.ndarray, np.ndarray], ...]) -> np.ndarray:
    start = max(float(x[0]) for x, _ in trajectories)
    stop = max(float(x[-1]) for x, _ in trajectories)
    events = [x[(x >= start) & (x <= stop)] for x, _ in trajectories]
    events.extend(
        np.asarray([np.nextafter(float(x[-1]), np.inf)])
        for x, _ in trajectories
        if float(x[-1]) < stop
    )
    return np.unique(np.concatenate(events))


def step_align(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(x, grid, side="right") - 1
    aligned = np.full(grid.shape, np.nan, dtype=float)
    valid = (indices >= 0) & (grid <= x[-1])
    aligned[valid] = values[np.clip(indices[valid], 0, values.size - 1)]
    return aligned


def plot_convergence(axis: plt.Axes, coordinate: str, label: str) -> np.ndarray:
    trajectories = {
        (method, seed): convergence_trajectory(seed, method, coordinate)
        for method in AVAILABLE_METHODS
        for seed in SEEDS
    }
    plotted_values = []
    x_max = max(float(x[-1]) for x, _ in trajectories.values())
    for method in METHOD_DRAW_ORDER:
        if method not in AVAILABLE_METHODS:
            continue
        method_trajectories = tuple(trajectories[(method, seed)] for seed in SEEDS)
        for seed_index, (x, loss) in enumerate(method_trajectories):
            foreground = "dense" if seed_index % 2 == 0 else "block-sparse"
            zorder = 5 if method == "ST" else (7 if method == foreground else 6)
            axis.plot(
                x,
                loss,
                color=COLORS[method],
                linestyle="--",
                linewidth=SEED_LINEWIDTH,
                alpha=SEED_ALPHA,
                drawstyle="steps-post",
                zorder=zorder,
            )
            axis.plot(
                x[-1],
                loss[-1],
                marker=ENDPOINT_MARKERS[method],
                markersize=ENDPOINT_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=COLORS[method],
                markeredgewidth=ENDPOINT_MARKER_EDGEWIDTH,
                alpha=ENDPOINT_MARKER_ALPHA,
                linestyle="none",
                zorder=zorder + 0.5,
                clip_on=False,
            )
        grid = common_coordinate_grid(method_trajectories)
        aligned = np.vstack([step_align(x, loss, grid) for x, loss in method_trajectories])
        active = np.any(np.isfinite(aligned), axis=0)
        grid, aligned = grid[active], aligned[:, active]
        plotted_values.append(aligned[np.isfinite(aligned)])
        axis.fill_between(
            grid,
            np.nanmin(aligned, axis=0),
            np.nanmax(aligned, axis=0),
            color=COLORS[method],
            alpha=CONVERGENCE_RANGE_ALPHA,
            linewidth=0,
            step="post",
            zorder=2,
        )
        if SHOW_CONVERGENCE_MEAN:
            axis.plot(
                grid,
                np.nanmean(aligned, axis=0),
                color=COLORS[method],
                linewidth=0.88,
                drawstyle="steps-post",
                zorder=4,
            )
        if method == "ST":
            for seed in SEEDS:
                saved = records[(seed, method)]["saved"]
                exact = np.asarray(saved["exact_objective_history"], dtype=float)
                if not exact.size:
                    continue
                key = (
                    "exact_evaluation_iterations"
                    if coordinate == "iteration"
                    else "exact_evaluation_times"
                )
                exact_x = float(np.asarray(saved[key], dtype=float)[-1])
                x_max = max(x_max, exact_x)
                plotted_values.append(np.asarray([exact[-1]]))
                axis.axhline(
                    float(exact[-1]),
                    color=COLORS["ST"],
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.58,
                    zorder=8,
                )
    axis.set_yscale("log")
    if coordinate == "time":
        positive = np.concatenate([x[x > 0] for x, _ in trajectories.values()])
        axis.set_xscale("log")
        axis.set_xlim(float(np.min(positive)), x_max)
    else:
        axis.set_xlim(0, x_max)
    axis.set_xlabel("Iteration" if coordinate == "iteration" else "Wall-clock time (s)")
    axis.set_ylabel("Loss")
    axis.grid(True, which="major", alpha=0.18, linewidth=0.45)
    axis.grid(False, which="minor", axis="y")
    panel_label(axis, label)
    return np.concatenate(plotted_values)


def apply_loss_limits(axes: tuple[plt.Axes, ...], values: tuple[np.ndarray, ...]) -> None:
    positive = np.concatenate(values)
    lo = np.floor(np.log10(np.min(positive)) - LOG_Y_PADDING_DECADES)
    hi = np.ceil(np.log10(np.max(positive)) + LOG_Y_PADDING_DECADES)
    for axis in axes:
        axis.set_ylim(10**lo, 10**hi)
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())


convergence_figure, convergence_axes = plt.subplots(1, 2, figsize=(8.35, 3.15))
loss_a = plot_convergence(convergence_axes[0], "iteration", "(a)")
loss_b = plot_convergence(convergence_axes[1], "time", "(b)")
apply_loss_limits(tuple(convergence_axes), (loss_a, loss_b))
convergence_handles = method_legend_handles()
if "ST" in AVAILABLE_METHODS:
    convergence_handles.append(
        Line2D(
            [0], [0], color=COLORS["ST"], linestyle=":", linewidth=1.05, label="ST final true loss"
        )
    )
convergence_figure.legend(
    handles=convergence_handles,
    loc="upper center",
    ncol=len(convergence_handles),
    frameon=False,
    bbox_to_anchor=(0.52, 0.995),
)
convergence_figure.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.81, wspace=0.27)
save_figure(convergence_figure, "exp03_loss_convergence")
plt.close("all")


def monte_carlo_case_directory(seed: int) -> Path:
    return MONTE_CARLO_ROOT / f"{N_QUBITS}q" / f"seed{seed}"


def monte_carlo_seed_is_complete(seed: int) -> bool:
    case = monte_carlo_case_directory(seed)
    return (case / "initial_fidelities.npz").is_file() and all(
        (case / method / "fidelities.npz").is_file() for method in AVAILABLE_METHODS
    )


MC_SEEDS = tuple(seed for seed in SEEDS if monte_carlo_seed_is_complete(seed))
if MC_SEEDS != SEEDS:
    raise FileNotFoundError(
        f"Monte Carlo samples are incomplete for paired seeds {SEEDS}: found {MC_SEEDS}."
    )
if MC_SEEDS:
    validation_path = MONTE_CARLO_ROOT / "monte_carlo_validation.json"
    sampling_seeds: set[int] = set()
    declared_sample_counts: set[int] = set()
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        sampling_seeds.add(int(validation["sampling_seed"]))
        declared_sample_counts.add(int(validation["samples_per_control"]))
    else:
        for seed in MC_SEEDS:
            for method in AVAILABLE_METHODS:
                metadata = json.loads(
                    (case_directory(seed) / method / "metadata.json").read_text(encoding="utf-8")
                )
                sampling_seeds.add(int(metadata["sampling_seed"]))
                declared_sample_counts.add(int(metadata["post_run_monte_carlo_samples"]))
    if len(sampling_seeds) != 1:
        raise ValueError(f"Methods do not share one Monte Carlo sampling seed: {sampling_seeds}")
    if declared_sample_counts != {MONTE_CARLO_SAMPLES}:
        raise ValueError(
            f"Expected {MONTE_CARLO_SAMPLES} Monte Carlo samples: {declared_sample_counts}"
        )
    if sampling_seeds != {MONTE_CARLO_SAMPLING_SEED}:
        raise ValueError(f"Expected sampling seed {MONTE_CARLO_SAMPLING_SEED}: {sampling_seeds}")


def load_fidelities(seed: int, method: str | None) -> np.ndarray:
    path = (
        monte_carlo_case_directory(seed) / "initial_fidelities.npz"
        if method is None
        else monte_carlo_case_directory(seed) / method / "fidelities.npz"
    )
    return np.asarray(load_npz(path)["values"], dtype=float)


def empirical_cdf(sorted_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, grid, side="right") / sorted_values.size


def plot_monte_carlo_cdf(axis: plt.Axes, label: str) -> None:
    if not MC_SEEDS:
        axis.text(
            0.5,
            0.5,
            "No saved Monte Carlo samples.\nRun exp03 with --samples N (N > 0).",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        panel_label(axis, label)
        return
    groups: list[tuple[str, str | None]] = [("Initial", None)] + [
        (METHOD_LABELS[method], method)
        for method in METHOD_DRAW_ORDER
        if method in AVAILABLE_METHODS
    ]
    errors: dict[tuple[str, int], np.ndarray] = {}
    sample_counts: set[int] = set()
    for group, method in groups:
        for seed in MC_SEEDS:
            fidelity = load_fidelities(seed, method)
            sample_counts.add(fidelity.size)
            errors[(group, seed)] = np.sort(np.maximum(1.0 - fidelity, INFIDELITY_FLOOR))
    if len(sample_counts) != 1:
        raise ValueError(f"Monte Carlo arrays do not share one sample count: {sample_counts}")
    if sample_counts != declared_sample_counts:
        raise ValueError(
            f"Saved sample count {sample_counts} disagrees with metadata {declared_sample_counts}."
        )
    lower_q, upper_q = CDF_DISPLAY_QUANTILES
    lower = min(float(np.quantile(values, lower_q)) for values in errors.values())
    upper = max(float(np.quantile(values, upper_q)) for values in errors.values())
    lower = max(lower, INFIDELITY_FLOOR)
    if upper <= lower:
        upper = max(float(values[-1]) for values in errors.values())
    display_lower = 10.0 ** (np.log10(lower) - 0.24)
    display_upper = 10.0 ** (np.log10(upper) + 0.24)
    grid = np.geomspace(display_lower, display_upper, 900)
    for group, method in groups:
        color = COLORS["Initial"] if method is None else COLORS[method]
        seed_cdfs = np.vstack([empirical_cdf(errors[(group, seed)], grid) for seed in MC_SEEDS])
        lower_bound = np.min(seed_cdfs, axis=0)
        upper_bound = np.max(seed_cdfs, axis=0)
        axis.fill_between(
            grid, lower_bound, upper_bound, color=color, alpha=0.17, linewidth=0, zorder=0
        )
        for bound in (lower_bound, upper_bound):
            axis.plot(
                grid,
                bound,
                color=color,
                linestyle=":",
                linewidth=BOUND_LINEWIDTH,
                alpha=BOUND_ALPHA,
                zorder=2,
            )
        axis.plot(
            grid,
            np.mean(seed_cdfs, axis=0),
            color=color,
            linewidth=MEAN_LINEWIDTH,
            linestyle=":" if method is None else "-",
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(display_lower, display_upper)
    axis.set_ylim(0.0, 1.005)
    axis.set_xlabel(r"Infidelity $\varepsilon_F=1-F$")
    axis.set_ylabel(r"Infidelity CDF $\mathcal{C}(\varepsilon)$")
    axis.grid(True, which="major", alpha=0.18, linewidth=0.45)
    axis.grid(False, which="minor", axis="y")
    panel_label(axis, label)
    print(
        f"CDF uses all {next(iter(sample_counts))} saved shared trajectories per seed; seeds={MC_SEEDS}."
    )


cdf_figure, cdf_axis = plt.subplots(figsize=(4.7, 3.65))
plot_monte_carlo_cdf(cdf_axis, "(c)")
cdf_figure.legend(
    handles=method_legend_handles(include_initial=True),
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
cdf_figure.subplots_adjust(left=0.15, right=0.985, bottom=0.14, top=0.83)
save_figure(cdf_figure, "exp03_monte_carlo_cdf")
plt.close("all")
if not MC_SEEDS:
    print(
        "No CDF was generated because this run used --samples 0. The optimizer results remain valid."
    )


# Final paper-format 1 x 3 composite figure.
paper_figure = plt.figure(figsize=(8.35, 3.10))
paper_grid = paper_figure.add_gridspec(
    1,
    3,
    left=0.075,
    right=0.995,
    bottom=0.19,
    top=0.79,
    width_ratios=(1.0, 1.0, 1.10),
    wspace=0.38,
)
axis_a = paper_figure.add_subplot(paper_grid[0, 0])
axis_b = paper_figure.add_subplot(paper_grid[0, 1], sharey=axis_a)
axis_c = paper_figure.add_subplot(paper_grid[0, 2])
loss_a = plot_convergence(axis_a, "iteration", "(a)")
loss_b = plot_convergence(axis_b, "time", "(b)")
apply_loss_limits((axis_a, axis_b), (loss_a, loss_b))
plot_monte_carlo_cdf(axis_c, "(c)")
paper_handles = method_legend_handles(include_initial=True)
if "ST" in AVAILABLE_METHODS:
    paper_handles.append(convergence_handles[-1])
paper_figure.legend(
    handles=paper_handles,
    loc="upper center",
    ncol=len(paper_handles),
    frameon=False,
    columnspacing=1.0,
    handlelength=2.2,
    bbox_to_anchor=(0.53, 0.99),
)
save_figure(paper_figure, "exp03_stress_test_three_panel")
plt.close("all")
print(f"Saved paper figure to {FIGURE_DIRECTORY / 'exp03_stress_test_three_panel.pdf'}")
