"""Generate nominal GHZ/QFT controls for subsequent robust optimization.

The command never runs from editable source-local parameters: use the CLI or a
versioned experiment preset. Controls are saved in the current result schema.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from stgrape import (
    DEFAULT_CHEBYSHEV_BASIS_SIZE,
    ControlBasis,
    OptimizationAlgorithm,
    OptimizationConfig,
    configure_ode_propagation,
    optimize,
)
from stgrape.benchmarks.common import ExperimentTask
from stgrape.execution import (
    PAPER_NOMINAL_ERROR_BOUND,
    PAPER_PULSE_CONFIG,
    prepare_resumable_output,
    progress_reporter,
    read_json_object,
    write_json,
)
from stgrape.models import build_paper_problem, default_paper_duration
from stgrape.optimizers import LBFGSBOptions
from stgrape.problem import ControlTask

EXPERIMENT_NAME = "E00 nominal seeds"
DEFAULT_OUTPUT_ROOT = Path("outputs/experiments/nominal")

PAPER_QFT_QUBITS = (2, 3, 4, 5)
PAPER_GHZ_QUBITS = tuple(range(2, 11))
PAPER_STANDARD_SEEDS = tuple(range(5))
PAPER_LARGE_STATE_SEEDS = tuple(range(3))
PAPER_TIME_STEP_US = 0.0005
DEFAULT_MAX_ITERATIONS = 100_000
PAPER_FUNCTION_TOLERANCE = 1e-10
PAPER_GRADIENT_TOLERANCE = 1e-10
WORKERS = 4
DEFAULT_PLOT_FIRST_N = 4
ExistingSeedAction = Literal["skip", "overwrite"]
SeedCase = tuple[ControlTask, int, int]


def default_output_directory() -> Path:
    """Return a timestamped output directory for the revised E00 workflow."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return DEFAULT_OUTPUT_ROOT / timestamp


@dataclass(frozen=True)
class NominalSeedJob:
    """Small, picklable description of one independent nominal optimization."""

    task: str
    n_qubits: int
    coupling: str
    # ``seed`` is the stable slot consumed by later paper experiments.  A
    # rejected optimization may be retried from a different random seed while
    # retaining this output slot.
    seed: int
    initialization_seed: int
    attempt: int
    error_bound: float
    time_step: float
    duration: float
    max_iterations: int
    ftol: float
    gtol: float
    ode_method: str
    ode_representation: str
    ode_rtol: float | None
    ode_atol: float | None
    time_limit_hours: float | None
    log_interval: int
    worker_count: int
    output_root: Path
    control_basis: ControlBasis
    chebyshev_k: int


def validate_nominal_error_bound(error_bound: float) -> float:
    """Return a physical nominal-error threshold or raise a clear error."""

    if not math.isfinite(error_bound) or not 0.0 < error_bound <= 1.0:
        raise ValueError("error_bound must be finite and lie in (0, 1].")
    return error_bound


def nominal_seed_is_acceptable(loss: float, error_bound: float) -> bool:
    """Whether a nominal run is accurate enough to become a paper seed."""

    validate_nominal_error_bound(error_bound)
    return math.isfinite(loss) and loss <= error_bound


def build_optimizer_options(
    *,
    max_iterations: int,
    ftol: float = PAPER_FUNCTION_TOLERANCE,
    gtol: float = PAPER_GRADIENT_TOLERANCE,
) -> LBFGSBOptions:
    """Build and validate the L-BFGS-B stopping configuration.

    ``ftol`` controls the relative reduction in the objective, while ``gtol``
    controls the maximum projected-gradient component.  The defaults preserve
    the original paper runs, but both can be relaxed from the command line.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not math.isfinite(ftol) or ftol < 0:
        raise ValueError("ftol must be finite and non-negative.")
    if not math.isfinite(gtol) or gtol < 0:
        raise ValueError("gtol must be finite and non-negative.")
    return LBFGSBOptions(
        max_iterations=max_iterations,
        function_tolerance=ftol,
        gradient_tolerance=gtol,
    )


def _run_nominal_seed(job: NominalSeedJob) -> dict[str, Any]:
    """Optimize and save one seed in the main process or a worker process.

    Each process constructs its own Hamiltonian and ODE solver.  This avoids
    serializing large QuTiP/NumPy system objects on Windows.  Pulse arrays are
    saved directly to the seed's unique directory; only a compact summary is
    returned to the parent process.
    """

    task = ControlTask(job.task)
    chain, problem = build_paper_problem(
        job.n_qubits,
        task,
        time_step=job.time_step,
        duration=job.duration,
        coupling=job.coupling,
    )
    propagation = configure_ode_propagation(
        method=job.ode_method,
        representation=job.ode_representation,
        rtol=job.ode_rtol,
        atol=job.ode_atol,
    )
    optimization = OptimizationConfig(
        algorithm=OptimizationAlgorithm.GRAPE,
        optimizer=build_optimizer_options(
            max_iterations=job.max_iterations,
            ftol=job.ftol,
            gtol=job.gtol,
        ),
        propagation=propagation,
        time_limit_seconds=(
            None if job.time_limit_hours is None else 3600.0 * job.time_limit_hours
        ),
    )

    label = (
        f"{task.value}-{job.n_qubits}q-slot{job.seed}-"
        f"{job.coupling}-rng{job.initialization_seed}-attempt{job.attempt}"
    )
    print(f"Generating nominal seed candidate {label}...", flush=True)
    pulse = PAPER_PULSE_CONFIG.random_pulse(
        problem,
        seed=job.initialization_seed,
        control_basis=job.control_basis,
        chebyshev_k=job.chebyshev_k,
    )
    run = optimize(
        problem,
        pulse,
        None,
        optimization,
        control_groups=chain.control_groups,
        progress_callback=progress_reporter(label, job.log_interval),
    )
    run_directory = (
        job.output_root
        / "attempts"
        / task.value
        / f"{job.n_qubits}q"
        / f"seed{job.seed}"
        / f"attempt{job.attempt}_rng{job.initialization_seed}"
    )
    run.result.save(
        run_directory,
        metadata={
            "experiment": "E00 nominal seeds",
            "paper_scope": "initial controls for Tables I-V",
            "task": task.value,
            "n_qubits": job.n_qubits,
            "coupling": job.coupling,
            "seed": job.seed,
            "initialization_seed": job.initialization_seed,
            "attempt": job.attempt,
            "nominal_error_bound": job.error_bound,
            "duration_us": problem.duration,
            "time_step_us": problem.time_step,
            "n_time_steps": problem.n_time_steps,
            "optimization": optimization.metadata(),
            "parallel_workers": job.worker_count,
            "end_to_end_wall_time_s": run.elapsed_seconds,
            "pulse": {
                **pulse.metadata(),
                "envelope": "Blackman",
                "rise_time_us": 0.01,
                "cutoff_frequency_mhz": 60.0,
                "filter_taps": 10,
                "initial_frequency_range_mhz": [1.0, 30.0],
                "initial_max_amplitude_mhz": 30.0,
                "n_waves": 10,
            },
        },
    )
    row = {
        "task": task.value,
        "n_qubits": job.n_qubits,
        "coupling": job.coupling,
        "seed": job.seed,
        "initialization_seed": job.initialization_seed,
        "attempt": job.attempt,
        "nominal_loss": run.result.objective,
        "nominal_fidelity": 1.0 - run.result.objective,
        "error_bound": job.error_bound,
        "accepted": nominal_seed_is_acceptable(
            run.result.objective,
            job.error_bound,
        ),
        "wall_time_s": run.elapsed_seconds,
        "iterations": len(run.result.objective_history),
        "success": run.result.success,
        "result": str(run_directory / "result.npz"),
    }
    print(
        f"  [{label}] nominal loss={row['nominal_loss']:.8g}, "
        f"fidelity={row['nominal_fidelity']:.8f}, "
        f"accepted={row['accepted']}, wall={run.elapsed_seconds:.2f} s",
        flush=True,
    )
    return row


def _run_candidate_batch(
    jobs: tuple[NominalSeedJob, ...],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Run one retry round, preserving logical-seed order."""

    if workers == 1:
        return [_run_nominal_seed(job) for job in jobs]
    process_count = min(workers, len(jobs))
    print(
        f"Running {len(jobs)} candidate(s) with {process_count} worker processes...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        return list(executor.map(_run_nominal_seed, jobs))


def _promote_accepted_seed(
    row: dict[str, Any],
    output_root: Path,
    *,
    overwrite_existing: bool = False,
) -> None:
    """Move an accepted attempt into its stable slot, optionally replacing it."""

    candidate_directory = Path(str(row["result"])).parent
    destination = (
        output_root / str(row["task"]) / f"{int(row['n_qubits'])}q" / f"seed{int(row['seed'])}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite_existing:
        raise FileExistsError(f"Nominal seed slot already exists: {destination}")
    if destination.exists():
        backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
        destination.rename(backup)
        try:
            shutil.move(str(candidate_directory), str(destination))
        except Exception:
            if not destination.exists() and backup.exists():
                backup.rename(destination)
            raise
        else:
            shutil.rmtree(backup)
        print(f"  Replaced existing nominal seed at {destination}", flush=True)
    else:
        shutil.move(str(candidate_directory), str(destination))
    row["result"] = str(destination / "result.npz")


def generate_nominal_seed_case(
    *,
    task: ControlTask,
    n_qubits: int,
    output_seeds: tuple[int, ...],
    output_root: Path,
    coupling: str = "XX+YY",
    time_step: float,
    duration: float | None = None,
    max_iterations: int,
    ftol: float,
    gtol: float,
    ode_method: str,
    ode_representation: str,
    ode_rtol: float | None,
    ode_atol: float | None,
    time_limit_hours: float | None,
    log_interval: int,
    workers: int,
    error_bound: float = PAPER_NOMINAL_ERROR_BOUND,
    max_attempts_per_seed: int = 0,
    initialization_seeds: dict[int, int] | None = None,
    control_basis: ControlBasis = "pwc",
    chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
    overwrite_existing: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate enough accepted controls to fill all requested seed slots.

    ``output_seeds`` define stable downstream slots.  Failed candidates are
    retained below ``attempts/`` and retried with fresh initialization seeds.
    A zero ``max_attempts_per_seed`` means retry until every slot is accepted.
    """

    validate_nominal_error_bound(error_bound)
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if max_attempts_per_seed < 0:
        raise ValueError("max_attempts_per_seed must be non-negative.")
    if not output_seeds:
        raise ValueError("At least one output seed is required.")
    if len(set(output_seeds)) != len(output_seeds):
        raise ValueError("Output seeds must be unique.")
    if control_basis not in {"pwc", "chebyshev"}:
        raise ValueError("control_basis must be 'pwc' or 'chebyshev'.")
    if chebyshev_k < 1:
        raise ValueError("chebyshev_k must be positive.")
    resolved_duration = default_paper_duration(task) if duration is None else float(duration)
    if not math.isfinite(resolved_duration) or resolved_duration <= 0.0:
        raise ValueError("duration must be finite and positive.")

    supplied = initialization_seeds or {}
    unknown_slots = set(supplied).difference(output_seeds)
    if unknown_slots:
        raise ValueError(f"Initialization seeds supplied for unknown slots: {unknown_slots}")
    pending = {seed: (supplied.get(seed, seed), 1) for seed in output_seeds}
    used_initialization_seeds = set(output_seeds) | set(supplied.values())
    next_initialization_seed = max(used_initialization_seeds, default=-1) + 1
    accepted_rows: dict[int, dict[str, Any]] = {}
    attempt_rows: list[dict[str, Any]] = []

    while pending:
        process_count = min(workers, len(pending))
        jobs = tuple(
            NominalSeedJob(
                task=task.value,
                n_qubits=n_qubits,
                coupling=coupling,
                seed=seed,
                initialization_seed=initialization_seed,
                attempt=attempt,
                error_bound=error_bound,
                time_step=time_step,
                duration=resolved_duration,
                max_iterations=max_iterations,
                ftol=ftol,
                gtol=gtol,
                ode_method=ode_method,
                ode_representation=ode_representation,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
                time_limit_hours=time_limit_hours,
                log_interval=log_interval,
                worker_count=process_count,
                output_root=output_root,
                control_basis=control_basis,
                chebyshev_k=chebyshev_k,
            )
            for seed, (initialization_seed, attempt) in sorted(pending.items())
        )
        pending.clear()
        for row in _run_candidate_batch(jobs, workers=workers):
            attempt_rows.append(row)
            seed = int(row["seed"])
            attempt = int(row["attempt"])
            if bool(row["accepted"]):
                _promote_accepted_seed(
                    row,
                    output_root,
                    overwrite_existing=overwrite_existing,
                )
                accepted_rows[seed] = row
                continue

            if max_attempts_per_seed and attempt >= max_attempts_per_seed:
                raise RuntimeError(
                    f"{task.value}-{n_qubits}q seed slot {seed} did not reach "
                    f"loss <= {error_bound:g} in {attempt} attempt(s)."
                )
            while next_initialization_seed in used_initialization_seeds:
                next_initialization_seed += 1
            print(
                f"  Rejecting slot seed{seed}: loss={float(row['nominal_loss']):.8g} "
                f"> {error_bound:g}; retrying with random seed "
                f"{next_initialization_seed}.",
                flush=True,
            )
            pending[seed] = (next_initialization_seed, attempt + 1)
            used_initialization_seeds.add(next_initialization_seed)
            next_initialization_seed += 1

    return [accepted_rows[seed] for seed in output_seeds], attempt_rows


def seeds_for_case(
    task: ControlTask,
    n_qubits: int,
    override: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return the paper seed count, unless the CLI explicitly overrides it."""

    if override is not None:
        return override
    if task == ControlTask.STATE_PREPARATION and n_qubits >= 7:
        return PAPER_LARGE_STATE_SEEDS
    return PAPER_STANDARD_SEEDS


def _selected_tasks(value: str) -> tuple[ControlTask, ...]:
    """Map the public GHZ/QFT target names to package-level objective types."""

    if value == "both":
        return (
            ExperimentTask.QFT.control_task,
            ExperimentTask.GHZ.control_task,
        )
    return (ExperimentTask(value).control_task,)


def _qubits_for_task(
    task: ControlTask,
    override: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return explicit qubit counts or the paper defaults for one task."""

    if override is not None:
        return override
    if task == ControlTask.GATE_SYNTHESIS:
        return PAPER_QFT_QUBITS
    return PAPER_GHZ_QUBITS


def _nominal_result_row(directory: Path) -> dict[str, Any]:
    """Reconstruct one E00 manifest row from its stable result directory."""

    metadata_path = directory / "metadata.json"
    result_path = directory / "result.npz"
    if not metadata_path.is_file() or not result_path.is_file():
        raise RuntimeError(
            f"Incomplete nominal-seed output detected at {directory}; move or "
            "remove it before continuing this run."
        )
    metadata = read_json_object(metadata_path)
    if metadata.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(f"{directory} belongs to a different experiment.")
    with np.load(result_path) as result:
        objective = float(result["objective"])
        iterations = int(np.asarray(result["objective_history"]).size)
    error_bound = float(metadata["nominal_error_bound"])
    return {
        "task": str(metadata["task"]),
        "n_qubits": int(metadata["n_qubits"]),
        "coupling": str(metadata.get("coupling", "ZZ")),
        "seed": int(metadata["seed"]),
        "initialization_seed": int(metadata["initialization_seed"]),
        "attempt": int(metadata["attempt"]),
        "nominal_loss": objective,
        "nominal_fidelity": 1.0 - objective,
        "error_bound": error_bound,
        "accepted": nominal_seed_is_acceptable(objective, error_bound),
        "wall_time_s": float(metadata["end_to_end_wall_time_s"]),
        "iterations": iterations,
        "success": bool(metadata["success"]),
        "result": str(result_path),
    }


def _load_nominal_rows(output: Path) -> list[dict[str, Any]]:
    """Load all completed stable E00 seed slots from disk."""

    rows: list[dict[str, Any]] = []
    for task in (ControlTask.GATE_SYNTHESIS, ControlTask.STATE_PREPARATION):
        task_directory = output / task.value
        if not task_directory.exists():
            continue
        for qubit_directory in task_directory.glob("*q"):
            if not qubit_directory.name[:-1].isdigit():
                raise ValueError(f"Invalid qubit directory: {qubit_directory}")
            for seed_directory in qubit_directory.glob("seed*"):
                suffix = seed_directory.name.removeprefix("seed")
                if seed_directory.is_dir() and suffix.isdigit():
                    rows.append(_nominal_result_row(seed_directory))
    rows.sort(key=lambda row: (str(row["task"]), int(row["n_qubits"]), int(row["seed"])))
    return rows


def _load_nominal_attempt_rows(
    output: Path,
    completed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load rejected candidates and combine them with promoted attempts."""

    rows = [dict(row) for row in completed_rows]
    attempts_root = output / "attempts"
    if attempts_root.exists():
        for metadata_path in attempts_root.rglob("metadata.json"):
            rows.append(_nominal_result_row(metadata_path.parent))
    rows.sort(
        key=lambda row: (
            str(row["task"]),
            int(row["n_qubits"]),
            int(row["seed"]),
            int(row["attempt"]),
        )
    )
    return rows


def _save_nominal_control_overview(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    task: ControlTask,
    n_qubits: int,
    first_n: int,
    time_step: float,
) -> tuple[Path, Path]:
    """Save one multi-panel comparison of the first nominal controls.

    Each panel contains one optimized seed. Qubit color and X/Y line style are
    held fixed across panels, while common axes make pulse amplitudes directly
    comparable. The function uses a non-interactive canvas so it also works in
    VS Code remote sessions and headless CI jobs.
    """

    if first_n < 1:
        raise ValueError("plot-first-n must be positive.")
    selected = sorted(rows, key=lambda row: int(row["seed"]))[:first_n]
    if not selected:
        raise ValueError(f"No nominal controls are available for {task.value}-{n_qubits}q.")

    controls: list[np.ndarray] = []
    for row in selected:
        result_path = Path(str(row["result"]))
        with np.load(result_path, allow_pickle=False) as saved:
            if "control_amplitudes" not in saved.files:
                raise ValueError(f"Current result is missing control_amplitudes: {result_path}")
            amplitudes = np.asarray(saved["control_amplitudes"], dtype=float)
        if amplitudes.ndim != 2 or amplitudes.shape[0] != 2 * n_qubits:
            raise ValueError(
                f"Expected {2 * n_qubits} control channels in {result_path}; "
                f"received shape {amplitudes.shape}."
            )
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError(f"Control amplitudes must be finite: {result_path}")
        controls.append(amplitudes)

    n_time_steps = controls[0].shape[1]
    if any(control.shape[1] != n_time_steps for control in controls):
        raise ValueError("Nominal controls in one overview must share a time grid.")
    times_ns = 1e3 * time_step * np.arange(n_time_steps + 1)
    scale = 2.0 * np.pi
    peak_mhz = max(float(np.max(np.abs(control))) for control in controls) / scale
    amplitude_limit = 1.05 * peak_mhz if peak_mhz > 0.0 else 1.0

    n_panels = len(selected)
    n_columns = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_columns)
    figure = Figure(figsize=(7.0 * n_columns, 3.4 * n_rows + 1.0))
    _ = FigureCanvasAgg(figure)
    axes = np.asarray(figure.subplots(n_rows, n_columns, sharex=True, sharey=True, squeeze=False))
    colors = colormaps.get_cmap("tab10").resampled(n_qubits)
    line_styles = ("-", "--")

    for axis, row, control in zip(axes.flat, selected, controls, strict=False):
        for qubit in range(n_qubits):
            for quadrature, line_style in enumerate(line_styles):
                channel = quadrature * n_qubits + qubit
                values_mhz = control[channel] / scale
                axis.step(
                    times_ns,
                    np.r_[values_mhz, values_mhz[-1]],
                    where="post",
                    color=colors(qubit),
                    linestyle=line_style,
                    linewidth=1.0,
                    alpha=0.9,
                )
        axis.set_title(f"seed {int(row['seed'])}: $F={float(row['nominal_fidelity']):.6f}$")
        axis.set_xlim(times_ns[0], times_ns[-1])
        axis.set_ylim(-amplitude_limit, amplitude_limit)
        axis.grid(alpha=0.2, linewidth=0.6)

    for axis in axes.flat[n_panels:]:
        axis.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=colors(qubit), linewidth=1.6, label=f"Q{qubit + 1}")
        for qubit in range(n_qubits)
    ]
    legend_handles.extend(
        (
            Line2D([0], [0], color="black", linestyle="-", label="X control"),
            Line2D([0], [0], color="black", linestyle="--", label="Y control"),
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(6, len(legend_handles)),
        frameon=False,
    )
    figure.supxlabel("Time (ns)")
    figure.supylabel("Control amplitude (MHz)")
    task_label = "QFT gate" if task == ControlTask.GATE_SYNTHESIS else "GHZ state"
    figure.suptitle(f"{task_label}, {n_qubits} qubits: first {n_panels} nominal controls")
    figure.tight_layout(rect=(0.02, 0.10, 1.0, 0.95))

    figure_directory = output / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    stem = figure_directory / f"nominal_controls_{task.value}_{n_qubits}q"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    print(
        f"Saved first {n_panels} {task.value}-{n_qubits}q controls to {png_path}",
        flush=True,
    )
    return png_path, pdf_path


def _select_pending_seed_cases(
    cases: Sequence[SeedCase],
    output: Path,
    *,
    existing_seed: ExistingSeedAction,
) -> tuple[tuple[SeedCase, ...], tuple[SeedCase, ...]]:
    """Apply the requested skip/overwrite action to existing stable seed slots.

    A skipped directory must contain a complete accepted result.  Overwriting
    clears only rejected candidate attempts before optimization; the stable
    result remains available until an accepted replacement is ready.
    """

    requested = tuple(cases)
    if existing_seed not in {"skip", "overwrite"}:
        raise ValueError("existing_seed must be 'skip' or 'overwrite'.")
    if len(set(requested)) != len(requested):
        raise ValueError("Requested nominal seed cases must be unique.")

    pending: list[SeedCase] = []
    skipped: list[SeedCase] = []
    reset_attempt_directories: list[Path] = []
    overwritten_directories: list[Path] = []
    for task, n_qubits, seed in requested:
        destination = output / task.value / f"{n_qubits}q" / f"seed{seed}"
        attempts = output / "attempts" / task.value / f"{n_qubits}q" / f"seed{seed}"
        if destination.exists() and existing_seed == "skip":
            row = _nominal_result_row(destination)
            if not bool(row["accepted"]):
                raise ValueError(
                    f"Existing nominal seed does not satisfy its error bound: {destination}. "
                    "Use --existing-seed overwrite to regenerate it."
                )
            skipped.append((task, n_qubits, seed))
            continue
        if attempts.exists() and existing_seed == "skip":
            raise RuntimeError(
                f"Incomplete candidate history exists without a completed seed at {attempts}. "
                "Use --existing-seed overwrite to restart this seed."
            )
        pending.append((task, n_qubits, seed))
        if existing_seed == "overwrite":
            if destination.exists():
                overwritten_directories.append(destination)
            if attempts.exists():
                reset_attempt_directories.append(attempts)

    if skipped:
        print("Skipping existing accepted nominal seeds:", flush=True)
        for task, n_qubits, seed in skipped:
            print(f"  - {output / task.value / f'{n_qubits}q' / f'seed{seed}'}", flush=True)
    if overwritten_directories:
        print(
            "Warning: accepted replacements will overwrite these nominal seeds:",
            flush=True,
        )
        for directory in overwritten_directories:
            print(f"  - {directory}", flush=True)
    for directory in reset_attempt_directories:
        print(f"Warning: removing obsolete candidate attempts at {directory}", flush=True)
        shutil.rmtree(directory)
    return tuple(pending), tuple(skipped)


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Generate every selected nominal seed and write a manifest."""

    if arguments.workers < 1:
        raise ValueError("workers must be at least 1.")
    if arguments.plot_first_n < 1:
        raise ValueError("plot-first-n must be positive.")
    if arguments.chebyshev_k < 1:
        raise ValueError("chebyshev-k must be positive.")
    error_bound = validate_nominal_error_bound(arguments.error_bound)
    max_attempts_per_seed = arguments.max_attempts_per_seed
    task_selection = arguments.task
    qubit_override = None if arguments.qubits is None else tuple(arguments.qubits)
    seed_override = None if arguments.seeds is None else tuple(arguments.seeds)
    time_step = arguments.time_step
    duration_override = arguments.duration
    max_iterations = arguments.max_iterations
    if arguments.smoke:
        task_selection = "both"
        qubit_override = (2,)
        seed_override = (0,)
        time_step = 0.0025
        max_iterations = 1
        # Smoke mode checks wiring, serialization, and both task types.  A
        # one-iteration run is not a scientific convergence test, so it uses
        # the physical upper bound and exactly one attempt.
        error_bound = 1.0
        max_attempts_per_seed = 1

    propagation = configure_ode_propagation(
        method=arguments.ode_method,
        representation=arguments.ode_representation,
        rtol=arguments.ode_rtol,
        atol=arguments.ode_atol,
    )
    optimizer = build_optimizer_options(
        max_iterations=max_iterations,
        ftol=arguments.ftol,
        gtol=arguments.gtol,
    )
    requested_cases = tuple(
        (task, n_qubits, seed)
        for task in _selected_tasks(task_selection)
        for n_qubits in _qubits_for_task(task, qubit_override)
        for seed in seeds_for_case(task, n_qubits, seed_override)
    )
    output = arguments.output or default_output_directory()
    configuration = {
        "coupling": arguments.coupling,
        "time_step_us": time_step,
        "max_iterations": max_iterations,
        "ftol": optimizer.function_tolerance,
        "gtol": optimizer.gradient_tolerance,
        "error_bound": error_bound,
        "max_attempts_per_seed": max_attempts_per_seed,
        "ode": propagation.metadata(),
        "time_limit_hours": arguments.time_limit_hours,
        "log_interval": arguments.log_interval,
        "control_basis": arguments.control_basis,
        "chebyshev_k": (arguments.chebyshev_k if arguments.control_basis == "chebyshev" else None),
        "smoke": arguments.smoke,
    }
    if duration_override is not None:
        if not math.isfinite(duration_override) or duration_override <= 0.0:
            raise ValueError("duration must be finite and positive.")
        configuration["duration_us"] = duration_override
    prepare_resumable_output(
        output,
        experiment=EXPERIMENT_NAME,
        configuration=configuration,
    )
    cases, skipped_cases = _select_pending_seed_cases(
        requested_cases,
        output,
        existing_seed=arguments.existing_seed,
    )

    selected_groups: dict[tuple[ControlTask, int], list[int]] = {}
    for task, n_qubits, seed in cases:
        selected_groups.setdefault((task, n_qubits), []).append(seed)
    for (task, n_qubits), selected_seed_list in selected_groups.items():
        generate_nominal_seed_case(
            task=task,
            n_qubits=n_qubits,
            output_seeds=tuple(selected_seed_list),
            output_root=output,
            coupling=arguments.coupling,
            time_step=time_step,
            duration=duration_override,
            max_iterations=max_iterations,
            ftol=optimizer.function_tolerance,
            gtol=optimizer.gradient_tolerance,
            ode_method=arguments.ode_method,
            ode_representation=arguments.ode_representation,
            ode_rtol=arguments.ode_rtol,
            ode_atol=arguments.ode_atol,
            time_limit_hours=arguments.time_limit_hours,
            log_interval=arguments.log_interval,
            workers=arguments.workers,
            error_bound=error_bound,
            max_attempts_per_seed=max_attempts_per_seed,
            control_basis=arguments.control_basis,
            chebyshev_k=arguments.chebyshev_k,
            overwrite_existing=arguments.existing_seed == "overwrite",
        )

    rows = _load_nominal_rows(output)
    attempts = _load_nominal_attempt_rows(output, rows)
    requested_groups: dict[tuple[ControlTask, int], set[int]] = {}
    for task, n_qubits, seed in requested_cases:
        requested_groups.setdefault((task, n_qubits), set()).add(seed)
    pulse_figure_paths: list[Path] = []
    for (task, n_qubits), requested_seeds in requested_groups.items():
        group_rows = [
            row
            for row in rows
            if row["task"] == task.value
            and int(row["n_qubits"]) == n_qubits
            and int(row["seed"]) in requested_seeds
        ]
        if len(group_rows) < arguments.plot_first_n:
            print(
                f"Requested the first {arguments.plot_first_n} controls for "
                f"{task.value}-{n_qubits}q, but only {len(group_rows)} are "
                "available; plotting all available controls.",
                flush=True,
            )
        pulse_figure_paths.extend(
            _save_nominal_control_overview(
                group_rows,
                output,
                task=task,
                n_qubits=n_qubits,
                first_n=arguments.plot_first_n,
                time_step=time_step,
            )
        )
    completed_cases: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        completed_cases.setdefault(str(row["task"]), {}).setdefault(
            str(int(row["n_qubits"])), []
        ).append(int(row["seed"]))
    manifest = {
        "experiment": EXPERIMENT_NAME,
        "scope": ("smoke" if arguments.smoke else "completed case batches"),
        "configuration": {
            **configuration,
            "workers": arguments.workers,
            "completed_cases": completed_cases,
        },
        "last_requested_cases": [
            {"task": task.value, "n_qubits": n_qubits, "seed": seed}
            for task, n_qubits, seed in requested_cases
        ],
        "last_existing_seed_action": arguments.existing_seed,
        "last_plot_first_n": arguments.plot_first_n,
        "pulse_overview_figures": [str(path.relative_to(output)) for path in pulse_figure_paths],
        "last_skipped_cases": [
            {"task": task.value, "n_qubits": n_qubits, "seed": seed}
            for task, n_qubits, seed in skipped_cases
        ],
        "runs": rows,
        "attempts": attempts,
    }
    write_json(output / "manifest.json", manifest)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Saved {len(rows)} total nominal seeds "
        f"({len(requested_cases)} requested, {len(cases)} optimized, "
        f"{len(skipped_cases)} skipped in this batch) to {output}",
        flush=True,
    )
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse full-paper controls and the reduced smoke preset."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "compatible result directory that may be reused for unfinished cases; "
            "existing seed handling is controlled by --existing-seed; defaults to "
            "outputs/experiments/nominal/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--existing-seed",
        choices=("skip", "overwrite"),
        default="skip",
        help=(
            "skip an existing accepted seed or optimize and atomically replace it "
            "after the new candidate passes the error bound (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--task",
        choices=(ExperimentTask.GHZ.value, ExperimentTask.QFT.value, "both"),
        default="both",
        help="physical target to generate: GHZ states, QFT gates, or both",
    )
    parser.add_argument(
        "--coupling",
        choices=("ZZ", "XX+YY"),
        default="XX+YY",
        help="nominal nearest-neighbour coupling (default: %(default)s)",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        help=(
            "qubit counts for every selected task; when omitted, use the paper "
            "defaults 2-5 for QFT and 2-10 for GHZ"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="override paper seed counts for every selected case",
    )
    parser.add_argument(
        "--plot-first-n",
        type=int,
        default=DEFAULT_PLOT_FIRST_N,
        help=(
            "save a multi-panel overview of the first N optimized controls for "
            "each task/qubit case; if fewer are available, plot all of them "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument("--time-step", type=float, default=PAPER_TIME_STEP_US)
    parser.add_argument(
        "--duration",
        type=float,
        help=(
            "total pulse duration in microseconds; defaults to 0.030 for GHZ/state "
            "and 0.100 for QFT/gate"
        ),
    )
    parser.add_argument(
        "--control-basis",
        choices=("pwc", "chebyshev"),
        default="pwc",
        help="optimization coordinates for every generated seed (default: %(default)s)",
    )
    parser.add_argument(
        "--chebyshev-k",
        type=int,
        default=DEFAULT_CHEBYSHEV_BASIS_SIZE,
        help=(
            "number of Chebyshev basis functions (orders 0,...,K-1); "
            "used only with --control-basis chebyshev (default: %(default)s)"
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--ftol",
        type=float,
        default=PAPER_FUNCTION_TOLERANCE,
        help="L-BFGS-B relative function tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=PAPER_GRADIENT_TOLERANCE,
        help="L-BFGS-B projected-gradient tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--error-bound",
        type=float,
        default=PAPER_NOMINAL_ERROR_BOUND,
        help=(
            "maximum accepted nominal infidelity; larger final errors are "
            "retried from a fresh random field (default: %(default)g)"
        ),
    )
    parser.add_argument(
        "--max-attempts-per-seed",
        type=int,
        default=0,
        help=(
            "safety cap on random initializations per output seed; "
            "0 retries until enough accepted seeds exist (default: %(default)s)"
        ),
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=(
            "seed-level process count for each task/qubit case; "
            "1 disables multiprocessing (default: %(default)s)"
        ),
    )
    parser.add_argument("--ode-method", choices=("dop853",), default="dop853")
    parser.add_argument(
        "--ode-representation",
        choices=("dense", "block-sparse"),
        default="dense",
    )
    parser.add_argument("--ode-rtol", type=float)
    parser.add_argument("--ode-atol", type=float)
    parser.add_argument("--time-limit-hours", type=float)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one two-qubit, one-iteration seed for each task",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run from terminal arguments or an editable in-file argument sequence."""

    run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
