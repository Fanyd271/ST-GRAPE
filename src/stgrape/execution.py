"""Shared execution helpers for reproducible optimization experiments.

This module contains execution plumbing, not a hidden scientific model. The
experiment entry points state their Hamiltonian and noise configuration
explicitly; these helpers keep seed loading, optimization, progress reporting,
serialization, and plotting consistent.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from stgrape import (
    DEFAULT_CHEBYSHEV_BASIS_SIZE,
    ControlBasis,
    OptimizationAlgorithm,
    OptimizationConfig,
    OptimizationRun,
    PulseConfig,
    configure_ode_propagation,
    optimize,
)
from stgrape.diagnostics import (
    TaylorSensitivityDiagnostics,
    evaluate_taylor_sensitivity_diagnostics,
    sensitivity_pair_metadata,
)
from stgrape.io import pulse_from_result
from stgrape.optimizers import LBFGSBOptions
from stgrape.optimizers.grape import GrapeSolver
from stgrape.paths import ensure_output_path
from stgrape.problem import ControlProblem, ControlTask, RobustnessModel
from stgrape.propagation.ode import build_ode_components
from stgrape.pulse import Pulse
from stgrape.sampling import ColoredNoiseSamplingMethod, sample_fidelities
from stgrape.systems import CouplingModel, SpinChain, normalize_coupling_model

PAPER_TIME_STEP_US = 0.0005
PAPER_PULSE_CONFIG = PulseConfig()
PAPER_NOMINAL_ERROR_BOUND = 0.01
RUN_MANIFEST_NAME = "run_manifest.json"
EXPERIMENT_METHODS = ("ST", "dense", "block-sparse")


def _rename_directory_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 6,
    initial_delay_seconds: float = 0.02,
) -> None:
    """Rename a result directory despite transient Windows file locks."""

    for attempt in range(attempts):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(initial_delay_seconds * 2**attempt)


@dataclass(frozen=True)
class PaperMethodRun:
    """One independently executable robust-control method and its initial loss."""

    method: str
    optimization: OptimizationRun
    initial_true_loss: float | None
    initial_sensitivity: TaylorSensitivityDiagnostics | None
    final_sensitivity: TaylorSensitivityDiagnostics | None
    final_nominal_fidelity: float | None = None
    final_robustness_penalty: float | None = None
    post_run_true_loss_wall_time_s: float | None = None

    @property
    def true_loss(self) -> float:
        """Return the final ODE-reference loss used for method comparison."""

        result = self.optimization.result
        if self.method == "ST":
            if result.exact_objective is None:
                raise RuntimeError("ST-GRAPE did not record a true-loss evaluation.")
            return result.exact_objective
        return result.objective


def selected_experiment_methods(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize the repeatable paper-experiment ``--method`` selection."""

    if not values:
        raise ValueError("At least one experiment method is required.")
    unknown = set(values).difference((*EXPERIMENT_METHODS, "all"))
    if unknown:
        choices = ", ".join((*EXPERIMENT_METHODS, "all"))
        raise ValueError(f"Unknown experiment method(s) {sorted(unknown)}; choose {choices}.")
    if "all" in values:
        return EXPERIMENT_METHODS
    requested = set(values)
    return tuple(method for method in EXPERIMENT_METHODS if method in requested)


def _baseline_label(representation: str) -> str:
    name = "Dense" if representation == "dense" else "Block-sparse"
    return f"{name} GRAPE"


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with a path-specific validation error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid experiment metadata from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def preflight_result_directories(directories: Sequence[Path]) -> None:
    """Reject duplicate or existing case directories before computation starts."""

    requested = tuple(directories)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested experiment cases contain duplicate output paths.")
    conflicts = [path for path in requested if path.exists()]
    if conflicts:
        paths = "\n".join(f"  - {path}" for path in conflicts)
        raise FileExistsError(
            "The following requested result directories already exist; no "
            f"optimization was started:\n{paths}\nSelect only unfinished cases "
            "or use another --output directory."
        )


def _configuration_mismatches(
    stored: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Reject missing, extra, or changed fixed settings in a current manifest."""

    return {
        key: (stored.get(key, "<missing>"), requested.get(key, "<missing>"))
        for key in stored.keys() | requested.keys()
        if key not in stored or key not in requested or stored[key] != requested[key]
    }


def prepare_resumable_output(
    output: Path,
    *,
    experiment: str,
    configuration: Mapping[str, Any],
) -> None:
    """Create or reopen a current-format run without upgrading historical files."""
    ensure_output_path(output)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / RUN_MANIFEST_NAME
    requested = json.loads(json.dumps(dict(configuration)))
    if path.is_file():
        stored = read_json_object(path)
        if stored.get("experiment") != experiment:
            raise ValueError(f"{output} belongs to a different experiment.")
        previous = stored.get("configuration")
        if not isinstance(previous, dict):
            raise ValueError(f"Missing configuration object in {path}.")
        differences = _configuration_mismatches(previous, requested)
        if differences:
            raise ValueError(f"Run configuration differs: {differences}")
    elif any(output.iterdir()):
        raise ValueError(f"Nonempty output has no current {RUN_MANIFEST_NAME}.")
    else:
        write_json(
            path, {"schema_version": 1, "experiment": experiment, "configuration": requested}
        )


def load_completed_seed_records(output: Path) -> list[dict[str, Any]]:
    """Load and validate all standard ``seed_summary.json`` case records."""

    if not output.exists():
        return []
    seed_directories = sorted(
        (
            path
            for path in output.rglob("seed*")
            if path.is_dir()
            and path.name.removeprefix("seed").isdigit()
            and (path.parent == output or path.parent.parent == output)
        ),
        key=lambda path: str(path),
    )
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for seed_directory in seed_directories:
        seed = int(seed_directory.name.removeprefix("seed"))
        summary_path = seed_directory / "seed_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Incomplete seed output detected at {seed_directory}; move or "
                "remove it before continuing this run."
            )
        record = read_json_object(summary_path)
        if record.get("seed") != seed:
            raise ValueError(f"Seed metadata in {summary_path} does not match its directory.")
        methods = record.get("methods")
        if not isinstance(methods, dict):
            raise ValueError(f"Missing method summaries in {summary_path}.")

        directory_qubits = -1
        if seed_directory.parent != output:
            qubit_label = seed_directory.parent.name
            if not qubit_label.endswith("q") or not qubit_label[:-1].isdigit():
                raise ValueError(f"Invalid qubit directory: {seed_directory.parent}")
            directory_qubits = int(qubit_label[:-1])
            if record.get("n_qubits", directory_qubits) != directory_qubits:
                raise ValueError(f"Qubit count in {summary_path} is inconsistent.")
            record.setdefault("n_qubits", directory_qubits)
        indexed.append((directory_qubits, seed, record))

    indexed.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in indexed]


def completed_seed_matrix(records: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Summarize completed seed labels by physical system size."""

    matrix: dict[str, list[int]] = {}
    for record in records:
        n_qubits = record.get("n_qubits")
        if n_qubits is None:
            continue
        matrix.setdefault(str(int(n_qubits)), []).append(int(record["seed"]))
    return {
        key: sorted(set(values)) for key, values in sorted(matrix.items(), key=lambda x: int(x[0]))
    }


def generated_seed_path(
    root: Path,
    *,
    task: ControlTask,
    n_qubits: int,
    seed: int,
) -> Path:
    """Return the stable layout written by ``exp00_nominal_seeds.py``."""

    return root / task.value / f"{n_qubits}q" / f"seed{seed}" / "result.npz"


def validate_generated_seed_root(root: Path) -> None:
    """Require an explicitly generated E00 result root before loading pulses."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"E00 manifest does not exist: {manifest_path}. "
            "Run stgrape experiment nominal and pass that run directory."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "E00 nominal seeds":
        raise ValueError(f"{root} is not an E00 nominal-seed result directory.")


def generated_seed_time_step(root: Path) -> float:
    """Return the common PWC time step recorded by an E00 result root."""

    validate_generated_seed_root(root)
    manifest_path = root / "manifest.json"
    manifest = read_json_object(manifest_path)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"E00 manifest has no configuration object: {manifest_path}")
    value: object = configuration.get("time_step_us")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"E00 manifest has no valid time_step_us: {manifest_path}")
    time_step = float(value)
    if not math.isfinite(time_step) or time_step <= 0:
        raise ValueError(f"E00 manifest has invalid time_step_us={value!r}: {manifest_path}")
    return time_step


def generated_seed_duration(
    root: Path,
    *,
    task: ControlTask,
    n_qubits: int,
    seed: int,
) -> float:
    """Return the pulse duration recorded by one current E00 seed."""

    path = generated_seed_path(
        root,
        task=task,
        n_qubits=n_qubits,
        seed=seed,
    )
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"E00 seed metadata does not exist: {metadata_path}")
    metadata = read_json_object(metadata_path)
    value = metadata.get("duration_us")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"E00 seed has no valid duration_us: {metadata_path}")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"E00 seed has invalid duration_us={value!r}: {metadata_path}")
    return duration


def generated_seed_coupling(root: Path) -> CouplingModel:
    """Return the nominal coupling recorded by an E00 result root.

    E00 manifests created before coupling selection was introduced are known
    to contain the original ZZ model and are interpreted accordingly.
    """

    validate_generated_seed_root(root)
    manifest_path = root / "manifest.json"
    manifest = read_json_object(manifest_path)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"E00 manifest has no configuration object: {manifest_path}")
    return normalize_coupling_model(str(configuration.get("coupling", "ZZ")))


def validate_generated_seed(
    path: Path,
    *,
    task: ControlTask,
    n_qubits: int,
    seed: int,
    error_bound: float = PAPER_NOMINAL_ERROR_BOUND,
    control_basis: ControlBasis = "pwc",
    chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
    duration: float | None = None,
    time_step: float | None = None,
    n_time_steps: int | None = None,
) -> None:
    """Reject mislabeled, legacy, or prematurely converged E00 seed inputs."""

    metadata_path = path.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"E00 seed metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "E00 nominal seeds",
        "task": task.value,
        "n_qubits": n_qubits,
        "seed": seed,
        "schema_version": 1,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"E00 seed metadata mismatch for {path}: {mismatches}")

    grid_mismatches: dict[str, tuple[object, object]] = {}
    for key, expected_value in (
        ("duration_us", duration),
        ("time_step_us", time_step),
    ):
        if expected_value is None:
            continue
        saved_value = metadata.get(key)
        try:
            matches = math.isclose(
                float(saved_value),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            grid_mismatches[key] = (saved_value, expected_value)
    if n_time_steps is not None and metadata.get("n_time_steps") != n_time_steps:
        grid_mismatches["n_time_steps"] = (metadata.get("n_time_steps"), n_time_steps)
    if grid_mismatches:
        raise ValueError(f"E00 seed time-grid mismatch for {path}: {grid_mismatches}")

    pulse_metadata = metadata.get("pulse", {})
    if not isinstance(pulse_metadata, Mapping):
        raise ValueError(f"E00 seed {path} has invalid pulse metadata.")
    # E00 results written before control-basis selection were all filtered PWC.
    saved_basis = str(pulse_metadata.get("control_basis", "pwc"))
    if saved_basis != control_basis:
        raise ValueError(
            f"E00 seed {path} uses control_basis={saved_basis!r}, but this run "
            f"requested {control_basis!r}. Generate and select a matching E00 root."
        )
    if control_basis == "chebyshev":
        saved_k = pulse_metadata.get("chebyshev_basis_size")
        if saved_k != chebyshev_k:
            raise ValueError(
                f"E00 seed {path} uses Chebyshev K={saved_k!r}, but this run "
                f"requested K={chebyshev_k}."
            )

    with np.load(path, allow_pickle=False) as data:
        if "coefficients" not in data.files or "objective" not in data.files:
            raise ValueError(
                f"{path} is not a current E00 result; legacy pulse inputs are forbidden."
            )
        objective = float(np.asarray(data["objective"]).item())
    if not math.isfinite(objective) or objective > error_bound:
        raise ValueError(
            f"E00 seed {path} has nominal loss {objective:.8g}, exceeding "
            f"the accepted bound {error_bound:g}. Regenerate or repair this seed first."
        )


def load_initial_pulse(
    problem: ControlProblem,
    *,
    n_qubits: int,
    seed: int,
    source: str,
    generated_root: Path | None = None,
    control_basis: ControlBasis = "pwc",
    chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
) -> tuple[Pulse, Path | None]:
    """Load a current E00 seed or create a deterministic random pulse.

    Random pulses are useful for reduced smoke runs with a coarser time grid;
    full paper runs should use a reviewed ``generated`` seed root.
    """

    if source == "random":
        return (
            PAPER_PULSE_CONFIG.random_pulse(
                problem,
                seed=seed,
                control_basis=control_basis,
                chebyshev_k=chebyshev_k,
            ),
            None,
        )
    if source == "generated":
        if generated_root is None:
            raise ValueError("--seed-root is required when --initial-source=generated.")
        validate_generated_seed_root(generated_root)
        path = generated_seed_path(
            generated_root,
            task=problem.task,
            n_qubits=n_qubits,
            seed=seed,
        )
        validate_generated_seed(
            path,
            task=problem.task,
            n_qubits=n_qubits,
            seed=seed,
            control_basis=control_basis,
            chebyshev_k=chebyshev_k,
            duration=problem.duration,
            time_step=problem.time_step,
            n_time_steps=problem.n_time_steps,
        )
    else:
        raise ValueError("initial source must be 'generated' or 'random'.")

    try:
        pulse = pulse_from_result(
            path,
            n_controls=problem.n_controls,
            n_time_steps=problem.n_time_steps,
            time_step=problem.time_step,
            control_basis=control_basis,
            chebyshev_k=chebyshev_k,
            chebyshev_normalization=PAPER_PULSE_CONFIG.chebyshev_normalization,
            chebyshev_amplitude_limit=PAPER_PULSE_CONFIG.chebyshev_amplitude_limit,
        )
    except ValueError as error:
        raise ValueError(
            f"Seed {path} cannot be reconstructed on the requested time grid "
            f"(duration={problem.duration:g} us, dt={problem.time_step:g} us, "
            f"steps={problem.n_time_steps}). Set --duration and --time-step to "
            "the values used by the selected E00 seed root, or regenerate "
            "matching E00 seeds. "
            f"Underlying pulse error: {error}"
        ) from error
    return pulse, path


def preflight_initial_pulses(
    cases: tuple[tuple[ControlProblem, int, int], ...],
    *,
    source: str,
    generated_root: Path | None,
    control_basis: ControlBasis = "pwc",
    chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
) -> None:
    """Validate every requested seed before a batch starts any optimization."""

    for problem, n_qubits, seed in cases:
        load_initial_pulse(
            problem,
            n_qubits=n_qubits,
            seed=seed,
            source=source,
            generated_root=generated_root,
            control_basis=control_basis,
            chebyshev_k=chebyshev_k,
        )
    source_description = (
        f"generated E00 root={generated_root}"
        if source == "generated"
        else "reproducible random controls"
    )
    print(
        f"Input preflight passed: {len(cases)} pulse(s), {source_description}, "
        f"control_basis={control_basis}"
        f"{f', K={chebyshev_k}' if control_basis == 'chebyshev' else ''}; "
        "current result schema enforced.",
        flush=True,
    )


def progress_reporter(label: str, interval: int) -> Callable[[int, float, float], None]:
    """Return a flushed optimizer callback that avoids excessive console output."""

    if interval < 1:
        raise ValueError("log interval must be positive.")

    def report(iteration: int, loss: float, elapsed: float) -> None:
        if iteration == 0 or iteration % interval == 0:
            print(
                f"  [{label}] iteration={iteration:6d}, loss={loss:.10e}, elapsed={elapsed:.2f} s",
                flush=True,
            )

    return report


def splitting_reporter(
    iteration: int,
    previous_steps: int,
    new_steps: int,
    gap_ratio: float,
) -> None:
    """Report an adaptive ST-GRAPE resolution change immediately."""

    print(
        f"  [ST splitting] iteration={iteration:6d}, "
        f"steps={previous_steps}->{new_steps}, "
        f"relative_gap={gap_ratio:.2%}; restarting L-BFGS-B",
        flush=True,
    )


def run_paper_method(
    *,
    method: str,
    chain: SpinChain,
    problem: ControlProblem,
    robustness: RobustnessModel,
    pulse: Pulse,
    max_iterations: int,
    ode_method: str,
    ode_rtol: float | None,
    ode_atol: float | None,
    exact_check_interval: int | None,
    exact_increase_patience: int,
    exact_increase_absolute_tolerance: float,
    exact_increase_relative_tolerance: float,
    time_limit_hours: float | None,
    log_interval: int,
    splitting_steps: int = 1,
    adaptive_splitting: bool = False,
    max_splitting_steps: int = 1,
    splitting_relative_tolerance: float = 0.15,
    splitting_absolute_tolerance: float = 2e-5,
    defer_final_exact_evaluation: bool = False,
    collect_sensitivity_diagnostics: bool = True,
) -> PaperMethodRun:
    """Run one selected comparison method from an immutable initial pulse."""

    selected = selected_experiment_methods((method,))
    if len(selected) != 1:
        raise ValueError("run_paper_method accepts one concrete method, not 'all'.")
    method = selected[0]
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if method == "ST" and defer_final_exact_evaluation:
        if exact_check_interval is not None:
            raise ValueError("Deferred ST reference evaluation requires no periodic checks.")
        if adaptive_splitting:
            raise ValueError("Adaptive splitting requires periodic reference checks.")

    optimizer = LBFGSBOptions(max_iterations=max_iterations)
    time_limit_seconds = None if time_limit_hours is None else 3600.0 * time_limit_hours
    # ST true-loss checks use the memory-efficient ODE representation and the
    # same requested integrator/tolerances as the block-sparse baseline.
    representation = "block-sparse" if method == "ST" else method
    propagation = configure_ode_propagation(
        method=ode_method,
        representation=representation,
        rtol=ode_rtol,
        atol=ode_atol,
    )
    initial_coefficients = pulse.coefficients.copy()
    algorithm = OptimizationAlgorithm.ST_GRAPE if method == "ST" else OptimizationAlgorithm.GRAPE
    initial_true_loss: float | None = None
    if method == "ST" and exact_check_interval is None and not defer_final_exact_evaluation:
        components = build_ode_components(problem, robustness, propagation)
        evaluated_initial_loss, _ = GrapeSolver._evaluate(
            components,
            problem,
            pulse.amplitudes(),
            compute_gradient=False,
        )
        initial_true_loss = float(evaluated_initial_loss)
    method_splitting_steps = splitting_steps if method == "ST" else 1
    method_adaptive_splitting = adaptive_splitting if method == "ST" else False
    method_max_splitting_steps = max_splitting_steps if method == "ST" else 1
    label = "ST-GRAPE" if method == "ST" else _baseline_label(method)
    print(f"  Running {label}...", flush=True)
    run = optimize(
        problem,
        pulse,
        robustness,
        OptimizationConfig(
            algorithm=algorithm,
            optimizer=optimizer,
            propagation=propagation,
            time_limit_seconds=time_limit_seconds,
            exact_check_interval=exact_check_interval,
            exact_increase_patience=exact_increase_patience,
            exact_increase_absolute_tolerance=exact_increase_absolute_tolerance,
            exact_increase_relative_tolerance=exact_increase_relative_tolerance,
            final_exact_evaluation=not (method == "ST" and defer_final_exact_evaluation),
            splitting_steps=method_splitting_steps,
            adaptive_splitting=method_adaptive_splitting,
            max_splitting_steps=method_max_splitting_steps,
            splitting_relative_tolerance=splitting_relative_tolerance,
            splitting_absolute_tolerance=splitting_absolute_tolerance,
        ),
        control_groups=chain.control_groups if method == "ST" else (),
        progress_callback=progress_reporter(label, log_interval),
        exact_progress_callback=(progress_reporter("ST true", 1) if method == "ST" else None),
        splitting_progress_callback=(
            splitting_reporter if method == "ST" and adaptive_splitting else None
        ),
    )
    np.testing.assert_array_equal(pulse.coefficients, initial_coefficients)
    final_nominal_fidelity: float | None = None
    final_robustness_penalty: float | None = None
    post_run_true_loss_wall_time_s: float | None = None
    if method == "ST" and defer_final_exact_evaluation:
        reference_start = time.perf_counter()
        components = build_ode_components(problem, robustness, propagation)
        exact_loss, _, fidelity_loss, penalty = GrapeSolver._evaluate_decomposed(
            components,
            problem,
            run.result.control_amplitudes,
            compute_gradient=False,
        )
        post_run_true_loss_wall_time_s = time.perf_counter() - reference_start
        exact_value = float(exact_loss)
        approximate_value = float(run.result.objective)
        absolute_error = abs(exact_value - approximate_value)
        splitting_tolerance = splitting_absolute_tolerance + splitting_relative_tolerance * abs(
            exact_value
        )
        final_nominal_fidelity = 1.0 - float(fidelity_loss)
        final_robustness_penalty = float(penalty)
        final_iteration = run.result.objective_history.size
        run = replace(
            run,
            result=replace(
                run.result,
                exact_objective=exact_value,
                exact_objective_iteration=final_iteration,
                exact_objective_history=np.asarray([exact_value]),
                exact_evaluation_iterations=np.asarray([final_iteration], dtype=np.int64),
                exact_evaluation_times=np.asarray(
                    [run.elapsed_seconds + post_run_true_loss_wall_time_s]
                ),
                exact_approximate_objective_history=np.asarray([approximate_value]),
                exact_approximation_gap_ratio_history=np.asarray(
                    [absolute_error / max(abs(exact_value), np.finfo(float).tiny)]
                ),
                exact_approximation_absolute_error_history=np.asarray([absolute_error]),
                exact_splitting_tolerance_history=np.asarray([splitting_tolerance]),
                exact_splitting_tolerance_exceeded_history=np.asarray(
                    [absolute_error > splitting_tolerance]
                ),
            ),
        )
    elif method == "ST" and initial_true_loss is None:
        if (
            run.result.exact_evaluation_iterations.size == 0
            or run.result.exact_evaluation_iterations[0] != 0
        ):
            raise RuntimeError("ST-GRAPE did not record the nominal pulse's initial true loss.")
        initial_true_loss = float(run.result.exact_objective_history[0])
    elif method != "ST":
        initial_true_loss = float(run.result.initial_objective)
        fidelity_losses = run.result.fidelity_loss_history
        penalties = run.result.robustness_penalty_history
        final_fidelity_loss = (
            float(fidelity_losses[-1]) if fidelity_losses.size else run.result.initial_fidelity_loss
        )
        final_robustness_penalty = (
            float(penalties[-1]) if penalties.size else run.result.initial_robustness_penalty
        )
        if final_fidelity_loss is not None:
            final_nominal_fidelity = 1.0 - final_fidelity_loss
    initial_sensitivity = (
        evaluate_taylor_sensitivity_diagnostics(
            problem, robustness, pulse.amplitudes(), propagation=propagation
        )
        if collect_sensitivity_diagnostics
        else None
    )
    final_sensitivity = (
        evaluate_taylor_sensitivity_diagnostics(
            problem, robustness, run.result.control_amplitudes, propagation=propagation
        )
        if collect_sensitivity_diagnostics
        else None
    )
    result = PaperMethodRun(
        method=method,
        optimization=run,
        initial_true_loss=initial_true_loss,
        initial_sensitivity=initial_sensitivity,
        final_sensitivity=final_sensitivity,
        final_nominal_fidelity=final_nominal_fidelity,
        final_robustness_penalty=final_robustness_penalty,
        post_run_true_loss_wall_time_s=post_run_true_loss_wall_time_s,
    )
    _ = result.true_loss
    return result


def sample_paper_controls(
    controls: Mapping[str, np.ndarray],
    problem: ControlProblem,
    robustness: RobustnessModel,
    *,
    n_samples: int,
    sampling_seed: int,
    workers: int,
    colored_noise_sampling: ColoredNoiseSamplingMethod | str = (ColoredNoiseSamplingMethod.AUTO),
) -> dict[str, np.ndarray]:
    """Sample named controls on deterministic, shared uncertainty draws."""

    if n_samples < 1:
        return {}
    return {
        key: sample_fidelities(
            problem,
            values,
            robustness,
            n_samples=n_samples,
            seed=sampling_seed,
            workers=workers,
            colored_noise_sampling=colored_noise_sampling,
        )
        for key, values in controls.items()
    }


def paper_method_summary(
    method_run: PaperMethodRun,
    fidelities: np.ndarray | None,
) -> dict[str, Any]:
    """Create the self-contained summary stored inside one method directory."""

    true_loss = method_run.true_loss
    return {
        "method": method_run.method,
        "initial_true_loss": method_run.initial_true_loss,
        "true_loss": true_loss,
        "true_objective": 1.0 - true_loss,
        "wall_time_s": method_run.optimization.elapsed_seconds,
        "final_nominal_fidelity": method_run.final_nominal_fidelity,
        "final_robustness_penalty": method_run.final_robustness_penalty,
        "post_run_true_loss_wall_time_s": method_run.post_run_true_loss_wall_time_s,
        "sensitivity_diagnostics": sensitivity_pair_metadata(
            method_run.initial_sensitivity,
            method_run.final_sensitivity,
        ),
        "sampling": None if fidelities is None else sampling_summary(fidelities),
    }


def save_paper_method(
    method_run: PaperMethodRun,
    directory: Path,
    *,
    metadata: Mapping[str, object],
    fidelities: np.ndarray | None,
) -> bool:
    """Atomically replace one method directory and return whether it existed."""

    directory.parent.mkdir(parents=True, exist_ok=True)
    existed = directory.exists()
    temporary = directory.with_name(f".{directory.name}.pending-{uuid4().hex}")
    backup = directory.with_name(f".{directory.name}.backup-{uuid4().hex}")
    try:
        run = method_run.optimization
        sensitivity_metadata = sensitivity_pair_metadata(
            method_run.initial_sensitivity,
            method_run.final_sensitivity,
        )
        run.result.save(
            temporary,
            metadata={
                **metadata,
                "experiment_method": method_run.method,
                "end_to_end_wall_time_s": run.elapsed_seconds,
                "optimization": run.config.metadata(),
                "sensitivity_diagnostics": sensitivity_metadata,
            },
        )
        initial_sensitivity = method_run.initial_sensitivity
        final_sensitivity = method_run.final_sensitivity
        np.savez_compressed(
            temporary / "sensitivity_diagnostics.npz",
            initial_p1=(np.nan if initial_sensitivity is None else initial_sensitivity.p1),
            initial_p1_squared=(
                np.nan if initial_sensitivity is None else initial_sensitivity.p1_squared
            ),
            initial_hybrid_p1_base=(
                np.nan if initial_sensitivity is None else initial_sensitivity.hybrid_p1_base
            ),
            initial_p2=(np.nan if initial_sensitivity is None else initial_sensitivity.p2),
            final_p1=(np.nan if final_sensitivity is None else final_sensitivity.p1),
            final_p1_squared=(
                np.nan if final_sensitivity is None else final_sensitivity.p1_squared
            ),
            final_hybrid_p1_base=(
                np.nan if final_sensitivity is None else final_sensitivity.hybrid_p1_base
            ),
            final_p2=(np.nan if final_sensitivity is None else final_sensitivity.p2),
        )
        if fidelities is not None:
            np.savez_compressed(temporary / "fidelities.npz", values=fidelities)
        write_json(
            temporary / "method_summary.json",
            paper_method_summary(method_run, fidelities),
        )
        if existed:
            _rename_directory_with_retry(directory, backup)
        _rename_directory_with_retry(temporary, directory)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not directory.exists():
            _rename_directory_with_retry(backup, directory)
        raise
    return existed


def save_case_inputs(
    directory: Path,
    *,
    pulse: Pulse,
    metadata: Mapping[str, object],
    initial_fidelities: np.ndarray | None,
) -> None:
    """Persist shared case inputs without touching existing method outputs."""

    directory.mkdir(parents=True, exist_ok=True)
    pulse_path = directory / "initial_pulse.npz"
    controls = pulse.amplitudes()
    if pulse_path.is_file():
        with np.load(pulse_path) as saved:
            np.testing.assert_allclose(saved["coefficients"], pulse.coefficients)
            np.testing.assert_allclose(saved["control_amplitudes"], controls)
    else:
        np.savez_compressed(
            pulse_path,
            coefficients=pulse.coefficients,
            control_amplitudes=controls,
        )
    write_json(directory / "case_metadata.json", dict(metadata))
    if initial_fidelities is not None:
        np.savez_compressed(
            directory / "initial_fidelities.npz",
            values=initial_fidelities,
        )


def warn_method_overwrites(directories: Sequence[Path]) -> tuple[Path, ...]:
    """Print all selected method outputs that a new run will replace."""

    conflicts = tuple(path for path in directories if path.exists())
    if conflicts:
        print("Warning: the following method results will be replaced:", flush=True)
        for path in conflicts:
            print(f"  - {path}", flush=True)
    return conflicts


def run_and_save_paper_methods(
    *,
    methods: Sequence[str],
    chain: SpinChain,
    problem: ControlProblem,
    robustness: RobustnessModel,
    pulse: Pulse,
    directory: Path,
    metadata: Mapping[str, object],
    max_iterations: int,
    ode_method: str,
    ode_rtol: float | None,
    ode_atol: float | None,
    exact_check_interval: int | None,
    exact_increase_patience: int,
    exact_increase_absolute_tolerance: float,
    exact_increase_relative_tolerance: float,
    time_limit_hours: float | None,
    log_interval: int,
    n_samples: int,
    sampling_seed: int,
    workers: int,
    splitting_steps: int = 1,
    adaptive_splitting: bool = False,
    max_splitting_steps: int = 1,
    splitting_relative_tolerance: float = 0.15,
    splitting_absolute_tolerance: float = 2e-5,
    defer_final_exact_evaluation: bool = False,
    collect_sensitivity_diagnostics: bool = True,
    method_execution_order: Sequence[str] | None = None,
    colored_noise_sampling: ColoredNoiseSamplingMethod | str = (ColoredNoiseSamplingMethod.AUTO),
) -> dict[str, PaperMethodRun]:
    """Run and atomically save any selected subset of paper comparison methods."""

    selected = selected_experiment_methods(methods)
    if method_execution_order is not None:
        requested_order = tuple(method_execution_order)
        if len(requested_order) != len(selected) or set(requested_order) != set(selected):
            raise ValueError("method_execution_order must contain each selected method once.")
        selected = requested_order
    method_directories = tuple(directory / method for method in selected)
    overwritten = set(warn_method_overwrites(method_directories))
    initial_samples = sample_paper_controls(
        {"initial": pulse.amplitudes()},
        problem,
        robustness,
        n_samples=n_samples,
        sampling_seed=sampling_seed,
        workers=workers,
        colored_noise_sampling=colored_noise_sampling,
    ).get("initial")
    save_case_inputs(
        directory,
        pulse=pulse,
        metadata=metadata,
        initial_fidelities=initial_samples,
    )

    completed: dict[str, PaperMethodRun] = {}
    for method in selected:
        method_run = run_paper_method(
            method=method,
            chain=chain,
            problem=problem,
            robustness=robustness,
            pulse=pulse,
            max_iterations=max_iterations,
            ode_method=ode_method,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            exact_check_interval=exact_check_interval,
            exact_increase_patience=exact_increase_patience,
            exact_increase_absolute_tolerance=exact_increase_absolute_tolerance,
            exact_increase_relative_tolerance=exact_increase_relative_tolerance,
            time_limit_hours=time_limit_hours,
            log_interval=log_interval,
            splitting_steps=splitting_steps,
            adaptive_splitting=adaptive_splitting,
            max_splitting_steps=max_splitting_steps,
            splitting_relative_tolerance=splitting_relative_tolerance,
            splitting_absolute_tolerance=splitting_absolute_tolerance,
            defer_final_exact_evaluation=defer_final_exact_evaluation,
            collect_sensitivity_diagnostics=collect_sensitivity_diagnostics,
        )
        sampled = sample_paper_controls(
            {method: method_run.optimization.result.control_amplitudes},
            problem,
            robustness,
            n_samples=n_samples,
            sampling_seed=sampling_seed,
            workers=workers,
            colored_noise_sampling=colored_noise_sampling,
        ).get(method)
        method_directory = directory / method
        save_paper_method(
            method_run,
            method_directory,
            metadata=metadata,
            fidelities=sampled,
        )
        action = "Replaced" if method_directory in overwritten else "Saved"
        print(f"  {action} {method} result at {method_directory}", flush=True)
        completed[method] = method_run
    return completed


def sampling_summary(values: np.ndarray) -> dict[str, float]:
    """Return the distribution statistics needed by the paper tables."""

    array = np.asarray(values, dtype=float)
    return {
        "mean_fidelity": float(np.mean(array)),
        "std_fidelity": float(np.std(array)),
        "minimum_fidelity": float(np.min(array)),
        "median_fidelity": float(np.median(array)),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable experiment metadata."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
