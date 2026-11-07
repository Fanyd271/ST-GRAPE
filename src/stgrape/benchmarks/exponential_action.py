"""Independent ``expm_multiply`` scaling benchmark for augmented GRAPE.

The benchmark reuses the physical systems, Taylor augmented operators,
terminal objective, costate, and approximate GRAPE gradient of the existing
block-sparse implementation.  Only propagation is replaced by SciPy's public
``LinearOperator``/``expm_multiply`` interface.  It collects raw data only;
plotting remains in ``notebooks/exp02_scaling.ipynb``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray

from stgrape.benchmarks.common import (
    BENCHMARK_RESET_PROTOCOL,
    ExperimentTask,
    augmented_block_count,
    block_structure,
    build_quasistatic_problem,
    random_controls,
    write_json,
)
from stgrape.paths import data_root
from stgrape.problem import ControlProblem, RobustnessModel
from stgrape.propagation.exponential_action import (
    OperatorCallCounts,
    build_exponential_action_components,
    evaluate_exponential_action,
)

RealArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

METHOD = "expm_multiply"
EXPERIMENT_NAME = "E04 scipy expm_multiply scaling"
DEFAULT_OUTPUT_ROOT = Path("outputs/exponential_action")
DEFAULT_REFERENCE_ROOTS = {
    ExperimentTask.GHZ: {
        "blocks": data_root() / "paper/scaling/ghz/blocks",
        "dimension": data_root() / "paper/scaling/ghz/dimension",
    },
    ExperimentTask.QFT: {
        "blocks": data_root() / "paper/scaling/qft/blocks",
        "dimension": data_root() / "paper/scaling/qft/dimension",
    },
}
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ActionMeasurement:
    """Repeated forward/full timings and diagnostics for one scaling point."""

    component_build_time_s: float
    forward_times_s: RealArray
    objective_gradient_times_s: RealArray
    forward_losses: RealArray
    objective_gradient_losses: RealArray
    forward_operator_calls: IntArray
    objective_gradient_operator_calls: IntArray
    forward_matvec_calls: IntArray
    forward_matmat_calls: IntArray
    forward_rmatvec_calls: IntArray
    forward_rmatmat_calls: IntArray
    objective_gradient_matvec_calls: IntArray
    objective_gradient_matmat_calls: IntArray
    objective_gradient_rmatvec_calls: IntArray
    objective_gradient_rmatmat_calls: IntArray
    persistent_operator_bytes: int
    peak_traced_allocation_bytes: int


def _reference_configuration(
    root: Path,
    *,
    expected_task: ExperimentTask,
) -> dict[str, Any]:
    """Read one existing E02 run manifest used to define formal defaults."""

    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Reference manifest does not exist: {manifest_path}. "
            "Run the corresponding E02 benchmark or pass another reference root."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Missing configuration object in {manifest_path}.")
    if configuration.get("task") != expected_task.value:
        raise ValueError(
            f"Reference task mismatch in {manifest_path}: expected "
            f"{expected_task.value!r}, found {configuration.get('task')!r}."
        )
    return configuration


def _runtime_metadata() -> dict[str, Any]:
    """Record software, CPU, BLAS, and thread settings relevant to timings."""

    blas: dict[str, Any]
    try:
        blas = np.__config__.show(mode="dicts")  # type: ignore[assignment,call-overload]
    except TypeError:  # NumPy versions without the structured ``mode`` argument.
        blas = {"description": "See NumPy build configuration for BLAS details."}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES},
        "timing_scope": (
            "Timings are machine-local and include online generator assembly, "
            "LinearOperator/adjoint preparation, SciPy action planning and norm "
            "estimation, propagation, objective, and approximate gradient."
        ),
    }


def _array_storage_bytes(value: object) -> int:
    """Count unique NumPy buffers in a component tree outside timed regions."""

    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def visit(candidate: object) -> int:
        identifier = id(candidate)
        if identifier in seen_objects:
            return 0
        seen_objects.add(identifier)
        if isinstance(candidate, np.ndarray):
            base = candidate
            while isinstance(base.base, np.ndarray):
                base = base.base
            buffer_id = id(base)
            if buffer_id in seen_arrays:
                return 0
            seen_arrays.add(buffer_id)
            return int(base.nbytes)
        if is_dataclass(candidate) and not isinstance(candidate, type):
            return sum(visit(getattr(candidate, field.name)) for field in fields(candidate))
        if isinstance(candidate, Mapping):
            return sum(visit(key) + visit(item) for key, item in candidate.items())
        if isinstance(candidate, (tuple, list, set, frozenset)):
            return sum(visit(item) for item in candidate)
        return 0

    return visit(value)


def _call_vector(counts: OperatorCallCounts) -> tuple[int, int, int, int, int]:
    return (
        counts.total,
        counts.matvec,
        counts.matmat,
        counts.rmatvec,
        counts.rmatmat,
    )


def benchmark_exponential_action(
    problem: ControlProblem,
    robustness: RobustnessModel,
    controls: RealArray,
    *,
    warmups: int,
    repeats: int,
) -> ActionMeasurement:
    """Benchmark the independent action backend using the E02 timing protocol."""

    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive.")
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.clear_traces()
    gc.collect()

    build_start = time.perf_counter()
    components = build_exponential_action_components(problem, robustness)
    component_build_time = time.perf_counter() - build_start

    def evaluate(gradient: bool) -> tuple[float, OperatorCallCounts]:
        loss, returned_gradient, calls, _ = evaluate_exponential_action(
            components,
            problem,
            controls,
            compute_gradient=gradient,
        )
        if gradient and returned_gradient is None:  # pragma: no cover - defensive.
            raise RuntimeError("The full action evaluation did not return a gradient.")
        return loss, calls

    for _ in range(warmups):
        evaluate(True)

    forward_times: list[float] = []
    full_times: list[float] = []
    forward_losses: list[float] = []
    full_losses: list[float] = []
    forward_calls: list[tuple[int, int, int, int, int]] = []
    full_calls: list[tuple[int, int, int, int, int]] = []
    for _ in range(repeats):
        start = time.perf_counter()
        loss, calls = evaluate(False)
        forward_times.append(time.perf_counter() - start)
        forward_losses.append(loss)
        forward_calls.append(_call_vector(calls))

        start = time.perf_counter()
        loss, calls = evaluate(True)
        full_times.append(time.perf_counter() - start)
        full_losses.append(loss)
        full_calls.append(_call_vector(calls))

    # Accuracy checks and allocation diagnostics are deliberately outside all
    # reported timing samples.
    tracemalloc.start()
    evaluate(True)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    forward_call_array = np.asarray(forward_calls, dtype=np.int64)
    full_call_array = np.asarray(full_calls, dtype=np.int64)
    return ActionMeasurement(
        component_build_time_s=component_build_time,
        forward_times_s=np.asarray(forward_times),
        objective_gradient_times_s=np.asarray(full_times),
        forward_losses=np.asarray(forward_losses),
        objective_gradient_losses=np.asarray(full_losses),
        forward_operator_calls=forward_call_array[:, 0],
        objective_gradient_operator_calls=full_call_array[:, 0],
        forward_matvec_calls=forward_call_array[:, 1],
        forward_matmat_calls=forward_call_array[:, 2],
        forward_rmatvec_calls=forward_call_array[:, 3],
        forward_rmatmat_calls=forward_call_array[:, 4],
        objective_gradient_matvec_calls=full_call_array[:, 1],
        objective_gradient_matmat_calls=full_call_array[:, 2],
        objective_gradient_rmatvec_calls=full_call_array[:, 3],
        objective_gradient_rmatmat_calls=full_call_array[:, 4],
        persistent_operator_bytes=_array_storage_bytes(components),
        peak_traced_allocation_bytes=peak_bytes,
    )


def _resolved(value: Any, reference: Mapping[str, Any], key: str) -> Any:
    return reference[key] if value is None else value


def _scan_settings(
    arguments: argparse.Namespace,
    scan: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Resolve one scan from its reference manifest and explicit overrides."""

    task = ExperimentTask(arguments.task)
    requested_reference = (
        arguments.block_reference_root if scan == "blocks" else arguments.dimension_reference_root
    )
    reference_root = requested_reference or DEFAULT_REFERENCE_ROOTS[task][scan]
    reference = _reference_configuration(reference_root, expected_task=task)
    if scan == "blocks":
        coordinates = tuple(
            arguments.source_counts
            if arguments.source_counts is not None
            else reference["source_counts"]
        )
        settings = {
            "n_qubits": int(_resolved(arguments.block_qubits, reference, "n_qubits")),
            "source_counts": coordinates,
            "order": int(_resolved(arguments.block_order, reference, "robustness_order")),
            "source_family": str(reference["source_family"]),
            "pauli_catalog_seed": int(reference["pauli_catalog_seed"]),
            "control_seed": int(_resolved(arguments.control_seed, reference, "control_seed")),
        }
    else:
        coordinates = tuple(
            arguments.qubits if arguments.qubits is not None else reference["qubits"]
        )
        settings = {
            "qubits": coordinates,
            "source_count": int(reference["source_count"]),
            "order": int(reference["robustness_order"]),
            "source_family": str(reference["source_family"]),
            "control_seed": int(_resolved(arguments.control_seed, reference, "control_seed")),
        }
    settings.update(
        {
            "task": task,
            "coupling": str(_resolved(arguments.coupling, reference, "coupling")),
            "time_step": float(_resolved(arguments.time_step, reference, "time_step_us")),
            "n_time_steps": int(_resolved(arguments.time_steps, reference, "n_time_steps")),
            "warmups": int(_resolved(arguments.warmups, reference, "warmups")),
            "repeats": int(_resolved(arguments.repeats, reference, "repeats")),
            "reference_root": str(reference_root),
        }
    )
    if arguments.smoke:
        coordinates = (1, 2) if scan == "blocks" else (2, 3)
        settings.update(
            {
                "n_qubits": 2,
                "source_counts": coordinates,
                "qubits": coordinates,
                "time_step": 0.0025,
                "n_time_steps": 2,
                "warmups": 0,
                "repeats": 1,
            }
        )
    return settings, coordinates


def _empty_results(n_points: int, repeats: int) -> dict[str, np.ndarray]:
    """Allocate explicit pending/NaN storage; failed points are never zeros."""

    repeated_float = (n_points, 1, repeats)
    repeated_int = (n_points, 1, repeats)
    return {
        "status": np.full(n_points, "pending", dtype="<U16"),
        "error_messages": np.full(n_points, "", dtype="<U1024"),
        "system_build_times_s": np.full(n_points, np.nan),
        "build_times_s": np.full((n_points, 1), np.nan),
        "forward_times_s": np.full(repeated_float, np.nan),
        "objective_gradient_times_s": np.full(repeated_float, np.nan),
        "forward_losses": np.full(repeated_float, np.nan),
        "objective_gradient_losses": np.full(repeated_float, np.nan),
        "forward_operator_calls": np.full(repeated_int, -1, dtype=np.int64),
        "objective_gradient_operator_calls": np.full(repeated_int, -1, dtype=np.int64),
        "forward_matvec_calls": np.full(repeated_int, -1, dtype=np.int64),
        "forward_matmat_calls": np.full(repeated_int, -1, dtype=np.int64),
        "forward_rmatvec_calls": np.full(repeated_int, -1, dtype=np.int64),
        "forward_rmatmat_calls": np.full(repeated_int, -1, dtype=np.int64),
        "objective_gradient_matvec_calls": np.full(repeated_int, -1, dtype=np.int64),
        "objective_gradient_matmat_calls": np.full(repeated_int, -1, dtype=np.int64),
        "objective_gradient_rmatvec_calls": np.full(repeated_int, -1, dtype=np.int64),
        "objective_gradient_rmatmat_calls": np.full(repeated_int, -1, dtype=np.int64),
        "persistent_operator_bytes": np.full((n_points, 1), -1, dtype=np.int64),
        "peak_traced_allocation_bytes": np.full((n_points, 1), -1, dtype=np.int64),
        "nonzero_operator_blocks": np.full(n_points, -1, dtype=np.int64),
    }


def _load_or_initialize(
    raw_path: Path,
    *,
    coordinate_name: str,
    coordinates: IntArray,
    repeats: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    arrays = _empty_results(len(coordinates), repeats)
    if overwrite or not raw_path.is_file():
        return arrays
    with np.load(raw_path, allow_pickle=False) as saved:
        stored_coordinates = np.asarray(saved[coordinate_name], dtype=np.int64)
        if not np.array_equal(stored_coordinates, coordinates):
            raise ValueError(f"Existing coordinates in {raw_path} differ from the requested scan.")
        for name in arrays:
            if name in saved:
                candidate = np.asarray(saved[name])
                if candidate.shape != arrays[name].shape:
                    raise ValueError(f"Existing array {name!r} has incompatible shape.")
                arrays[name] = candidate.copy()
    return arrays


def _save_raw(
    raw_path: Path,
    *,
    shared: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".tmp.npz")
    payload: dict[str, Any] = {"method": METHOD, **shared, **arrays}
    np.savez_compressed(temporary, **payload)
    temporary.replace(raw_path)


def _store_measurement(
    arrays: dict[str, np.ndarray],
    index: int,
    measurement: ActionMeasurement,
) -> None:
    arrays["build_times_s"][index, 0] = measurement.component_build_time_s
    for name in (
        "forward_times_s",
        "objective_gradient_times_s",
        "forward_losses",
        "objective_gradient_losses",
        "forward_operator_calls",
        "objective_gradient_operator_calls",
        "forward_matvec_calls",
        "forward_matmat_calls",
        "forward_rmatvec_calls",
        "forward_rmatmat_calls",
        "objective_gradient_matvec_calls",
        "objective_gradient_matmat_calls",
        "objective_gradient_rmatvec_calls",
        "objective_gradient_rmatmat_calls",
    ):
        arrays[name][index, 0] = getattr(measurement, name)
    arrays["persistent_operator_bytes"][index, 0] = measurement.persistent_operator_bytes
    arrays["peak_traced_allocation_bytes"][index, 0] = measurement.peak_traced_allocation_bytes


def _run_scan(arguments: argparse.Namespace, scan: str) -> Path:
    settings, coordinates_tuple = _scan_settings(arguments, scan)
    task = settings["task"]
    coordinates = np.asarray(coordinates_tuple, dtype=np.int64)
    if len(coordinates) == 0 or len(np.unique(coordinates)) != len(coordinates):
        raise ValueError(f"{scan} coordinates must be non-empty and unique.")
    if np.any(coordinates < (1 if scan == "blocks" else 2)):
        raise ValueError(f"Invalid {scan} coordinate.")
    if settings["warmups"] < 0 or settings["repeats"] < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive.")

    output_root = arguments.output or DEFAULT_OUTPUT_ROOT / task.value
    scan_root = output_root / scan
    manifest_configuration = {
        "task": task.value,
        "method": METHOD,
        "scan": scan,
        "coupling": settings["coupling"],
        "time_step_us": settings["time_step"],
        "n_time_steps": settings["n_time_steps"],
        "warmups": settings["warmups"],
        "repeats": settings["repeats"],
        "pre_measurement_cleanup": BENCHMARK_RESET_PROTOCOL,
        "reference_root": settings["reference_root"],
        "smoke": arguments.smoke,
    }
    if scan == "blocks":
        manifest_configuration.update(
            {
                "n_qubits": settings["n_qubits"],
                "source_counts": coordinates.tolist(),
                "robustness_order": settings["order"],
                "source_family": settings["source_family"],
                "pauli_catalog_seed": settings["pauli_catalog_seed"],
                "control_seed": settings["control_seed"],
            }
        )
    else:
        manifest_configuration.update(
            {
                "qubits": coordinates.tolist(),
                "source_count": settings["source_count"],
                "robustness_order": settings["order"],
                "source_family": settings["source_family"],
                "control_seed_rule": f"{settings['control_seed']} + n_qubits",
            }
        )
    scan_root.mkdir(parents=True, exist_ok=True)
    manifest_path = scan_root / "run_manifest.json"
    current_runtime = _runtime_metadata()
    write_manifest = True
    if manifest_path.is_file() and arguments.existing_result == "skip":
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("configuration") != manifest_configuration:
            raise ValueError(
                f"{scan_root} contains an incompatible run manifest. Use a new --output."
            )
        if stored.get("runtime") != current_runtime:
            raise ValueError(
                f"Runtime/BLAS/thread metadata changed for {scan_root}. Resume in the "
                "original environment or use a new --output directory; cross-machine "
                "timings cannot be merged."
            )
        write_manifest = False
    elif not manifest_path.is_file() and any(scan_root.iterdir()):
        raise ValueError(f"Refusing non-empty output without a run manifest: {scan_root}")
    if write_manifest:
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "experiment": EXPERIMENT_NAME,
                "configuration": manifest_configuration,
                "runtime": current_runtime,
            },
        )

    raw_path = scan_root / METHOD / "raw_data.npz"
    coordinate_name = "source_counts" if scan == "blocks" else "n_qubits"
    arrays = _load_or_initialize(
        raw_path,
        coordinate_name=coordinate_name,
        coordinates=coordinates,
        repeats=settings["repeats"],
        overwrite=arguments.existing_result == "overwrite",
    )
    if scan == "blocks":
        block_counts = np.asarray(
            [augmented_block_count(int(m), settings["order"]) for m in coordinates],
            dtype=np.int64,
        )
        shared = {
            "source_counts": coordinates,
            "block_counts": block_counts,
            "n_qubits": settings["n_qubits"],
            "hilbert_dimension": 2 ** settings["n_qubits"],
            "robustness_order": settings["order"],
        }
    else:
        shared = {
            "n_qubits": coordinates,
            "hilbert_dimensions": 2**coordinates,
            "source_count": settings["source_count"],
            "robustness_order": settings["order"],
            "block_count": augmented_block_count(settings["source_count"], settings["order"]),
        }
    shared.update(
        {
            "task": task.value,
            "coupling": settings["coupling"],
            "time_step_us": settings["time_step"],
            "n_time_steps": settings["n_time_steps"],
            "source_family": settings["source_family"],
            "repeats": settings["repeats"],
            "warmups": settings["warmups"],
            "pre_measurement_cleanup": BENCHMARK_RESET_PROTOCOL,
        }
    )

    for index, coordinate in enumerate(coordinates):
        if arguments.existing_result == "skip" and arrays["status"][index] == "ok":
            print(f"E04 {scan}: skipping completed point {coordinate}", flush=True)
            continue
        try:
            gc.collect()
            system_start = time.perf_counter()
            if scan == "blocks":
                source_count = int(coordinate)
                n_qubits = settings["n_qubits"]
                control_seed = settings["control_seed"]
            else:
                source_count = settings["source_count"]
                n_qubits = int(coordinate)
                control_seed = settings["control_seed"] + n_qubits
            chain, problem, robustness = build_quasistatic_problem(
                task,
                n_qubits,
                time_step=settings["time_step"],
                n_time_steps=settings["n_time_steps"],
                source_count=source_count,
                order=settings["order"],
                source_family=settings["source_family"],
                pauli_catalog_seed=settings.get("pauli_catalog_seed", 42),
                coupling=settings["coupling"],
            )
            arrays["system_build_times_s"][index] = time.perf_counter() - system_start
            actual_blocks, stored_blocks = block_structure(problem, robustness)
            arrays["nonzero_operator_blocks"][index] = stored_blocks
            controls = random_controls(problem, seed=control_seed)
            print(
                f"E04 {task.value} {scan}: n={n_qubits}, d={2**n_qubits}, "
                f"m={source_count}, "
                f"B={actual_blocks}",
                flush=True,
            )
            measurement = benchmark_exponential_action(
                problem,
                robustness,
                controls,
                warmups=settings["warmups"],
                repeats=settings["repeats"],
            )
            _store_measurement(arrays, index, measurement)
            arrays["status"][index] = "ok"
            arrays["error_messages"][index] = ""
            print(
                f"  median full time: {np.median(measurement.objective_gradient_times_s):.6g} s",
                flush=True,
            )
        except Exception as error:  # Save explicit point failure, then continue.
            arrays["status"][index] = "failed"
            arrays["error_messages"][index] = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {arrays['error_messages'][index]}", flush=True)
        finally:
            _save_raw(raw_path, shared=shared, arrays=arrays)

    print(f"Saved E04 {scan} raw data to {raw_path}", flush=True)
    return scan_root


def run_experiment(arguments: argparse.Namespace) -> tuple[Path, ...]:
    """Run one or both independent scaling scans."""

    scans = ("blocks", "dimension") if arguments.scan == "both" else (arguments.scan,)
    return tuple(_run_scan(arguments, scan) for scan in scans)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reference-manifest-driven E04 configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", choices=("blocks", "dimension", "both"), default="both")
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in ExperimentTask),
        default=ExperimentTask.GHZ.value,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--existing-result", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--block-reference-root", type=Path)
    parser.add_argument("--dimension-reference-root", type=Path)
    parser.add_argument("--source-counts", nargs="+", type=int)
    parser.add_argument("--qubits", nargs="+", type=int)
    parser.add_argument("--block-qubits", type=int)
    parser.add_argument("--block-order", type=int)
    parser.add_argument("--coupling", choices=("ZZ", "XX+YY"))
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--control-seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run terminal arguments or the editable one-click setup."""

    run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
