"""Shared builders and raw-data utilities for the revised experiments.

This module deliberately contains no plotting or statistical aggregation.
Experiment entry points save measurements and reproducibility metadata; their
companion notebooks perform all analysis and visualization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import qutip
import scipy
from numpy.typing import NDArray
from qutip import Qobj

from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.models import (
    build_multisource_qft_robustness,
    build_multisource_spin_chain_robustness,
    build_paper_problem,
)
from stgrape.paths import ensure_output_path
from stgrape.problem import ControlProblem, ControlTask, RobustnessModel, TaylorNoise
from stgrape.propagation import (
    OdeRepresentation,
    build_ode_components,
    build_trotter_components,
)
from stgrape.propagation.ode import OdeComponent
from stgrape.propagation.trotter import TrotterComponent
from stgrape.systems import SpinChain, normalize_coupling_model
from stgrape.workflow import configure_ode_propagation

RealArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
OUTPUT_ROOT = Path("outputs/experiments")
COMPARISON_METHODS = ("ST", "dense", "block-sparse")
BENCHMARK_RESET_PROTOCOL = "gc.collect+tracemalloc.reset"
HYBRID_P1_CONVENTION = "lambda-times-p1-squared"
ExistingResultAction = Literal["skip", "overwrite"]
AUTO_NOISE_QUBIT = "auto"
NoiseQubitSelection = int | None | Literal["auto"]
NoisePairSelection = tuple[int, int] | None | Literal["auto"]


class ExperimentTask(StrEnum):
    """Physical targets supported by every revised experiment."""

    GHZ = "ghz"
    QFT = "qft"

    @property
    def control_task(self) -> ControlTask:
        """Return the corresponding package-level terminal objective."""

        if self == ExperimentTask.GHZ:
            return ControlTask.STATE_PREPARATION
        return ControlTask.GATE_SYNTHESIS

    @property
    def default_qubits(self) -> int:
        """Return a practical default size for end-to-end optimization."""

        return 4 if self == ExperimentTask.GHZ else 3


@dataclass(frozen=True)
class NoiseLocations:
    """One-based public labels for heterogeneous uncertainty locations."""

    static_coupling_pairs: tuple[tuple[int, int], ...]
    frequency_drift_qubits: tuple[int, ...]
    control_drift_qubits: tuple[int, ...]
    non_markovian_qubit: int | None


@dataclass(frozen=True)
class EvaluationMeasurement:
    """Raw repeated measurements for one propagation method."""

    build_time_s: float
    forward_times_s: RealArray
    objective_gradient_times_s: RealArray
    forward_losses: RealArray
    objective_gradient_losses: RealArray
    forward_rhs_evaluations: IntArray
    objective_gradient_rhs_evaluations: IntArray
    persistent_operator_bytes: int
    peak_traced_allocation_bytes: int


def unique_output_directory(experiment: str, task: ExperimentTask | str) -> Path:
    """Return a collision-resistant revised-experiment result directory."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return OUTPUT_ROOT / experiment / ExperimentTask(task).value / timestamp


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON after creating the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")


def runtime_metadata() -> dict[str, object]:
    """Return software and machine information needed to interpret timings."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
    }


def prepare_raw_output(
    output: Path,
    *,
    experiment: str,
    configuration: Mapping[str, Any],
) -> None:
    """Create a raw benchmark directory and validate an existing manifest."""

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output is not a directory: {output}")
    ensure_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    requested = dict(configuration)
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("experiment") != experiment:
            raise ValueError(f"{output} belongs to a different experiment.")
        stored_configuration = stored.get("configuration")
        if not isinstance(stored_configuration, dict):
            raise ValueError(f"Missing configuration object in {manifest_path}.")
        if stored_configuration != requested:
            keys = sorted(stored_configuration.keys() | requested.keys())
            mismatches = [
                (key, stored_configuration.get(key), requested.get(key))
                for key in keys
                if stored_configuration.get(key) != requested.get(key)
            ]
            details = "\n".join(
                f"  - {key}: stored={stored_value!r}, requested={requested_value!r}"
                for key, stored_value, requested_value in mismatches
            )
            raise ValueError(
                f"{output} contains an incompatible experiment configuration:\n"
                f"{details}\nUse another --output directory or restore the original settings."
            )
    elif any(output.iterdir()):
        raise ValueError(
            f"Refusing non-empty output without a compatible run_manifest.json: {output}"
        )
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment": experiment,
            "configuration": requested,
            "runtime": runtime_metadata(),
        },
    )


def select_pending_methods(
    output: Path,
    methods: Sequence[str],
    *,
    existing_result: ExistingResultAction,
    required_files: Sequence[str],
) -> tuple[str, ...]:
    """Return methods that should run under a skip/overwrite output policy.

    Under ``skip``, a method directory counts as completed only when every
    required file exists.  An incomplete directory is reported instead of
    being silently replaced.  ``overwrite`` leaves replacement and warning
    handling to the method-specific atomic saver.
    """

    if existing_result not in {"skip", "overwrite"}:
        raise ValueError("existing_result must be 'skip' or 'overwrite'.")
    selected = tuple(methods)
    if existing_result == "overwrite":
        return selected

    pending: list[str] = []
    skipped: list[Path] = []
    for method in selected:
        method_directory = output / method
        if not method_directory.exists():
            pending.append(method)
            continue
        if not method_directory.is_dir():
            raise NotADirectoryError(f"Method output is not a directory: {method_directory}")
        missing = [name for name in required_files if not (method_directory / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Incomplete existing result at {method_directory}; missing {missing}. "
                "Use --existing-result overwrite to replace it."
            )
        skipped.append(method_directory)

    if skipped:
        print("Skipping existing completed method results:", flush=True)
        for directory in skipped:
            print(f"  - {directory}", flush=True)
    return tuple(pending)


def should_run_raw_result(
    path: Path,
    *,
    existing_result: ExistingResultAction,
) -> bool:
    """Return whether one file-level raw result should be generated."""

    if existing_result not in {"skip", "overwrite"}:
        raise ValueError("existing_result must be 'skip' or 'overwrite'.")
    if not path.exists():
        return True
    if not path.is_file():
        raise ValueError(f"Raw result path is not a file: {path}")
    if existing_result == "skip":
        print(f"Skipping existing raw result at {path}", flush=True)
        return False
    print(f"Warning: replacing existing raw result at {path}", flush=True)
    return True


def selected_methods(values: Sequence[str], *, allow_dense: bool = True) -> tuple[str, ...]:
    """Normalize a multi-value method selector in stable display order."""

    if not values:
        raise ValueError("At least one method is required.")
    allowed = COMPARISON_METHODS if allow_dense else ("ST", "block-sparse")
    unknown = set(values).difference((*allowed, "all"))
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}")
    if "all" in values:
        return allowed
    requested = set(values)
    return tuple(method for method in allowed if method in requested)


def parse_optional_noise_qubit(value: str) -> NoiseQubitSelection:
    """Parse a one-based qubit label, ``auto``, or the disabling value ``None``."""

    normalized = value.strip().casefold()
    if normalized == "auto":
        return AUTO_NOISE_QUBIT
    if normalized in {"none", "off", "null"}:
        return None
    try:
        qubit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a positive one-based qubit label, 'auto', or 'None'"
        ) from error
    if qubit < 1:
        raise argparse.ArgumentTypeError("the non-Markovian qubit must be positive")
    return qubit


def parse_optional_noise_pair(value: str) -> NoisePairSelection:
    """Parse ``LEFT-RIGHT``, ``auto``, or the disabling value ``None``."""

    normalized = value.strip().casefold()
    if normalized == "auto":
        return AUTO_NOISE_QUBIT
    if normalized in {"none", "off", "null"}:
        return None
    labels = value.split("-")
    if len(labels) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid qubit pair {value!r}; expected '2-3', 'auto', or 'None'"
        )
    try:
        pair = int(labels[0]), int(labels[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid qubit pair {value!r}; both labels must be integers"
        ) from error
    if min(pair) < 1:
        raise argparse.ArgumentTypeError("qubit-pair labels must be positive")
    return pair


def resolve_optional_noise_qubit(
    value: NoiseQubitSelection,
) -> tuple[int | None, bool]:
    """Return the requested site and whether non-Markovian noise is enabled."""

    if value == AUTO_NOISE_QUBIT:
        return None, True
    return value, value is not None


def resolve_optional_noise_qubits(
    values: Sequence[NoiseQubitSelection] | Literal["auto"],
) -> tuple[tuple[int, ...] | None, bool]:
    """Resolve a CLI sequence into default, disabled, or explicit qubits."""

    if values == AUTO_NOISE_QUBIT:
        return None, True
    selected = tuple(values)
    if selected == (AUTO_NOISE_QUBIT,):
        return None, True
    if selected == (None,):
        return None, False
    if any(value is None or value == AUTO_NOISE_QUBIT for value in selected):
        raise ValueError("'None' or 'auto' must be the only qubit-list value.")
    if not all(isinstance(value, int) for value in selected):
        raise TypeError("Explicit control-drift locations must be integer qubits.")
    return tuple(value for value in selected if isinstance(value, int)), True


def resolve_optional_noise_pairs(
    values: Sequence[NoisePairSelection] | Literal["auto"],
) -> tuple[tuple[tuple[int, int], ...] | None, bool]:
    """Resolve a CLI sequence into default, disabled, or explicit pairs."""

    if values == AUTO_NOISE_QUBIT:
        return None, True
    selected = tuple(values)
    if selected == (AUTO_NOISE_QUBIT,):
        return None, True
    if selected == (None,):
        return None, False
    if any(value is None or value == AUTO_NOISE_QUBIT for value in selected):
        raise ValueError("'None' or 'auto' must be the only qubit-pair value.")
    if not all(isinstance(pair, tuple) for pair in selected):
        raise TypeError("Explicit static locations must be integer qubit pairs.")
    return tuple(pair for pair in selected if isinstance(pair, tuple)), True


def normalize_active_noise_weights(
    coupling_weight: float,
    frequency_drift_weight: float,
    control_drift_weight: float,
    colored_weight: float,
    *,
    include_static_noise: bool = True,
    include_frequency_drift: bool = True,
    include_control_drift: bool = True,
    include_non_markovian_noise: bool,
) -> tuple[float, float, float, float]:
    """Normalize enabled noise-family weights while preserving their ratios."""

    weights = np.asarray(
        [
            coupling_weight,
            frequency_drift_weight,
            control_drift_weight,
            colored_weight,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Noise-family weights must be finite and non-negative.")
    active = np.asarray(
        [
            include_static_noise,
            include_frequency_drift,
            include_control_drift,
            include_non_markovian_noise,
        ],
        dtype=bool,
    )
    if not np.any(active):
        raise ValueError("At least one physical noise family must be enabled.")
    active_total = float(np.sum(weights[active]))
    if active_total <= 0.0:
        raise ValueError("At least one enabled noise-family weight must be positive.")
    normalized = np.zeros(4, dtype=float)
    normalized[active] = weights[active] / active_total
    return (
        float(normalized[0]),
        float(normalized[1]),
        float(normalized[2]),
        float(normalized[3]),
    )


def default_noise_locations(n_qubits: int) -> NoiseLocations:
    """Choose scalable heterogeneous-noise sites using one-based labels."""

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two.")
    static_pairs = tuple((qubit, qubit + 1) for qubit in range(1, n_qubits))
    frequency_qubits = tuple(range(1, n_qubits + 1))
    if n_qubits == 2:
        return NoiseLocations(static_pairs, frequency_qubits, (1,), 2)
    center = max(2, n_qubits // 2)
    drifts = tuple(dict.fromkeys((center, min(center + 1, n_qubits))))
    return NoiseLocations(static_pairs, frequency_qubits, drifts, n_qubits)


def validate_noise_locations(
    n_qubits: int,
    static_coupling_pairs: Sequence[Sequence[int]] | None,
    frequency_drift_qubits: Sequence[int] | None,
    control_drift_qubits: Sequence[int] | None,
    non_markovian_qubit: int | None,
    *,
    include_static_noise: bool = True,
    include_frequency_drift: bool = True,
    include_control_drift: bool = True,
    include_non_markovian_noise: bool = True,
) -> NoiseLocations:
    """Validate or infer public one-based heterogeneous-noise locations."""

    defaults = default_noise_locations(n_qubits)
    requested_pairs = (
        (
            defaults.static_coupling_pairs
            if static_coupling_pairs is None
            else tuple(tuple(pair) for pair in static_coupling_pairs)
        )
        if include_static_noise
        else ()
    )
    if include_static_noise and (
        not requested_pairs or any(len(pair) != 2 for pair in requested_pairs)
    ):
        raise ValueError("static coupling pairs must be a non-empty sequence of qubit pairs.")
    static_pairs = tuple((min(left, right), max(left, right)) for left, right in requested_pairs)
    if len(set(static_pairs)) != len(static_pairs):
        raise ValueError("static coupling pairs cannot contain duplicate links.")
    if any(left < 1 or right > n_qubits or right != left + 1 for left, right in static_pairs):
        raise ValueError("static coupling pairs must connect valid adjacent one-based qubits.")
    frequency_drifts = (
        (
            defaults.frequency_drift_qubits
            if frequency_drift_qubits is None
            else tuple(frequency_drift_qubits)
        )
        if include_frequency_drift
        else ()
    )
    drifts = (
        (
            defaults.control_drift_qubits
            if control_drift_qubits is None
            else tuple(control_drift_qubits)
        )
        if include_control_drift
        else ()
    )
    non_markovian = (
        (defaults.non_markovian_qubit if non_markovian_qubit is None else non_markovian_qubit)
        if include_non_markovian_noise
        else None
    )
    if include_control_drift and (not drifts or len(set(drifts)) != len(drifts)):
        raise ValueError("control drift qubits must be a non-empty unique sequence.")
    if any(qubit < 1 or qubit > n_qubits for qubit in drifts):
        raise ValueError("control drift qubits must use valid one-based labels.")
    if include_frequency_drift and (
        not frequency_drifts or len(set(frequency_drifts)) != len(frequency_drifts)
    ):
        raise ValueError("frequency drift qubits must be a non-empty unique sequence.")
    if any(qubit < 1 or qubit > n_qubits for qubit in frequency_drifts):
        raise ValueError("frequency drift qubits must use valid one-based labels.")
    if non_markovian is not None and (non_markovian < 1 or non_markovian > n_qubits):
        raise ValueError("non-Markovian noise qubit must use a valid one-based label.")
    return NoiseLocations(static_pairs, frequency_drifts, drifts, non_markovian)


def build_multisource_problem(
    task: ExperimentTask | str,
    n_qubits: int,
    *,
    time_step: float,
    duration: float | None = None,
    order: int,
    coupling: str = "XX+YY",
    static_standard_deviation_mhz: float = 1.0,
    frequency_drift_standard_deviation_mhz: float = 1.0,
    colored_standard_deviation_mhz: float = 1.0,
    control_drift_standard_deviation: float = 0.01,
    first_order_penalty: float = 1.0,
    second_order_penalty: float = 1.0,
    penalty_mode: str = "squared",
    rms_smoothing: float = 1e-12,
    p1_smoothing: float | None = None,
    p2_smoothing: float | None = None,
    pc_smoothing: float | None = None,
    p1_coefficient: float = 1.0,
    p2_coefficient: float = 1.0,
    pc_coefficient: float = 1.0,
    coupling_weight: float = 1.0 / 4.0,
    frequency_drift_weight: float = 1.0 / 4.0,
    control_drift_weight: float = 1.0 / 4.0,
    colored_weight: float = 1.0 / 4.0,
    static_coupling_pairs: Sequence[Sequence[int]] | None = None,
    frequency_drift_qubits: Sequence[int] | None = None,
    control_drift_qubits: Sequence[int] | None = None,
    non_markovian_qubit: int | None = None,
    include_static_noise: bool = True,
    include_frequency_drift: bool = True,
    include_control_drift: bool = True,
    include_non_markovian_noise: bool = True,
) -> tuple[SpinChain, ControlProblem, RobustnessModel, NoiseLocations]:
    """Build one GHZ/QFT problem with configurable heterogeneous noise."""

    selected_task = ExperimentTask(task)
    chain, problem = build_paper_problem(
        n_qubits,
        selected_task.control_task,
        time_step=time_step,
        duration=duration,
        coupling=coupling,
    )
    locations = validate_noise_locations(
        n_qubits,
        static_coupling_pairs,
        frequency_drift_qubits,
        control_drift_qubits,
        non_markovian_qubit,
        include_static_noise=include_static_noise,
        include_frequency_drift=include_frequency_drift,
        include_control_drift=include_control_drift,
        include_non_markovian_noise=include_non_markovian_noise,
    )
    zero_based_static_pairs = tuple(
        (left - 1, right - 1) for left, right in locations.static_coupling_pairs
    )
    zero_based_drifts = tuple(qubit - 1 for qubit in locations.control_drift_qubits)
    zero_based_frequency_drifts = tuple(qubit - 1 for qubit in locations.frequency_drift_qubits)
    zero_based_colored = (
        None if locations.non_markovian_qubit is None else locations.non_markovian_qubit - 1
    )
    builder = (
        build_multisource_spin_chain_robustness
        if selected_task == ExperimentTask.GHZ
        else build_multisource_qft_robustness
    )
    robustness = builder(
        chain,
        order=order,
        static_standard_deviation_mhz=static_standard_deviation_mhz,
        frequency_drift_standard_deviation_mhz=(frequency_drift_standard_deviation_mhz),
        control_drift_standard_deviation=control_drift_standard_deviation,
        colored_standard_deviation_mhz=colored_standard_deviation_mhz,
        first_order_penalty=first_order_penalty,
        second_order_penalty=second_order_penalty,
        penalty_mode=penalty_mode,
        rms_smoothing=rms_smoothing,
        p1_smoothing=p1_smoothing,
        p2_smoothing=p2_smoothing,
        pc_smoothing=pc_smoothing,
        p1_coefficient=p1_coefficient,
        p2_coefficient=p2_coefficient,
        pc_coefficient=pc_coefficient,
        coupling_weight=coupling_weight,
        frequency_drift_weight=frequency_drift_weight,
        control_drift_weight=control_drift_weight,
        colored_weight=colored_weight,
        static_coupling_pairs=zero_based_static_pairs,
        frequency_drift_qubits=zero_based_frequency_drifts,
        control_drift_qubits=zero_based_drifts,
        colored_noise_qubit=zero_based_colored,
    )
    return chain, problem, robustness, locations


def quasistatic_operator_catalog(chain: SpinChain) -> tuple[Qobj, ...]:
    """Return deterministic unit-scale local operators for block scaling.

    The ordered catalog starts with pair interactions matching the spin-chain
    coupling (``ZZ`` or ``XX+YY``), followed by local ``Z``, ``X``, and ``Y``
    terms.  Increasing the requested source count therefore changes only the
    Taylor source count while keeping the physical Hilbert space and control
    system fixed.
    """

    n_qubits = len(chain.z_operators)

    def pair_interaction(left: int, right: int) -> Qobj:
        if chain.coupling == "ZZ":
            return chain.z_operators[left] @ chain.z_operators[right]
        lowering = chain.lowering_operators
        return 2.0 * (
            lowering[left] @ lowering[right].dag() + lowering[left].dag() @ lowering[right]
        )

    nearest_couplings = chain.coupling_operators
    local_z = chain.z_operators
    local_x = chain.control_hamiltonians[:n_qubits]
    local_y = chain.control_hamiltonians[n_qubits:]
    non_nearest_couplings = tuple(
        pair_interaction(left, right)
        for left in range(n_qubits)
        for right in range(left + 2, n_qubits)
    )
    return (
        *nearest_couplings,
        *local_z,
        *local_x,
        *local_y,
        *non_nearest_couplings,
    )


def pauli_catalog_labels(n_qubits: int, *, seed: int) -> tuple[str, ...]:
    """Return a reproducibly shuffled basis of non-identity Pauli strings."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    labels = tuple(
        "".join(symbols)
        for symbols in product("IXYZ", repeat=n_qubits)
        if any(symbol != "I" for symbol in symbols)
    )
    permutation = np.random.default_rng(seed).permutation(len(labels))
    return tuple(labels[int(index)] for index in permutation)


def pauli_operator_catalog(chain: SpinChain, *, seed: int) -> tuple[Qobj, ...]:
    """Return all non-identity Pauli strings in one reproducible order."""

    n_qubits = len(chain.z_operators)
    labels = pauli_catalog_labels(n_qubits, seed=seed)
    single_qubit = {
        "I": qutip.qeye(2),
        "X": qutip.sigmax(),
        "Y": qutip.sigmay(),
        "Z": qutip.sigmaz(),
    }
    return tuple(qutip.tensor(*(single_qubit[symbol] for symbol in label)) for label in labels)


def coupling_matched_catalog_labels(
    n_qubits: int,
    *,
    coupling: str,
) -> tuple[str, ...]:
    """Describe the local and pair sources in coupling-matched catalog order."""

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two.")
    interaction = normalize_coupling_model(coupling)
    nearest = tuple(f"{interaction}[{site + 1},{site + 2}]" for site in range(n_qubits - 1))
    local = tuple(f"{axis}{site + 1}" for axis in "ZXY" for site in range(n_qubits))
    non_nearest = tuple(
        f"{interaction}[{left + 1},{right + 1}]"
        for left in range(n_qubits)
        for right in range(left + 2, n_qubits)
    )
    return (*nearest, *local, *non_nearest)


def quasistatic_catalog_labels(
    chain: SpinChain,
    *,
    source_family: str,
    pauli_catalog_seed: int,
) -> tuple[str, ...]:
    """Return labels in exactly the same order as the selected operator catalog."""

    n_qubits = len(chain.z_operators)
    if source_family == "local-z":
        return tuple(f"Z{site + 1}" for site in range(n_qubits))
    if source_family == "pauli":
        return pauli_catalog_labels(n_qubits, seed=pauli_catalog_seed)
    if source_family not in {"catalog", "coupling-matched"}:
        raise ValueError("source_family must be 'coupling-matched', 'pauli', or 'local-z'.")

    return coupling_matched_catalog_labels(
        n_qubits,
        coupling=chain.coupling,
    )


def build_quasistatic_problem(
    task: ExperimentTask | str,
    n_qubits: int,
    *,
    time_step: float,
    n_time_steps: int,
    source_count: int,
    order: int,
    source_family: str = "coupling-matched",
    pauli_catalog_seed: int = 42,
    coupling: str = "XX+YY",
) -> tuple[SpinChain, ControlProblem, RobustnessModel]:
    """Build a short controlled benchmark with a chosen Taylor block count."""

    selected_task = ExperimentTask(task)
    chain, paper_problem = build_paper_problem(
        n_qubits,
        selected_task.control_task,
        time_step=time_step,
        coupling=coupling,
    )
    if n_time_steps < 1:
        raise ValueError("n_time_steps must be positive.")
    problem = replace(paper_problem, duration=n_time_steps * time_step)
    if source_family == "local-z":
        catalog = chain.z_operators
    elif source_family in {"catalog", "coupling-matched"}:
        catalog = quasistatic_operator_catalog(chain)
    elif source_family == "pauli":
        catalog = pauli_operator_catalog(chain, seed=pauli_catalog_seed)
    else:
        raise ValueError("source_family must be 'coupling-matched', 'pauli', or 'local-z'.")
    if source_count < 1 or source_count > len(catalog):
        raise ValueError(
            f"source_count={source_count} is outside [1, {len(catalog)}] "
            f"for n={n_qubits} and source_family={source_family!r}."
        )
    scale = 2.0 * np.pi
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=tuple(scale * operator for operator in catalog[:source_count]),
            order=order,
            penalty_weight=1.0,
        )
    )
    return chain, problem, robustness


def augmented_block_count(source_count: int, order: int) -> int:
    """Return ``binomial(source_count + order, order)``."""

    if source_count < 1 or order < 1:
        raise ValueError("source_count and order must be positive.")
    return math.comb(source_count + order, order)


def block_structure(problem: ControlProblem, robustness: RobustnessModel) -> tuple[int, int]:
    """Return the Taylor block count and stored nonzero operator-block count."""

    components = build_augmented_hamiltonians(problem, robustness)
    if len(components) != 1:
        raise ValueError("Controlled scaling requires one Taylor component only.")
    component = components[0]
    operators = (component.drift, *component.controls)
    nonzero_blocks = sum(len(operator.rows) for operator in operators)
    return component.n_blocks, nonzero_blocks


def random_controls(
    problem: ControlProblem,
    *,
    seed: int,
    maximum_mhz: float = 30.0,
) -> RealArray:
    """Generate one deterministic applied-PWC field for controlled timing."""

    rng = np.random.default_rng(seed)
    maximum = maximum_mhz * 2.0 * np.pi
    return rng.uniform(
        -maximum,
        maximum,
        size=(problem.n_controls, problem.n_time_steps),
    )


def _evaluate_ode(
    components: Sequence[OdeComponent],
    problem: ControlProblem,
    controls: RealArray,
    *,
    compute_gradient: bool,
) -> tuple[float, int]:
    loss, _, rhs_evaluations = _evaluate_ode_detailed(
        components,
        problem,
        controls,
        compute_gradient=compute_gradient,
    )
    return loss, rhs_evaluations


def _evaluate_ode_detailed(
    components: Sequence[OdeComponent],
    problem: ControlProblem,
    controls: RealArray,
    *,
    compute_gradient: bool,
) -> tuple[float, RealArray | None, int]:
    fidelity_weight = 1.0 / len(components)
    loss = 0.0
    gradient = np.zeros_like(controls) if compute_gradient else None
    rhs_evaluations = 0
    for component in components:
        evaluation = component.evaluate(
            problem,
            controls,
            fidelity_weight=fidelity_weight,
            compute_gradient=compute_gradient,
        )
        loss += evaluation.objective.loss
        rhs_evaluations += evaluation.rhs_evaluations
        if gradient is not None and evaluation.amplitude_gradient is not None:
            gradient += evaluation.amplitude_gradient
    return loss, gradient, rhs_evaluations


def _evaluate_trotter(
    components: Sequence[TrotterComponent],
    problem: ControlProblem,
    controls: RealArray,
    *,
    compute_gradient: bool,
) -> tuple[float, int]:
    loss, _, rhs_evaluations = _evaluate_trotter_detailed(
        components,
        problem,
        controls,
        compute_gradient=compute_gradient,
    )
    return loss, rhs_evaluations


def _evaluate_trotter_detailed(
    components: Sequence[TrotterComponent],
    problem: ControlProblem,
    controls: RealArray,
    *,
    compute_gradient: bool,
) -> tuple[float, RealArray | None, int]:
    fidelity_weight = 1.0 / len(components)
    loss = 0.0
    gradient = np.zeros_like(controls) if compute_gradient else None
    for component in components:
        evaluation = component.evaluate(
            problem,
            controls,
            fidelity_weight=fidelity_weight,
            compute_gradient=compute_gradient,
        )
        loss += evaluation.objective.loss
        if gradient is not None and evaluation.amplitude_gradient is not None:
            gradient += evaluation.amplitude_gradient
    return loss, gradient, 0


def evaluate_method_once(
    method: str,
    *,
    chain: SpinChain,
    problem: ControlProblem,
    robustness: RobustnessModel,
    controls: RealArray,
    ode_method: str,
    ode_rtol: float | None,
    ode_atol: float | None,
    splitting_steps: int = 1,
) -> tuple[float, RealArray, int]:
    """Evaluate one loss and amplitude gradient for cross-method diagnostics."""

    if method == "ST":
        components = build_trotter_components(
            problem,
            robustness,
            chain.control_groups,
            splitting_steps=splitting_steps,
        )
        loss, gradient, rhs_evaluations = _evaluate_trotter_detailed(
            components,
            problem,
            controls,
            compute_gradient=True,
        )
    elif method in {"dense", "block-sparse"}:
        propagation = configure_ode_propagation(
            method=ode_method,
            representation=OdeRepresentation(method),
            rtol=ode_rtol,
            atol=ode_atol,
        )
        components = build_ode_components(problem, robustness, propagation)
        loss, gradient, rhs_evaluations = _evaluate_ode_detailed(
            components,
            problem,
            controls,
            compute_gradient=True,
        )
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    if gradient is None:  # pragma: no cover - compute_gradient is always true.
        raise RuntimeError("The requested method did not return a gradient.")
    return loss, gradient, rhs_evaluations


def _array_storage_bytes(value: object) -> int:
    """Count unique NumPy buffers reachable from a component tree."""

    visited_objects: set[int] = set()
    visited_arrays: set[int] = set()

    def visit(candidate: object) -> int:
        identifier = id(candidate)
        if identifier in visited_objects:
            return 0
        visited_objects.add(identifier)
        if isinstance(candidate, np.ndarray):
            base = candidate
            while isinstance(base.base, np.ndarray):
                base = base.base
            buffer_id = id(base)
            if buffer_id in visited_arrays:
                return 0
            visited_arrays.add(buffer_id)
            return int(base.nbytes)
        if is_dataclass(candidate) and not isinstance(candidate, type):
            return sum(visit(getattr(candidate, field.name)) for field in fields(candidate))
        if isinstance(candidate, Mapping):
            return sum(visit(key) + visit(item) for key, item in candidate.items())
        if isinstance(candidate, (tuple, list, set, frozenset)):
            return sum(visit(item) for item in candidate)
        return 0

    return visit(value)


def benchmark_method(
    method: str,
    *,
    chain: SpinChain,
    problem: ControlProblem,
    robustness: RobustnessModel,
    controls: RealArray,
    ode_method: str,
    ode_rtol: float | None,
    ode_atol: float | None,
    splitting_steps: int,
    warmups: int,
    repeats: int,
) -> EvaluationMeasurement:
    """Measure raw forward and objective-gradient costs for one method."""

    if method not in COMPARISON_METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive.")
    # Components from the preceding method are local to its benchmark call.
    # Collect them before starting this method, outside every timed interval.
    # Warmups remain responsible for normalizing method-local cold-start effects;
    # deliberately flushing CPU/OS caches would be non-portable and distortive.
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.clear_traces()
    gc.collect()
    build_start = time.perf_counter()
    if method == "ST":
        components = build_trotter_components(
            problem,
            robustness,
            chain.control_groups,
            splitting_steps=splitting_steps,
        )

        def evaluate(gradient: bool) -> tuple[float, int]:
            return _evaluate_trotter(
                components,
                problem,
                controls,
                compute_gradient=gradient,
            )
    else:
        propagation = configure_ode_propagation(
            method=ode_method,
            representation=OdeRepresentation(method),
            rtol=ode_rtol,
            atol=ode_atol,
        )
        components = build_ode_components(problem, robustness, propagation)

        def evaluate(gradient: bool) -> tuple[float, int]:
            return _evaluate_ode(
                components,
                problem,
                controls,
                compute_gradient=gradient,
            )

    build_time = time.perf_counter() - build_start

    for _ in range(warmups):
        evaluate(True)

    forward_times: list[float] = []
    full_times: list[float] = []
    forward_losses: list[float] = []
    full_losses: list[float] = []
    forward_rhs: list[int] = []
    full_rhs: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter()
        loss, rhs = evaluate(False)
        forward_times.append(time.perf_counter() - start)
        forward_losses.append(loss)
        forward_rhs.append(rhs)

        start = time.perf_counter()
        loss, rhs = evaluate(True)
        full_times.append(time.perf_counter() - start)
        full_losses.append(loss)
        full_rhs.append(rhs)

    tracemalloc.start()
    evaluate(True)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return EvaluationMeasurement(
        build_time_s=build_time,
        forward_times_s=np.asarray(forward_times, dtype=float),
        objective_gradient_times_s=np.asarray(full_times, dtype=float),
        forward_losses=np.asarray(forward_losses, dtype=float),
        objective_gradient_losses=np.asarray(full_losses, dtype=float),
        forward_rhs_evaluations=np.asarray(forward_rhs, dtype=np.int64),
        objective_gradient_rhs_evaluations=np.asarray(full_rhs, dtype=np.int64),
        persistent_operator_bytes=_array_storage_bytes(components),
        peak_traced_allocation_bytes=peak_bytes,
    )


def measurements_to_arrays(
    measurements: Sequence[Sequence[EvaluationMeasurement]],
) -> dict[str, np.ndarray]:
    """Stack a point-by-method table of raw benchmark measurements."""

    if not measurements or not measurements[0]:
        raise ValueError("measurements cannot be empty.")
    width = len(measurements[0])
    if any(len(row) != width for row in measurements):
        raise ValueError("every benchmark point must contain the same methods.")
    return {
        "build_times_s": np.asarray(
            [[entry.build_time_s for entry in row] for row in measurements],
            dtype=float,
        ),
        "forward_times_s": np.asarray(
            [[entry.forward_times_s for entry in row] for row in measurements],
            dtype=float,
        ),
        "objective_gradient_times_s": np.asarray(
            [[entry.objective_gradient_times_s for entry in row] for row in measurements],
            dtype=float,
        ),
        "forward_losses": np.asarray(
            [[entry.forward_losses for entry in row] for row in measurements],
            dtype=float,
        ),
        "objective_gradient_losses": np.asarray(
            [[entry.objective_gradient_losses for entry in row] for row in measurements],
            dtype=float,
        ),
        "forward_rhs_evaluations": np.asarray(
            [[entry.forward_rhs_evaluations for entry in row] for row in measurements],
            dtype=np.int64,
        ),
        "objective_gradient_rhs_evaluations": np.asarray(
            [[entry.objective_gradient_rhs_evaluations for entry in row] for row in measurements],
            dtype=np.int64,
        ),
        "persistent_operator_bytes": np.asarray(
            [[entry.persistent_operator_bytes for entry in row] for row in measurements],
            dtype=np.int64,
        ),
        "peak_traced_allocation_bytes": np.asarray(
            [[entry.peak_traced_allocation_bytes for entry in row] for row in measurements],
            dtype=np.int64,
        ),
    }


def save_method_benchmark_data(
    output: Path,
    *,
    methods: Sequence[str],
    shared_arrays: Mapping[str, Any],
    measurement_arrays: Mapping[str, np.ndarray],
) -> None:
    """Save each benchmark method independently for resumable comparisons.

    Every measurement array has point and method as its first two axes.  A
    later invocation may therefore replace one selected method without
    touching measurements already collected for other methods.
    """

    if not methods:
        raise ValueError("At least one benchmark method is required.")
    for name, values in measurement_arrays.items():
        if values.ndim < 2 or values.shape[1] != len(methods):
            raise ValueError(
                f"measurement array {name!r} must have method axis length {len(methods)}."
            )
    for method_index, method in enumerate(methods):
        method_directory = output / method
        method_directory.mkdir(parents=True, exist_ok=True)
        raw_path = method_directory / "raw_data.npz"
        if raw_path.exists():
            print(f"Warning: replacing existing {method} benchmark at {raw_path}", flush=True)
        payload: dict[str, Any] = {
            "method": method,
            **shared_arrays,
            **{
                name: values[:, method_index : method_index + 1]
                for name, values in measurement_arrays.items()
            },
        }
        np.savez_compressed(raw_path, **payload)
