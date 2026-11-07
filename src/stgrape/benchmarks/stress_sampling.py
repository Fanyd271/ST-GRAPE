"""Backfill paired Monte Carlo validation for completed Experiment 03 results.

This command never invokes an optimizer.  It reconstructs the physical model
from the saved run manifest, loads the initial and final applied controls, and
writes deterministic paired-fidelity samples to a configuration-hashed
validation directory. Different sample counts and seeds therefore coexist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from stgrape.benchmarks.common import ExperimentTask
from stgrape.execution import (
    EXPERIMENT_METHODS,
    read_json_object,
    sampling_summary,
    write_json,
)
from stgrape.models import build_paper_problem, build_static_coupling_robustness
from stgrape.paths import ensure_output_path
from stgrape.sampling import sample_fidelities

VALIDATION_MANIFEST_NAME = "monte_carlo_validation.json"
VALIDATION_SUMMARY_NAME = "monte_carlo_summary.csv"
VALIDATION_ROOT_NAME = "monte_carlo"
VALIDATION_INDEX_NAME = "index.json"
VALIDATION_SCHEMA_VERSION = 2
HASH_LENGTH = 12


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for JSON-compatible configuration data."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_directory(
    result_root: Path,
    source_configuration: Mapping[str, Any],
    *,
    samples: int,
    sampling_seed: int,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve the immutable directory for one Monte Carlo configuration."""

    sampling_configuration: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples_per_control": samples,
        "sampling_seed": sampling_seed,
        "sampling_api": "sample_fidelities-v1",
        "colored_noise_sampling": "auto",
        "source_configuration_sha256": _canonical_hash(source_configuration),
    }
    configuration_hash = _canonical_hash(sampling_configuration)[:HASH_LENGTH]
    directory_name = f"samples-{samples}_seed-{sampling_seed}_{configuration_hash}"
    return (
        result_root / VALIDATION_ROOT_NAME / directory_name,
        configuration_hash,
        sampling_configuration,
    )


def _update_validation_index(
    result_root: Path,
    validation_root: Path,
    configuration_hash: str,
    sampling_configuration: Mapping[str, Any],
) -> None:
    """Register one hashed validation run without disturbing earlier entries."""

    index_path = result_root / VALIDATION_ROOT_NAME / VALIDATION_INDEX_NAME
    if index_path.is_file():
        index = read_json_object(index_path)
    else:
        index = {"schema_version": 1, "runs": {}}
    runs = index.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"Invalid Monte Carlo index: {index_path}")
    runs[configuration_hash] = {
        "directory": validation_root.name,
        "configuration": dict(sampling_configuration),
    }
    index["runs"] = runs
    write_json(index_path, index)


def _load_controls(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Saved controls do not exist: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if "control_amplitudes" not in saved:
            raise ValueError(f"control_amplitudes are absent from {path}")
        return np.asarray(saved["control_amplitudes"], dtype=float)


def _save_fidelities(path: Path, values: np.ndarray) -> None:
    """Atomically replace one fidelity sample file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.pending-{uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, values=np.asarray(values, dtype=float))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _available_seeds(result_root: Path, n_qubits: int) -> tuple[int, ...]:
    qubit_root = result_root / f"{n_qubits}q"
    return tuple(
        sorted(
            int(path.name.removeprefix("seed"))
            for path in qubit_root.glob("seed*")
            if path.is_dir() and path.name.removeprefix("seed").isdigit()
        )
    )


def _selected_methods(
    result_root: Path,
    n_qubits: int,
    seeds: Sequence[int],
    requested: Sequence[str],
) -> tuple[str, ...]:
    available = tuple(
        method
        for method in EXPERIMENT_METHODS
        if all(
            (result_root / f"{n_qubits}q" / f"seed{seed}" / method / "result.npz").is_file()
            for seed in seeds
        )
    )
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("--method all cannot be combined with named methods.")
        if not available:
            raise FileNotFoundError("No method is complete for every selected seed.")
        return available
    methods = tuple(dict.fromkeys(requested))
    missing = tuple(method for method in methods if method not in available)
    if missing:
        raise FileNotFoundError(
            f"Methods are incomplete for the selected seeds: {', '.join(missing)}"
        )
    return methods


def _check_existing_validation(
    path: Path,
    *,
    samples: int,
    sampling_seed: int,
    existing_result: str,
) -> None:
    if not path.is_file() or existing_result == "overwrite":
        return
    saved = read_json_object(path)
    expected = {
        "samples_per_control": samples,
        "sampling_seed": sampling_seed,
    }
    mismatches = {
        key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing validation settings conflict at {path}: {mismatches}. "
            "Use --existing-result overwrite to replace the fidelity samples."
        )


def _write_sampling_summary(
    validation_root: Path,
    completed: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    rows: list[dict[str, Any]] = []
    for qubit_key, seed_records in sorted(completed.items()):
        n_qubits = int(qubit_key.removesuffix("q"))
        for seed_key, stages in sorted(seed_records.items()):
            seed = int(seed_key.removeprefix("seed"))
            case = validation_root / qubit_key / seed_key
            for stage in stages:
                path = (
                    case / "initial_fidelities.npz"
                    if stage == "initial"
                    else case / stage / "fidelities.npz"
                )
                if not path.is_file():
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    values = np.asarray(saved["values"], dtype=float)
                rows.append(
                    {
                        "n_qubits": n_qubits,
                        "seed": seed,
                        "stage": stage,
                        "n_samples": values.size,
                        **sampling_summary(values),
                    }
                )
    if not rows:
        return
    with (validation_root / VALIDATION_SUMMARY_NAME).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def backfill_monte_carlo(arguments: argparse.Namespace) -> Path:
    """Sample existing controls into a hashed validation directory."""

    result_root = arguments.result_root.resolve()
    ensure_output_path(result_root)
    manifest_path = result_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment 03 run manifest does not exist: {manifest_path}")
    if arguments.samples < 1:
        raise ValueError("--samples must be positive.")
    if arguments.sampling_workers < 1:
        raise ValueError("--sampling-workers must be positive.")

    manifest = read_json_object(manifest_path)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"Invalid run configuration in {manifest_path}")
    validation_root, configuration_hash, sampling_configuration = _validation_directory(
        result_root,
        configuration,
        samples=arguments.samples,
        sampling_seed=arguments.sampling_seed,
    )
    validation_root.mkdir(parents=True, exist_ok=True)
    task = ExperimentTask(str(configuration["task"]))
    coupling = str(configuration["coupling"])
    duration = float(configuration["duration_us"])
    time_step = float(configuration["time_step_us"])
    order = int(configuration["robustness_order"])
    standard_deviation_mhz = float(configuration["static_standard_deviation_mhz"])
    penalty_weight = float(configuration["taylor_penalty_weight"])

    qubit_counts = tuple(
        sorted(
            int(path.name[:-1])
            for path in result_root.glob("*q")
            if path.is_dir() and path.name[:-1].isdigit()
        )
    )
    if arguments.qubits is not None:
        if arguments.qubits not in qubit_counts:
            raise FileNotFoundError(f"{arguments.qubits}q is absent from {result_root}")
        qubit_counts = (arguments.qubits,)
    if not qubit_counts:
        raise FileNotFoundError(f"No qubit result directories exist in {result_root}")

    validation_path = validation_root / VALIDATION_MANIFEST_NAME
    _check_existing_validation(
        validation_path,
        samples=arguments.samples,
        sampling_seed=arguments.sampling_seed,
        existing_result=arguments.existing_result,
    )
    completed: dict[str, dict[str, list[str]]]
    all_methods: set[str]
    if validation_path.is_file():
        previous_validation = read_json_object(validation_path)
        previous_completed = previous_validation.get("completed", {})
        completed = {
            str(qubit_key): {
                str(seed_key): list(stages) for seed_key, stages in seed_records.items()
            }
            for qubit_key, seed_records in previous_completed.items()
        }
        all_methods = set(previous_validation.get("methods", ()))
    else:
        completed = {}
        all_methods = set()

    for n_qubits in qubit_counts:
        available_seeds = _available_seeds(result_root, n_qubits)
        seeds = available_seeds if arguments.seeds is None else tuple(arguments.seeds)
        missing_seeds = tuple(seed for seed in seeds if seed not in available_seeds)
        if missing_seeds:
            raise FileNotFoundError(f"Seeds are absent from {n_qubits}q results: {missing_seeds}")
        methods = _selected_methods(
            result_root,
            n_qubits,
            seeds,
            arguments.method,
        )
        all_methods.update(methods)

        chain, problem = build_paper_problem(
            n_qubits,
            task.control_task,
            time_step=time_step,
            duration=duration,
            coupling=coupling,
        )
        pairs_by_size = configuration["static_coupling_pairs_by_size"]
        pairs = pairs_by_size.get(str(n_qubits), pairs_by_size.get(n_qubits))
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"Static coupling pairs are missing for {n_qubits} qubits.")
        coupling_links = tuple(int(pair[0]) - 1 for pair in pairs)
        robustness = build_static_coupling_robustness(
            chain,
            order=order,
            standard_deviation_mhz=standard_deviation_mhz,
            penalty_weight=penalty_weight,
            coupling_link_indices=coupling_links,
        )

        for seed in seeds:
            source_case = result_root / f"{n_qubits}q" / f"seed{seed}"
            validation_case = validation_root / f"{n_qubits}q" / f"seed{seed}"
            requested_controls: dict[str, np.ndarray] = {}
            initial_path = validation_case / "initial_fidelities.npz"
            if arguments.existing_result == "overwrite" or not initial_path.is_file():
                requested_controls["initial"] = _load_controls(source_case / "initial_pulse.npz")
            for method in methods:
                fidelity_path = validation_case / method / "fidelities.npz"
                if arguments.existing_result == "overwrite" or not fidelity_path.is_file():
                    requested_controls[method] = _load_controls(source_case / method / "result.npz")

            if requested_controls:
                print(
                    f"Sampling {n_qubits}q seed={seed}: "
                    f"{', '.join(requested_controls)} with "
                    f"{arguments.samples} trajectories on "
                    f"{arguments.sampling_workers} process(es).",
                    flush=True,
                )
                sampled: dict[str, np.ndarray] = {}
                for stage, controls in requested_controls.items():
                    print(f"  [{stage}] starting ...", flush=True)

                    def report_progress(
                        completed_count: int,
                        total_count: int,
                        *,
                        stage_name: str = stage,
                    ) -> None:
                        if (
                            completed_count == 1
                            or completed_count % arguments.progress_interval == 0
                            or completed_count == total_count
                        ):
                            percentage = 100.0 * completed_count / total_count
                            print(
                                f"  [{stage_name}] {completed_count}/{total_count} "
                                f"({percentage:.1f}%)",
                                flush=True,
                            )

                    sampled[stage] = sample_fidelities(
                        problem,
                        controls,
                        robustness,
                        n_samples=arguments.samples,
                        seed=arguments.sampling_seed,
                        workers=arguments.sampling_workers,
                        progress_callback=report_progress,
                    )
                for stage, values in sampled.items():
                    destination = (
                        initial_path
                        if stage == "initial"
                        else validation_case / stage / "fidelities.npz"
                    )
                    _save_fidelities(destination, values)
            else:
                print(f"Skipping completed Monte Carlo data for {n_qubits}q seed={seed}.")

            with np.load(initial_path, allow_pickle=False) as saved:
                initial_values = np.asarray(saved["values"], dtype=float)
            if initial_values.size != arguments.samples:
                raise ValueError(
                    f"{initial_path} contains {initial_values.size} samples; "
                    f"expected {arguments.samples}."
                )
            completed_stages = ["initial"]
            for method in methods:
                fidelity_path = validation_case / method / "fidelities.npz"
                if not fidelity_path.is_file():
                    raise RuntimeError(f"Monte Carlo output is incomplete: {fidelity_path}")
                with np.load(fidelity_path, allow_pickle=False) as saved:
                    values = np.asarray(saved["values"], dtype=float)
                if values.size != arguments.samples:
                    raise ValueError(
                        f"{fidelity_path} contains {values.size} samples; "
                        f"expected {arguments.samples}."
                    )
                completed_stages.append(method)
            qubit_completed = completed.setdefault(f"{n_qubits}q", {})
            previous_stages = qubit_completed.get(f"seed{seed}", [])
            qubit_completed[f"seed{seed}"] = list(
                dict.fromkeys((*previous_stages, *completed_stages))
            )

    write_json(
        validation_path,
        {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "experiment": "Experiment 03 post-run Monte Carlo validation",
            "configuration_hash": configuration_hash,
            "sampling_configuration": sampling_configuration,
            "source_result_root": str(result_root),
            "samples_per_control": arguments.samples,
            "sampling_seed": arguments.sampling_seed,
            "sampling_workers": arguments.sampling_workers,
            "methods": [method for method in EXPERIMENT_METHODS if method in all_methods],
            "completed": completed,
            "optimization_rerun": False,
        },
    )
    _write_sampling_summary(validation_root, completed)
    _update_validation_index(
        result_root,
        validation_root,
        configuration_hash,
        sampling_configuration,
    )
    print(
        f"Saved Monte Carlo validation under {validation_root} (hash={configuration_hash})",
        flush=True,
    )
    return validation_root


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--qubits", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=("ST", "dense", "block-sparse", "all"),
        default=("all",),
    )
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-workers",
        "--workers",
        dest="sampling_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--existing-result",
        choices=("skip", "overwrite"),
        default="skip",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print completed Monte Carlo trajectories at this interval.",
    )
    parsed = parser.parse_args(arguments)
    if parsed.progress_interval < 1:
        parser.error("--progress-interval must be positive.")
    return parsed


def main(arguments: Sequence[str] | None = None) -> None:
    backfill_monte_carlo(parse_arguments(arguments))


if __name__ == "__main__":
    main()
