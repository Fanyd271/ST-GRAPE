"""Paired Monte Carlo infidelity scans for the saved GHZ and QFT controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stgrape.benchmarks.common import build_multisource_problem
from stgrape.paths import data_root, ensure_output_path
from stgrape.sampling import NoiseRealization, evaluate_realization, generate_noise_realizations

DEFAULT_RESULT_DIR = Path("outputs/strength_scan")
DEFAULT_STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CONTROL_LABELS = ("initial", "r1", "r2")


@dataclass(frozen=True)
class TaskRun:
    name: str
    n_qubits: int
    directory: Path


def validate_strengths(strengths: np.ndarray) -> np.ndarray:
    """Validate the sweep grid regardless of where its values came from."""

    if (
        strengths.ndim != 1
        or strengths.size == 0
        or not np.all(np.isfinite(strengths))
        or np.any(strengths < 0)
        or np.any(np.diff(strengths) <= 0)
    ):
        raise ValueError("strengths must be finite, nonnegative, and strictly increasing")
    return strengths


def parse_strengths(value: str) -> np.ndarray:
    """Parse a comma-separated CLI sweep grid."""

    try:
        strengths = np.asarray([float(part.strip()) for part in value.split(",")], dtype=float)
        return validate_strengths(strengths)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def scaled_realization(realization: NoiseRealization, strength: float) -> NoiseRealization:
    """Scale the same static, multiplicative, and colored draw by one factor."""

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    return NoiseRealization(
        static_amplitudes=strength * realization.static_amplitudes,
        control_amplitudes=strength * realization.control_amplitudes,
        colored_trajectories=strength * realization.colored_trajectories,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(run: TaskRun):
    manifest_path = run.directory / "run_manifest.json"
    config = json.loads(manifest_path.read_text(encoding="utf-8"))["configuration"]
    if config["task"] != run.name:
        raise ValueError(f"Unexpected task in {manifest_path}")
    _, problem, robustness, _ = build_multisource_problem(
        run.name,
        run.n_qubits,
        time_step=config["time_step_us"],
        duration=config["duration_us"],
        order=1,
        coupling=config["coupling"],
        static_standard_deviation_mhz=config["static_std_mhz"],
        frequency_drift_standard_deviation_mhz=config["frequency_drift_std_mhz"],
        control_drift_standard_deviation=config["control_drift_std"],
        colored_standard_deviation_mhz=config["colored_std_mhz"],
        first_order_penalty=config["first_order_penalty"],
        second_order_penalty=config["second_order_penalty"],
        penalty_mode=config["penalty_mode"],
        rms_smoothing=config["rms_smoothing"],
        p1_smoothing=config["p1_smoothing"],
        p2_smoothing=config["p2_smoothing"],
        pc_smoothing=config["pc_smoothing"],
        p1_coefficient=config["p1_coefficient"],
        p2_coefficient=config["p2_coefficient"],
        pc_coefficient=config["pc_coefficient"],
        coupling_weight=config["coupling_weight"],
        frequency_drift_weight=config["frequency_drift_weight"],
        control_drift_weight=config["control_drift_weight"],
        colored_weight=config["colored_weight"],
        static_coupling_pairs=config["requested_static_coupling_pairs"],
        frequency_drift_qubits=config["requested_frequency_drift_qubits"],
        control_drift_qubits=config["requested_control_drift_qubits"],
        non_markovian_qubit=config["requested_non_markovian_qubit"],
        include_static_noise=config["static_noise_enabled"],
        include_frequency_drift=config["frequency_drift_noise_enabled"],
        include_control_drift=config["control_drift_noise_enabled"],
        include_non_markovian_noise=config["non_markovian_noise_enabled"],
    )
    return problem, robustness, manifest_path


def _load_controls(run: TaskRun, seed: int, problem):
    base = run.directory / f"{run.n_qubits}q"
    initial_path = base / "r1" / f"seed{seed}" / "initial_pulse.npz"
    second_initial_path = base / "r2" / f"seed{seed}" / "initial_pulse.npz"
    paths = {
        "initial": initial_path,
        "r1": base / "r1" / f"seed{seed}" / "ST" / "result.npz",
        "r2": base / "r2" / f"seed{seed}" / "ST" / "result.npz",
    }
    controls = {}
    expected_shape = (problem.n_controls, problem.n_time_steps)
    for label, path in paths.items():
        with np.load(path, allow_pickle=False) as saved:
            controls[label] = np.asarray(saved["control_amplitudes"], dtype=float)
        if controls[label].shape != expected_shape:
            raise ValueError(f"{path} has shape {controls[label].shape}, expected {expected_shape}")
    with np.load(second_initial_path, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["control_amplitudes"], controls["initial"])
    return controls, paths


_WORKER_CONTEXT = None


def _init_worker(problem, robustness, controls) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (problem, robustness, controls)


def _evaluate_controls(draw, problem, robustness, controls) -> np.ndarray:
    return np.asarray(
        [
            np.clip(
                1.0 - evaluate_realization(draw, problem, controls[label], robustness), 0.0, 1.0
            )
            for label in CONTROL_LABELS
        ],
        dtype=float,
    )


def _evaluate_worker(draw) -> np.ndarray:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker has not been initialized")
    return _evaluate_controls(draw, *_WORKER_CONTEXT)


def run_scan(
    run: TaskRun,
    *,
    seed: int,
    sampling_seed: int,
    n_samples: int,
    workers: int,
    strengths: np.ndarray,
    output_directory: Path,
    overwrite: bool,
) -> Path:
    """Use common random numbers across pulses and across all sweep strengths."""

    ensure_output_path(output_directory)
    output_path = output_directory / f"{run.name}_seed{seed}.npz"
    metadata_path = output_directory / f"{run.name}_seed{seed}.json"
    summary_path = output_directory / f"{run.name}_seed{seed}.csv"
    existing = [path for path in (output_path, metadata_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output exists: {existing}. Pass --overwrite to replace it.")

    problem, robustness, manifest_path = _load_model(run)
    controls, paths = _load_controls(run, seed, problem)
    draws = generate_noise_realizations(
        problem,
        robustness,
        n_samples=n_samples,
        rng=np.random.default_rng(sampling_seed),
        colored_noise_sampling="reference-gaussian",
    )
    values = np.empty((len(CONTROL_LABELS), len(strengths), n_samples), dtype=float)
    started = time.perf_counter()
    pool = (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(problem, robustness, controls),
        )
        if workers > 1
        else nullcontext(None)
    )
    with pool as executor:
        for strength_index, strength in enumerate(strengths):
            if strength == 0:
                nominal = _evaluate_controls(
                    scaled_realization(draws[0], 0.0), problem, robustness, controls
                )
                values[:, strength_index, :] = nominal[:, None]
            else:
                scaled_draws = (scaled_realization(draw, float(strength)) for draw in draws)
                if executor is None:
                    evaluated = (
                        _evaluate_controls(draw, problem, robustness, controls)
                        for draw in scaled_draws
                    )
                else:
                    evaluated = executor.map(_evaluate_worker, scaled_draws, chunksize=8)
                values[:, strength_index, :] = np.asarray(list(evaluated)).T
            means = values[:, strength_index, :].mean(axis=1)
            print(
                f"{run.name} s={strength:g}: "
                + ", ".join(
                    f"{label}={mean:.6g}" for label, mean in zip(CONTROL_LABELS, means, strict=True)
                )
                + f" ({time.perf_counter() - started:.1f} s)",
                flush=True,
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        strengths=strengths,
        labels=np.asarray(CONTROL_LABELS),
        infidelities=values,
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("strength", "control", "mean_infidelity", "sem_infidelity"))
        for strength_index, strength in enumerate(strengths):
            for control_index, label in enumerate(CONTROL_LABELS):
                observations = values[control_index, strength_index]
                sem = (
                    observations.std(ddof=1) / np.sqrt(n_samples)
                    if strength > 0 and n_samples > 1
                    else 0.0
                )
                writer.writerow((strength, label, observations.mean(), sem))
    metadata = {
        "task": run.name,
        "n_qubits": run.n_qubits,
        "pulse_seed": seed,
        "sampling_seed": sampling_seed,
        "samples_per_nonzero_strength": n_samples,
        "workers": workers,
        "colored_noise_sampling": "reference-gaussian",
        "pairing": "same standardized draws for all controls and strengths",
        "zero_strength": "one nominal evaluation per control, broadcast to all samples",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "control_paths": {label: str(path.resolve()) for label, path in paths.items()},
        "control_sha256": {label: _sha256(path) for label, path in paths.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("ghz", "qft"), default=("ghz", "qft"))
    parser.add_argument("--seed", type=int, default=0, help="Paired exp01 pulse seed")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=500, help="Draws per nonzero strength")
    parser.add_argument("--workers", type=int, default=8, help="Parallel simulation processes")
    parser.add_argument(
        "--strengths",
        type=parse_strengths,
        default=np.asarray(DEFAULT_STRENGTHS),
        help="Increasing nonnegative, comma-separated factors (default: 0 to 3)",
    )
    parser.add_argument("--ghz-run", type=Path, default=data_root() / "paper/ghz")
    parser.add_argument("--qft-run", type=Path, default=data_root() / "paper/qft")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(arguments)
    if args.samples < 1 or args.workers < 1 or args.seed < 0:
        parser.error("--samples and --workers must be positive; --seed must be nonnegative")
    if not args.tasks or set(args.tasks) - {"ghz", "qft"}:
        parser.error("--tasks must contain ghz, qft, or both")
    try:
        strengths = validate_strengths(np.asarray(args.strengths, dtype=float))
    except ValueError as error:
        parser.error(str(error))
    runs = {
        "ghz": TaskRun("ghz", 4, Path(args.ghz_run)),
        "qft": TaskRun("qft", 3, Path(args.qft_run)),
    }
    ensure_output_path(Path(args.output_dir))
    for task in args.tasks:
        output_path = run_scan(
            runs[task],
            seed=args.seed,
            sampling_seed=args.sampling_seed,
            n_samples=args.samples,
            workers=args.workers,
            strengths=strengths,
            output_directory=Path(args.output_dir),
            overwrite=args.overwrite,
        )
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
