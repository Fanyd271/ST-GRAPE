"""Experiment C: fixed-budget large-system GHZ/QFT stress test.

All three methods can start from the same optimized pulse and receive the same
optimization-time cap. The default GHZ case uses nine qubits, independent
fluctuations on four disjoint nearest-neighbor couplings, and first-order
robustness. Optional Monte Carlo validation and the ST final-only reference
check are performed outside the timed optimization.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.benchmarks.common import (
    ExperimentTask,
    select_pending_methods,
    selected_methods,
    unique_output_directory,
)
from stgrape.execution import (
    generated_seed_coupling,
    generated_seed_duration,
    generated_seed_time_step,
    load_initial_pulse,
    preflight_initial_pulses,
    prepare_resumable_output,
    read_json_object,
    run_and_save_paper_methods,
)
from stgrape.io import pulse_from_result
from stgrape.models import (
    build_paper_problem,
    build_static_coupling_robustness,
    default_paper_duration,
)
from stgrape.problem import ControlProblem
from stgrape.pulse import Pulse

EXPERIMENT_NAME = "C fixed-budget large-system stress test"
DEFAULT_SEEDS = (0,)
DEFAULT_GHZ_QUBITS = (9,)
DEFAULT_QFT_QUBITS = tuple(range(4, 7))
DEFAULT_STRESS_GHZ_DURATION_US = 0.2


def _disjoint_coupling_link_indices(n_qubits: int) -> tuple[int, ...]:
    """Select the zero-based links (1,2), (3,4), and so on."""

    return tuple(range(0, n_qubits - 1, 2))


def _parse_static_pair(value: str) -> tuple[int, int]:
    """Parse a one-based nearest-neighbor bond such as ``3-4``."""

    try:
        left, right = (int(part) for part in value.split("-"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LEFT-RIGHT, for example 3-4.") from error
    if left < 1 or right != left + 1:
        raise argparse.ArgumentTypeError("Static coupling pairs must be adjacent, e.g. 3-4.")
    return left, right


def _coupling_link_indices(
    n_qubits: int, pairs: Sequence[tuple[int, int]] | None
) -> tuple[int, ...]:
    if pairs is None:
        return _disjoint_coupling_link_indices(n_qubits)
    if not pairs or any(left < 1 or right != left + 1 for left, right in pairs):
        raise ValueError("Static coupling pairs must be adjacent one-based links.")
    links = tuple(left - 1 for left, right in pairs)
    if len(set(links)) != len(links) or any(right > n_qubits for _, right in pairs):
        raise ValueError("Static coupling pairs must be distinct links within the chain.")
    return links


def _load_file_pulse(
    path: Path,
    problem: ControlProblem,
    *,
    task: ExperimentTask,
    n_qubits: int,
    coupling: str,
) -> Pulse:
    """Reconstruct a saved PWC optimum only if its model and grid match."""

    metadata_path = path.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Initial pulse metadata does not exist: {metadata_path}")
    metadata = read_json_object(metadata_path)
    expected_task_names = (task.value, task.control_task.value)
    if metadata.get("task") not in expected_task_names:
        raise ValueError(f"Initial pulse task does not match {task.value}: {metadata_path}")
    if metadata.get("n_qubits") != n_qubits or metadata.get("coupling") != coupling:
        raise ValueError(f"Initial pulse qubit count or coupling does not match: {metadata_path}")
    for key, expected in (("duration_us", problem.duration), ("time_step_us", problem.time_step)):
        try:
            matches = math.isclose(float(metadata[key]), expected, rel_tol=0.0, abs_tol=1e-12)
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"Initial pulse {key} does not match the stress-test grid: {metadata_path}"
            )
    if "n_time_steps" in metadata and metadata["n_time_steps"] != problem.n_time_steps:
        raise ValueError(f"Initial pulse step count does not match: {metadata_path}")
    pulse_metadata = metadata.get("pulse", {})
    if not isinstance(pulse_metadata, dict) or pulse_metadata.get("control_basis", "pwc") != "pwc":
        raise ValueError(f"Initial pulse must use filtered PWC controls: {metadata_path}")
    pulse = pulse_from_result(
        path,
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
    )
    applied_controls = pulse.amplitudes()
    with np.load(path, allow_pickle=False) as saved:
        if "control_amplitudes" not in saved.files:
            raise ValueError(f"Initial pulse result lacks control_amplitudes: {path}")
        if saved["control_amplitudes"].shape != applied_controls.shape or not np.allclose(
            saved["control_amplitudes"], applied_controls, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(f"Initial pulse waveform cannot be reconstructed: {path}")
    return pulse


def _time_step(arguments: argparse.Namespace) -> float:
    if arguments.time_step is not None:
        return arguments.time_step
    if arguments.initial_pulse is not None:
        return 0.0005
    if arguments.initial_source == "generated":
        if arguments.seed_root is None:
            raise ValueError("--seed-root is required for generated initial pulses.")
        return generated_seed_time_step(arguments.seed_root)
    return 0.0005


def _duration(
    arguments: argparse.Namespace,
    *,
    task: ExperimentTask,
    n_qubits: int,
    seed: int,
    initial_source: str,
) -> float:
    """Resolve an explicit, generated-seed, or paper-default duration."""

    if arguments.duration is not None:
        duration = float(arguments.duration)
    elif initial_source == "generated":
        if arguments.seed_root is None:
            raise ValueError("--seed-root is required for generated initial pulses.")
        duration = generated_seed_duration(
            arguments.seed_root,
            task=task.control_task,
            n_qubits=n_qubits,
            seed=seed,
        )
    else:
        duration = (
            DEFAULT_STRESS_GHZ_DURATION_US
            if task == ExperimentTask.GHZ
            else default_paper_duration(task.control_task)
        )
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive.")
    return duration


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Run the paired fixed-budget stress-test matrix and save raw outputs."""

    task = ExperimentTask(arguments.task)
    methods = selected_methods(arguments.method)
    qubits = (
        tuple(arguments.qubits)
        if arguments.qubits is not None
        else (DEFAULT_GHZ_QUBITS if task == ExperimentTask.GHZ else DEFAULT_QFT_QUBITS)
    )
    seeds = tuple(arguments.seeds)
    time_step = 0.0025 if arguments.smoke else _time_step(arguments)
    max_iterations = arguments.max_iterations
    time_limit_hours = arguments.time_limit_hours
    samples = arguments.samples
    initial_source = "file" if arguments.initial_pulse is not None else arguments.initial_source
    if arguments.initial_pulse is not None and arguments.seed_root is not None:
        raise ValueError("Use either --initial-pulse or --seed-root, not both.")
    if arguments.initial_pulse is not None and arguments.smoke:
        raise ValueError("--initial-pulse cannot be combined with --smoke.")
    periodic_max_qubits = arguments.periodic_true_loss_max_qubits
    if periodic_max_qubits is None:
        periodic_max_qubits = 8 if task == ExperimentTask.GHZ else 5
    if arguments.smoke:
        qubits = (2,)
        seeds = (0,)
        max_iterations = 1
        time_limit_hours = None
        samples = 2
        initial_source = "random"
        periodic_max_qubits = 2

    duration = _duration(
        arguments,
        task=task,
        n_qubits=qubits[0],
        seed=seeds[0],
        initial_source=initial_source,
    )

    if len(set(qubits)) != len(qubits) or any(value < 2 for value in qubits):
        raise ValueError("qubits must be unique integers of at least two.")
    if len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("seeds must be unique non-negative integers.")
    if arguments.order != 1:
        raise ValueError("The stress-test protocol fixes robustness order r=1.")
    if arguments.sampling_workers < 1 or samples < 0:
        raise ValueError("sampling workers must be positive and samples must be non-negative.")
    if arguments.splitting_steps < 1:
        raise ValueError("splitting steps must be positive.")
    if not math.isfinite(arguments.static_std_mhz) or arguments.static_std_mhz <= 0:
        raise ValueError("--static-std-mhz must be finite and positive.")
    if not math.isfinite(arguments.p1_coefficient) or arguments.p1_coefficient < 0:
        raise ValueError("--p1-coefficient must be finite and non-negative.")
    if initial_source == "file" and (len(qubits) != 1 or len(seeds) != 1):
        raise ValueError("--initial-pulse requires exactly one qubit count and one seed.")
    if initial_source == "generated" and arguments.seed_root is None:
        raise ValueError("--seed-root is required for generated initial pulses.")
    if initial_source == "generated" and arguments.seed_root is not None:
        seed_coupling = generated_seed_coupling(arguments.seed_root)
        if seed_coupling != arguments.coupling:
            raise ValueError(
                f"Initial E00 pulses use coupling={seed_coupling!r}, but this "
                f"run requests coupling={arguments.coupling!r}. Generate a "
                "matching E00 seed root first."
            )

    output = arguments.output or unique_output_directory("stress", task)
    problems = {
        n_qubits: build_paper_problem(
            n_qubits,
            task.control_task,
            time_step=time_step,
            duration=duration,
            coupling=arguments.coupling,
        )
        for n_qubits in qubits
    }
    coupling_links_by_size = {
        n_qubits: _coupling_link_indices(n_qubits, arguments.static_coupling_pairs)
        for n_qubits in qubits
    }
    file_pulse: Pulse | None = None
    if initial_source == "file":
        assert arguments.initial_pulse is not None
        file_pulse = _load_file_pulse(
            arguments.initial_pulse,
            problems[qubits[0]][1],
            task=task,
            n_qubits=qubits[0],
            coupling=arguments.coupling,
        )
        print(f"Input preflight passed: optimized pulse {arguments.initial_pulse}", flush=True)
    else:
        preflight_initial_pulses(
            tuple((problems[n_qubits][1], n_qubits, seed) for n_qubits in qubits for seed in seeds),
            source=initial_source,
            generated_root=arguments.seed_root,
        )
    configuration = {
        "task": task.value,
        "coupling": arguments.coupling,
        "duration_us": duration,
        "time_step_us": time_step,
        "robustness_order": 1,
        "static_noise": "independent fluctuations on selected nearest-neighbor couplings",
        "static_coupling_pairs_by_size": {
            str(n_qubits): [[index + 1, index + 2] for index in links]
            for n_qubits, links in coupling_links_by_size.items()
        },
        "static_standard_deviation_mhz": arguments.static_std_mhz,
        "p1_coefficient": arguments.p1_coefficient,
        "taylor_penalty_weight": 2.0 * arguments.p1_coefficient,
        "initial_source": initial_source,
        "initial_seed_root": (
            str(arguments.seed_root.resolve())
            if initial_source == "generated" and arguments.seed_root is not None
            else None
        ),
        "initial_pulse_path": (
            str(arguments.initial_pulse.resolve())
            if initial_source == "file" and arguments.initial_pulse is not None
            else None
        ),
        "ode_method": arguments.ode_method,
        "ode_rtol": arguments.ode_rtol,
        "ode_atol": arguments.ode_atol,
        "max_iterations": max_iterations,
        "time_limit_hours": time_limit_hours,
        "periodic_true_loss_max_qubits": periodic_max_qubits,
        "st_true_loss_mode": arguments.st_true_loss_mode,
        "exact_check_interval_when_enabled": arguments.exact_check_interval,
        "exact_increase_patience": arguments.exact_increase_patience,
        "exact_increase_absolute_tolerance": (arguments.exact_increase_absolute_tolerance),
        "exact_increase_relative_tolerance": (arguments.exact_increase_relative_tolerance),
        "splitting_steps": arguments.splitting_steps,
        "sampling_seed": arguments.sampling_seed,
        "monte_carlo_samples_per_control": samples,
        "sampling_workers": arguments.sampling_workers,
        "smoke": arguments.smoke,
    }
    prepare_resumable_output(
        output,
        experiment=EXPERIMENT_NAME,
        configuration=configuration,
    )

    for n_qubits in qubits:
        chain, problem = problems[n_qubits]
        coupling_links = coupling_links_by_size[n_qubits]
        coupling_pairs = tuple((index + 1, index + 2) for index in coupling_links)
        robustness = build_static_coupling_robustness(
            chain,
            order=1,
            standard_deviation_mhz=arguments.static_std_mhz,
            penalty_weight=2.0 * arguments.p1_coefficient,
            coupling_link_indices=coupling_links,
        )
        block_counts = tuple(
            component.n_blocks for component in build_augmented_hamiltonians(problem, robustness)
        )
        if "dense" in methods:
            augmented_dimension = block_counts[0] * problem.hilbert_dimension
            matrix_gib = 16 * augmented_dimension**2 / 2**30
            stored_gib = matrix_gib * (1 + problem.n_controls)
            print(
                f"Dense {n_qubits}q memory estimate: {matrix_gib:.2f} GiB per "
                f"augmented matrix, at least {stored_gib:.1f} GiB for drift "
                "and controls before temporary arrays.",
                flush=True,
            )
        if arguments.st_true_loss_mode == "final-only":
            check_interval = None
        elif arguments.st_true_loss_mode == "periodic":
            check_interval = arguments.exact_check_interval
        else:
            check_interval = (
                arguments.exact_check_interval if n_qubits <= periodic_max_qubits else None
            )
        for seed in seeds:
            case_directory = output / f"{n_qubits}q" / f"seed{seed}"
            case_methods = select_pending_methods(
                case_directory,
                methods,
                existing_result=arguments.existing_result,
                required_files=("result.npz", "metadata.json", "method_summary.json"),
            )
            if not case_methods:
                continue
            print(
                f"C: task={task.value}, n={n_qubits}, seed={seed}, "
                f"noise_pairs={coupling_pairs}, std={arguments.static_std_mhz:g} MHz, "
                f"blocks={block_counts}, exact_interval={check_interval}",
                flush=True,
            )
            if file_pulse is None:
                pulse, seed_path = load_initial_pulse(
                    problem,
                    n_qubits=n_qubits,
                    seed=seed,
                    source=initial_source,
                    generated_root=arguments.seed_root,
                )
            else:
                pulse, seed_path = file_pulse, arguments.initial_pulse
            metadata = {
                "experiment": EXPERIMENT_NAME,
                "task": task.value,
                "coupling": arguments.coupling,
                "n_qubits": n_qubits,
                "hilbert_dimension": problem.hilbert_dimension,
                "seed": seed,
                "robustness_order": 1,
                "taylor_sources": len(coupling_links),
                "static_coupling_pairs": coupling_pairs,
                "static_standard_deviation_mhz": arguments.static_std_mhz,
                "p1_coefficient": arguments.p1_coefficient,
                "augmented_block_counts": list(block_counts),
                "duration_us": problem.duration,
                "time_step_us": problem.time_step,
                "optimization_time_limit_hours": time_limit_hours,
                "periodic_true_loss_enabled": check_interval is not None,
                "st_true_loss_mode": arguments.st_true_loss_mode,
                "st_final_reference_outside_optimization_time": check_interval is None,
                "exact_check_interval": check_interval,
                "initial_source": initial_source,
                "initial_seed_path": None if seed_path is None else str(seed_path),
                "initial_pulse_path": (
                    str(arguments.initial_pulse.resolve())
                    if initial_source == "file" and arguments.initial_pulse is not None
                    else None
                ),
                "post_run_monte_carlo_samples": samples,
                "sampling_seed": arguments.sampling_seed,
                "baseline_trace_fields": ["fidelity_loss_history", "robustness_penalty_history"],
            }
            run_and_save_paper_methods(
                methods=case_methods,
                chain=chain,
                problem=problem,
                robustness=robustness,
                pulse=pulse,
                directory=case_directory,
                metadata=metadata,
                max_iterations=max_iterations,
                ode_method=arguments.ode_method,
                ode_rtol=arguments.ode_rtol,
                ode_atol=arguments.ode_atol,
                exact_check_interval=check_interval,
                exact_increase_patience=arguments.exact_increase_patience,
                exact_increase_absolute_tolerance=(arguments.exact_increase_absolute_tolerance),
                exact_increase_relative_tolerance=(arguments.exact_increase_relative_tolerance),
                time_limit_hours=time_limit_hours,
                log_interval=arguments.log_interval,
                n_samples=samples,
                sampling_seed=arguments.sampling_seed,
                workers=arguments.sampling_workers,
                splitting_steps=arguments.splitting_steps,
                max_splitting_steps=arguments.splitting_steps,
                defer_final_exact_evaluation=check_interval is None,
                collect_sensitivity_diagnostics=False,
                method_execution_order=tuple(
                    method for method in ("ST", "block-sparse", "dense") if method in case_methods
                ),
            )

    print(f"Saved stress-test raw outputs to {output}", flush=True)
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse fixed-budget, paired-seed, and numerical settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--existing-result",
        choices=("skip", "overwrite"),
        default="skip",
        help="skip completed seed/method results or replace them (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in ExperimentTask),
        default=ExperimentTask.GHZ.value,
    )
    parser.add_argument("--coupling", choices=("ZZ", "XX+YY"), default="XX+YY")
    parser.add_argument("--qubits", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument(
        "--static-coupling-pairs",
        nargs="+",
        type=_parse_static_pair,
        metavar="LEFT-RIGHT",
        help="one-based fluctuating nearest-neighbor bonds; default: 1-2 3-4 5-6 7-8 for 9 qubits",
    )
    parser.add_argument("--static-std-mhz", type=float, default=1.0)
    parser.add_argument(
        "--p1-coefficient",
        type=float,
        default=0.5,
        help="coefficient multiplying P1 squared; default 0.5 preserves the original stress-test objective",
    )
    parser.add_argument("--time-step", type=float)
    parser.add_argument(
        "--duration",
        type=float,
        help=(
            "total pulse duration in microseconds; generated inputs inherit it "
            "from E00 when omitted, while direct pulse inputs must match it"
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--initial-source",
        choices=("generated", "random"),
        default="generated",
    )
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument(
        "--initial-pulse",
        type=Path,
        help="optimized current-schema result.npz; selects file input instead of --seed-root",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Monte Carlo trajectories per saved control; zero disables sampling",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-workers",
        "--workers",
        dest="sampling_workers",
        type=int,
        default=1,
        help=(
            "processes used only for post-run Monte Carlo sampling; "
            "one keeps serial execution (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=("ST", "dense", "block-sparse", "all"),
        default=("all",),
    )
    parser.add_argument(
        "--st-true-loss-mode",
        choices=("final-only", "periodic", "auto"),
        default="final-only",
        help="evaluate ST true loss only after optimization, periodically, or by size",
    )
    parser.add_argument("--ode-method", choices=("dop853",), default="dop853")
    parser.add_argument("--ode-rtol", type=float)
    parser.add_argument("--ode-atol", type=float)
    parser.add_argument("--exact-check-interval", type=int, default=25)
    parser.add_argument("--periodic-true-loss-max-qubits", type=int)
    parser.add_argument("--exact-increase-patience", type=int, default=2)
    parser.add_argument("--exact-increase-absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--exact-increase-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--splitting-steps", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run from terminal arguments or an editable in-file argument tuple."""

    run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
