"""Experiment A: end-to-end GHZ/QFT optimization under heterogeneous noise.

For every paired nominal seed, this experiment compares ST-GRAPE, dense GRAPE,
and block-sparse GRAPE from the same pulse at Taylor robustness orders one and
two.  Quasistatic coupling/frequency noise, control-amplitude drift, and colored noise are
present simultaneously.  The script saves raw optimizer histories, controls,
metadata, and paired Monte Carlo fidelities.  Optimization uses the finite
exponential colored-noise model, while validation defaults to the unfitted
Gaussian reference covariance; analysis is notebook-only.

The default ``hybrid`` objective is used in the manuscript. The
optional ``rms`` objective uses physically scaled smooth RMS Taylor corrections
with independently configurable ``P1``, ``P2``, and colored mean-shift
coefficients. The ``hybrid`` objective instead squares ``P1`` and retains
independently smoothed RMS norms for ``P2`` and the colored mean shift.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.benchmarks.common import (
    AUTO_NOISE_QUBIT,
    COMPARISON_METHODS,
    HYBRID_P1_CONVENTION,
    ExperimentTask,
    build_multisource_problem,
    normalize_active_noise_weights,
    parse_optional_noise_pair,
    parse_optional_noise_qubit,
    resolve_optional_noise_pairs,
    resolve_optional_noise_qubit,
    resolve_optional_noise_qubits,
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
    run_and_save_paper_methods,
)
from stgrape.models import default_paper_duration
from stgrape.sampling import colored_noise_sampling_metadata

EXPERIMENT_NAME = "A heterogeneous-noise end-to-end optimization"
DEFAULT_ORDERS = (1, 2)
DEFAULT_SEEDS = tuple(range(10))


def _time_step(arguments: argparse.Namespace) -> float:
    """Resolve an explicit, generated-seed, or paper-default PWC step."""

    if arguments.time_step is not None:
        return arguments.time_step
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
        duration = default_paper_duration(task.control_task)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive.")
    return duration


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Run paired methods and save raw method-level data only."""

    task = ExperimentTask(arguments.task)
    methods = selected_methods(arguments.method)
    qubits = tuple(arguments.qubits) if arguments.qubits else (task.default_qubits,)
    orders = tuple(arguments.orders)
    seeds = tuple(arguments.seeds)
    time_step = 0.0025 if arguments.smoke else _time_step(arguments)
    max_iterations = arguments.max_iterations
    samples = arguments.samples
    initial_source = arguments.initial_source
    time_limit_hours = arguments.time_limit_hours
    coupling = arguments.coupling
    static_coupling_pairs, include_static_noise = resolve_optional_noise_pairs(
        arguments.static_coupling_pairs
    )
    frequency_drift_qubits, include_frequency_drift = resolve_optional_noise_qubits(
        arguments.frequency_drift_qubits
    )
    control_drift_qubits, include_control_drift = resolve_optional_noise_qubits(
        arguments.control_drift_qubits
    )
    non_markovian_selection = arguments.non_markovian_qubit
    static_std_mhz = arguments.static_std_mhz
    frequency_drift_std_mhz = arguments.frequency_drift_std_mhz
    control_drift_std = arguments.control_drift_std
    colored_std_mhz = arguments.colored_std_mhz
    penalty_mode = arguments.penalty_mode
    rms_smoothing = arguments.rms_smoothing
    p1_smoothing = arguments.p1_smoothing
    p2_smoothing = arguments.p2_smoothing
    pc_smoothing = arguments.pc_smoothing
    p1_coefficient = arguments.p1_coefficient
    p2_coefficient = arguments.p2_coefficient
    pc_coefficient = arguments.pc_coefficient
    first_order_penalty = arguments.first_order_penalty
    second_order_penalty = arguments.second_order_penalty
    coupling_weight = arguments.coupling_weight
    frequency_drift_weight = arguments.frequency_drift_weight
    control_drift_weight = arguments.control_drift_weight
    colored_weight = arguments.colored_weight
    if arguments.smoke:
        qubits = (2,)
        orders = (1,)
        seeds = (0,)
        max_iterations = 1
        samples = 2
        initial_source = "random"
        time_limit_hours = None
        if include_static_noise:
            static_coupling_pairs = None
        if include_frequency_drift:
            frequency_drift_qubits = None
        if include_control_drift:
            control_drift_qubits = None
        if non_markovian_selection is not None:
            non_markovian_selection = AUTO_NOISE_QUBIT

    duration = _duration(
        arguments,
        task=task,
        n_qubits=qubits[0],
        seed=seeds[0],
        initial_source=initial_source,
    )

    non_markovian_qubit, include_non_markovian_noise = resolve_optional_noise_qubit(
        non_markovian_selection
    )
    requested_group_weights = (
        coupling_weight,
        frequency_drift_weight,
        control_drift_weight,
        colored_weight,
    )
    (
        coupling_weight,
        frequency_drift_weight,
        control_drift_weight,
        colored_weight,
    ) = normalize_active_noise_weights(
        *requested_group_weights,
        include_static_noise=include_static_noise,
        include_frequency_drift=include_frequency_drift,
        include_control_drift=include_control_drift,
        include_non_markovian_noise=include_non_markovian_noise,
    )
    disabled_families = [
        name
        for name, enabled in (
            ("static", include_static_noise),
            ("frequency drift", include_frequency_drift),
            ("control drift", include_control_drift),
            ("non-Markovian", include_non_markovian_noise),
        )
        if not enabled
    ]
    if disabled_families:
        print(
            f"Disabled noise families: {', '.join(disabled_families)}; active "
            "weights were renormalized to "
            f"{coupling_weight:.6g}/{frequency_drift_weight:.6g}/"
            f"{control_drift_weight:.6g}/{colored_weight:.6g}.",
            flush=True,
        )

    if len(set(qubits)) != len(qubits) or any(value < 2 for value in qubits):
        raise ValueError("qubits must be unique integers of at least two.")
    if len(set(orders)) != len(orders) or any(value < 1 for value in orders):
        raise ValueError("orders must be unique positive integers.")
    if any(value > 1 for value in orders) and not (
        include_static_noise or include_frequency_drift or include_control_drift
    ):
        raise ValueError(
            "Robustness orders above one require static noise or control drift; "
            "colored noise uses a separate first-order Van Loan component."
        )
    if len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("seeds must be unique non-negative integers.")
    if arguments.workers < 1 or samples < 0:
        raise ValueError("workers must be positive and samples must be non-negative.")
    if arguments.splitting_steps < 1:
        raise ValueError("splitting steps must be positive.")
    if arguments.max_splitting_steps < arguments.splitting_steps:
        raise ValueError("max splitting steps cannot be smaller than splitting steps.")
    if arguments.splitting_relative_tolerance < 0.0 or arguments.splitting_absolute_tolerance < 0.0:
        raise ValueError("splitting tolerances must be non-negative.")
    if (
        arguments.splitting_relative_tolerance == 0.0
        and arguments.splitting_absolute_tolerance == 0.0
    ):
        raise ValueError("At least one splitting tolerance must be positive.")
    if first_order_penalty < 0.0 or second_order_penalty < 0.0:
        raise ValueError("All robustness penalty weights must be non-negative.")
    if (
        not np.isfinite(rms_smoothing)
        or rms_smoothing <= 0.0
        or any(
            value is not None and (not np.isfinite(value) or value <= 0.0)
            for value in (p1_smoothing, p2_smoothing, pc_smoothing)
        )
        or any(
            not np.isfinite(value) or value < 0.0
            for value in (p1_coefficient, p2_coefficient, pc_coefficient)
        )
    ):
        raise ValueError("Direct coefficients must be non-negative and smoothing must be positive.")
    if initial_source == "generated" and arguments.seed_root is None:
        raise ValueError("--seed-root is required for generated initial pulses.")
    if initial_source == "generated" and arguments.seed_root is not None:
        seed_coupling = generated_seed_coupling(arguments.seed_root)
        if seed_coupling != coupling:
            raise ValueError(
                f"Initial E00 pulses use coupling={seed_coupling!r}, but this "
                f"run requests coupling={coupling!r}. Generate a matching E00 "
                "seed root."
            )

    output = arguments.output or unique_output_directory("exp01_multisource", task)
    problems = {
        (n_qubits, order): build_multisource_problem(
            task,
            n_qubits,
            time_step=time_step,
            duration=duration,
            order=order,
            coupling=coupling,
            static_standard_deviation_mhz=static_std_mhz,
            frequency_drift_standard_deviation_mhz=frequency_drift_std_mhz,
            control_drift_standard_deviation=control_drift_std,
            colored_standard_deviation_mhz=colored_std_mhz,
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
            static_coupling_pairs=static_coupling_pairs,
            frequency_drift_qubits=frequency_drift_qubits,
            control_drift_qubits=control_drift_qubits,
            non_markovian_qubit=non_markovian_qubit,
            include_static_noise=include_static_noise,
            include_frequency_drift=include_frequency_drift,
            include_control_drift=include_control_drift,
            include_non_markovian_noise=include_non_markovian_noise,
        )
        for n_qubits in qubits
        for order in orders
    }
    preflight_initial_pulses(
        tuple(
            (problems[(n_qubits, orders[0])][1], n_qubits, seed)
            for n_qubits in qubits
            for seed in seeds
        ),
        source=initial_source,
        generated_root=arguments.seed_root,
    )
    configuration = {
        "task": task.value,
        "duration_us": duration,
        "time_step_us": time_step,
        "coupling": coupling,
        "static_noise": "independent nominal-coupling strength drifts",
        "static_std_mhz": static_std_mhz,
        "frequency_drift_std_mhz": frequency_drift_std_mhz,
        "control_drift_std": control_drift_std,
        "colored_std_mhz": colored_std_mhz,
        "requested_static_coupling_pairs": (
            None
            if static_coupling_pairs is None
            else [list(pair) for pair in static_coupling_pairs]
        ),
        "requested_control_drift_qubits": control_drift_qubits,
        "requested_frequency_drift_qubits": frequency_drift_qubits,
        "requested_non_markovian_qubit": non_markovian_qubit,
        "static_noise_enabled": include_static_noise,
        "frequency_drift_noise_enabled": include_frequency_drift,
        "control_drift_noise_enabled": include_control_drift,
        "non_markovian_noise_enabled": include_non_markovian_noise,
        "first_order_penalty": first_order_penalty,
        "second_order_penalty": second_order_penalty,
        "penalty_mode": penalty_mode,
        "hybrid_p1_convention": (HYBRID_P1_CONVENTION if penalty_mode == "hybrid" else None),
        "rms_smoothing": rms_smoothing,
        "p1_smoothing": p1_smoothing,
        "p2_smoothing": p2_smoothing,
        "pc_smoothing": pc_smoothing,
        "p1_coefficient": p1_coefficient,
        "p2_coefficient": p2_coefficient,
        "pc_coefficient": pc_coefficient,
        "effective_first_order_coefficient": (
            p1_coefficient if penalty_mode in {"rms", "hybrid"} else None
        ),
        "effective_second_order_coefficient": (
            p2_coefficient if penalty_mode in {"rms", "hybrid"} else None
        ),
        "effective_colored_coefficient": (
            pc_coefficient if penalty_mode in {"rms", "hybrid"} else None
        ),
        "coupling_weight": coupling_weight,
        "frequency_drift_weight": frequency_drift_weight,
        "control_drift_weight": control_drift_weight,
        "colored_weight": colored_weight,
        "requested_group_weights": list(requested_group_weights),
        "initial_source": initial_source,
        "initial_seed_root": (
            str(arguments.seed_root.resolve())
            if initial_source == "generated" and arguments.seed_root is not None
            else None
        ),
        "ode_method": arguments.ode_method,
        "ode_rtol": arguments.ode_rtol,
        "ode_atol": arguments.ode_atol,
        "max_iterations": max_iterations,
        "time_limit_hours": time_limit_hours,
        "exact_check_interval": arguments.exact_check_interval,
        "exact_increase_patience": arguments.exact_increase_patience,
        "exact_increase_absolute_tolerance": (arguments.exact_increase_absolute_tolerance),
        "exact_increase_relative_tolerance": (arguments.exact_increase_relative_tolerance),
        "splitting_steps": arguments.splitting_steps,
        "adaptive_splitting": arguments.adaptive_splitting,
        "max_splitting_steps": arguments.max_splitting_steps,
        "splitting_relative_tolerance": arguments.splitting_relative_tolerance,
        "splitting_absolute_tolerance": arguments.splitting_absolute_tolerance,
        "sampling_seed": arguments.sampling_seed,
        "monte_carlo_samples_per_control": samples,
        "sampling_workers": arguments.workers,
        "colored_noise_validation": colored_noise_sampling_metadata(
            next(iter(problems.values()))[2],
            arguments.colored_noise_sampling,
        ),
        "smoke": arguments.smoke,
    }
    prepare_resumable_output(
        output,
        experiment=EXPERIMENT_NAME,
        configuration=configuration,
    )

    for n_qubits in qubits:
        for order in orders:
            chain, problem, robustness, locations = problems[(n_qubits, order)]
            block_counts = tuple(
                component.n_blocks
                for component in build_augmented_hamiltonians(problem, robustness)
            )
            for seed in seeds:
                case_directory = output / f"{n_qubits}q" / f"r{order}" / f"seed{seed}"
                case_methods = select_pending_methods(
                    case_directory,
                    methods,
                    existing_result=arguments.existing_result,
                    required_files=("result.npz", "metadata.json", "method_summary.json"),
                )
                if not case_methods:
                    continue
                print(
                    f"A: task={task.value}, n={n_qubits}, r={order}, seed={seed}, "
                    f"blocks={block_counts}",
                    flush=True,
                )
                pulse, seed_path = load_initial_pulse(
                    problem,
                    n_qubits=n_qubits,
                    seed=seed,
                    source=initial_source,
                    generated_root=arguments.seed_root,
                )
                metadata = {
                    "experiment": EXPERIMENT_NAME,
                    "task": task.value,
                    "coupling": coupling,
                    "n_qubits": n_qubits,
                    "hilbert_dimension": problem.hilbert_dimension,
                    "seed": seed,
                    "robustness_order": order,
                    "taylor_sources": (
                        None if robustness.taylor is None else robustness.taylor.n_sources
                    ),
                    "augmented_block_counts": list(block_counts),
                    "static_coupling_pairs": [
                        list(pair) for pair in locations.static_coupling_pairs
                    ],
                    "coupling_drift_sources": len(locations.static_coupling_pairs),
                    "frequency_drift_qubits": list(locations.frequency_drift_qubits),
                    "frequency_drift_sources": len(locations.frequency_drift_qubits),
                    "quasistatic_sources": (
                        len(locations.static_coupling_pairs) + len(locations.frequency_drift_qubits)
                    ),
                    "control_drift_qubits": list(locations.control_drift_qubits),
                    "non_markovian_qubit": locations.non_markovian_qubit,
                    "static_noise_enabled": include_static_noise,
                    "frequency_drift_noise_enabled": include_frequency_drift,
                    "control_drift_noise_enabled": include_control_drift,
                    "non_markovian_noise_enabled": include_non_markovian_noise,
                    "static_std_mhz": static_std_mhz,
                    "frequency_drift_std_mhz": frequency_drift_std_mhz,
                    "control_drift_std": control_drift_std,
                    "colored_std_mhz": colored_std_mhz,
                    "first_order_penalty": first_order_penalty,
                    "second_order_penalty": second_order_penalty,
                    "penalty_mode": penalty_mode,
                    "hybrid_p1_convention": (
                        HYBRID_P1_CONVENTION if penalty_mode == "hybrid" else None
                    ),
                    "rms_smoothing": rms_smoothing,
                    "p1_smoothing": p1_smoothing,
                    "p2_smoothing": p2_smoothing,
                    "pc_smoothing": pc_smoothing,
                    "effective_p1_smoothing": (
                        robustness.resolved_p1_smoothing if penalty_mode == "rms" else None
                    ),
                    "effective_p2_smoothing": robustness.resolved_p2_smoothing,
                    "effective_pc_smoothing": robustness.resolved_pc_smoothing,
                    "p1_coefficient": p1_coefficient,
                    "p2_coefficient": p2_coefficient,
                    "pc_coefficient": pc_coefficient,
                    "effective_p1_coefficient": robustness.resolved_p1_coefficient,
                    "effective_p2_coefficient": robustness.resolved_p2_coefficient,
                    "effective_pc_coefficient": robustness.resolved_pc_coefficient,
                    "coupling_weight": coupling_weight,
                    "frequency_drift_weight": frequency_drift_weight,
                    "control_drift_weight": control_drift_weight,
                    "colored_weight": colored_weight,
                    "resolved_taylor_degree_multipliers": (
                        []
                        if robustness.taylor is None
                        else list(robustness.taylor.degree_penalty_weights)
                    ),
                    "duration_us": problem.duration,
                    "time_step_us": problem.time_step,
                    "initial_source": initial_source,
                    "initial_seed_path": None if seed_path is None else str(seed_path),
                    "monte_carlo_samples": samples,
                    "sampling_seed": arguments.sampling_seed,
                    "colored_noise_validation": colored_noise_sampling_metadata(
                        robustness,
                        arguments.colored_noise_sampling,
                    ),
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
                    exact_check_interval=arguments.exact_check_interval,
                    exact_increase_patience=arguments.exact_increase_patience,
                    exact_increase_absolute_tolerance=(arguments.exact_increase_absolute_tolerance),
                    exact_increase_relative_tolerance=(arguments.exact_increase_relative_tolerance),
                    time_limit_hours=time_limit_hours,
                    log_interval=arguments.log_interval,
                    n_samples=samples,
                    sampling_seed=arguments.sampling_seed,
                    workers=arguments.workers,
                    colored_noise_sampling=arguments.colored_noise_sampling,
                    splitting_steps=arguments.splitting_steps,
                    adaptive_splitting=arguments.adaptive_splitting,
                    max_splitting_steps=arguments.max_splitting_steps,
                    splitting_relative_tolerance=(arguments.splitting_relative_tolerance),
                    splitting_absolute_tolerance=(arguments.splitting_absolute_tolerance),
                )

    print(f"Saved experiment-A raw outputs to {output}", flush=True)
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse end-to-end comparison, uncertainty, and solver settings."""

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
    parser.add_argument(
        "--coupling",
        choices=("ZZ", "XX+YY"),
        default="XX+YY",
        help="nominal nearest-neighbour coupling (default: %(default)s)",
    )
    parser.add_argument("--qubits", nargs="+", type=int)
    parser.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--time-step", type=float)
    parser.add_argument(
        "--duration",
        type=float,
        help=(
            "total pulse duration in microseconds; generated inputs inherit it "
            "from E00 when omitted"
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--initial-source",
        choices=("generated", "random"),
        default="generated",
    )
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument(
        "--static-coupling-pairs",
        nargs="+",
        type=parse_optional_noise_pair,
        default=AUTO_NOISE_QUBIT,
        metavar="LEFT-RIGHT|None",
        help=(
            "one-based links carrying independent nominal-coupling drift; "
            "use None to remove this family (default: all nearest-neighbour links)"
        ),
    )
    parser.add_argument(
        "--frequency-drift-qubits",
        nargs="+",
        type=parse_optional_noise_qubit,
        default=AUTO_NOISE_QUBIT,
        metavar="QUBIT|None",
        help=(
            "one-based qubits carrying independent quasistatic Z-frequency "
            "drift; use None to remove it (default: all qubits)"
        ),
    )
    parser.add_argument(
        "--control-drift-qubits",
        nargs="+",
        type=parse_optional_noise_qubit,
        default=AUTO_NOISE_QUBIT,
        metavar="QUBIT|None",
        help=(
            "one-based qubits whose X/Y controls carry independent amplitude drift; "
            "use None to remove this family (default: the central two qubits)"
        ),
    )
    parser.add_argument(
        "--non-markovian-qubit",
        type=parse_optional_noise_qubit,
        default=AUTO_NOISE_QUBIT,
        metavar="QUBIT|None",
        help=(
            "one-based colored-noise qubit; use None to remove this noise "
            "family (default: auto, the final qubit)"
        ),
    )
    parser.add_argument(
        "--static-std-mhz",
        type=float,
        default=1.0,
        help=("standard deviation of each coupling-strength drift in MHz (default: %(default)s)"),
    )
    parser.add_argument(
        "--frequency-drift-std-mhz",
        type=float,
        default=1.0,
        help="standard deviation of each local-Z frequency drift in MHz",
    )
    parser.add_argument(
        "--control-drift-std",
        type=float,
        default=0.01,
        help="relative control-amplitude standard deviation (default: 1%%)",
    )
    parser.add_argument(
        "--colored-std-mhz",
        type=float,
        default=1.0,
        help=(
            "RMS strength of the colored frequency noise in MHz, independent "
            "of its correlation shape (default: %(default)s)"
        ),
    )
    parser.add_argument("--first-order-penalty", type=float, default=1.0)
    parser.add_argument("--second-order-penalty", type=float, default=1.0)
    parser.add_argument(
        "--penalty-mode",
        choices=("squared", "rms", "hybrid"),
        default="hybrid",
        help=(
            "squared uses blockwise squared norms; rms uses smooth RMS norms; "
            "hybrid uses squared P1 with smooth-RMS P2 and Pc"
        ),
    )
    parser.add_argument(
        "--rms-smoothing",
        type=float,
        default=1e-12,
        help="shared fallback smoothing scale for RMS norms",
    )
    parser.add_argument(
        "--p1-smoothing",
        type=float,
        help=(
            "P1 smoothing scale in RMS mode; defaults to --rms-smoothing "
            "(hybrid P1 is squared and does not use smoothing)"
        ),
    )
    parser.add_argument(
        "--p2-smoothing",
        type=float,
        help="P2 smoothing scale; defaults to --rms-smoothing",
    )
    parser.add_argument(
        "--pc-smoothing",
        type=float,
        help="colored P_c smoothing scale; defaults to --rms-smoothing",
    )
    parser.add_argument(
        "--p1-coefficient",
        type=float,
        default=1.0,
        help="direct P1 coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--p2-coefficient",
        type=float,
        default=1.0,
        help="direct P2 coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--pc-coefficient",
        type=float,
        default=1.0,
        help="direct colored mean-shift coefficient (default: %(default)s)",
    )
    parser.add_argument("--coupling-weight", type=float, default=1.0 / 4.0)
    parser.add_argument("--frequency-drift-weight", type=float, default=1.0 / 4.0)
    parser.add_argument("--control-drift-weight", type=float, default=1.0 / 4.0)
    parser.add_argument("--colored-weight", type=float, default=1.0 / 4.0)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--colored-noise-sampling",
        choices=("reference-gaussian", "rtp"),
        default="reference-gaussian",
        help=(
            "reference-gaussian validates against the unfitted 1/f covariance; "
            "rtp reproduces the finite exponential surrogate"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--method",
        nargs="+",
        choices=(*COMPARISON_METHODS, "all"),
        default=("all",),
    )
    parser.add_argument("--ode-method", choices=("dop853",), default="dop853")
    parser.add_argument("--ode-rtol", type=float)
    parser.add_argument("--ode-atol", type=float)
    parser.add_argument("--exact-check-interval", type=int, default=25)
    parser.add_argument("--exact-increase-patience", type=int, default=2)
    parser.add_argument("--exact-increase-absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--exact-increase-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--splitting-steps", type=int, default=1)
    parser.add_argument("--adaptive-splitting", action="store_true")
    parser.add_argument("--max-splitting-steps", type=int, default=1)
    parser.add_argument(
        "--splitting-relative-tolerance",
        type=float,
        default=0.15,
        help="relative part of the true/surrogate mixed-error tolerance",
    )
    parser.add_argument(
        "--splitting-absolute-tolerance",
        type=float,
        default=2e-5,
        help="absolute part of the true/surrogate mixed-error tolerance",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run from terminal arguments or an editable in-file argument tuple."""

    run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
