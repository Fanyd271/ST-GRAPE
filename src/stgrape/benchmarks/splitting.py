"""Experiment B3: cost/accuracy ablation of ST-GRAPE splitting steps.

The physical problem, Taylor block count, PWC controls, and ODE reference are
fixed while ``q`` is varied.  The script saves raw timings, loss discrepancies,
and gradient discrepancies; the companion notebook performs all aggregation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stgrape.benchmarks.common import (
    BENCHMARK_RESET_PROTOCOL,
    ExperimentTask,
    augmented_block_count,
    benchmark_method,
    block_structure,
    build_quasistatic_problem,
    evaluate_method_once,
    prepare_raw_output,
    quasistatic_catalog_labels,
    random_controls,
    should_run_raw_result,
    unique_output_directory,
)

EXPERIMENT_NAME = "B3 ST-GRAPE splitting ablation"
DEFAULT_SPLITTING_STEPS = (1, 2, 4)


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Collect raw ST timing and ODE-reference accuracy measurements."""

    task = ExperimentTask(arguments.task)
    splitting_steps = tuple(arguments.splitting_steps)
    n_qubits = arguments.qubits
    source_count = arguments.source_count
    order = arguments.order
    time_step = arguments.time_step
    n_time_steps = arguments.time_steps
    repeats = arguments.repeats
    warmups = arguments.warmups
    if arguments.smoke:
        splitting_steps = (1, 2)
        n_qubits = 2
        source_count = 1
        order = 1
        time_step = 0.0025
        n_time_steps = 2
        repeats = 1
        warmups = 0
    if len(set(splitting_steps)) != len(splitting_steps) or any(
        step < 1 for step in splitting_steps
    ):
        raise ValueError("splitting steps must be unique positive integers.")
    if arguments.pauli_catalog_seed < 0:
        raise ValueError("pauli catalog seed must be non-negative.")

    chain, problem, robustness = build_quasistatic_problem(
        task,
        n_qubits,
        time_step=time_step,
        n_time_steps=n_time_steps,
        source_count=source_count,
        order=order,
        source_family=arguments.source_family,
        pauli_catalog_seed=arguments.pauli_catalog_seed,
        coupling=arguments.coupling,
    )
    controls = random_controls(problem, seed=arguments.control_seed)
    expected_blocks = augmented_block_count(source_count, order)
    actual_blocks, nonzero_blocks = block_structure(problem, robustness)
    if actual_blocks != expected_blocks:
        raise RuntimeError(f"Expected B={expected_blocks}, constructed B={actual_blocks}.")
    catalog_labels = quasistatic_catalog_labels(
        chain,
        source_family=arguments.source_family,
        pauli_catalog_seed=arguments.pauli_catalog_seed,
    )

    output = arguments.output or unique_output_directory("splitting", task)
    configuration = {
        "task": task.value,
        "coupling": arguments.coupling,
        "n_qubits": n_qubits,
        "hilbert_dimension": 2**n_qubits,
        "source_count": source_count,
        "source_family": arguments.source_family,
        "source_catalog_size": len(catalog_labels),
        "pauli_catalog_seed": (
            arguments.pauli_catalog_seed if arguments.source_family == "pauli" else None
        ),
        "source_operator_labels": list(catalog_labels[:source_count]),
        "robustness_order": order,
        "augmented_blocks": actual_blocks,
        "splitting_steps": list(splitting_steps),
        "time_step_us": time_step,
        "n_time_steps": n_time_steps,
        "reference_method": "block-sparse",
        "ode_method": arguments.ode_method,
        "ode_rtol": arguments.ode_rtol,
        "ode_atol": arguments.ode_atol,
        "warmups": warmups,
        "repeats": repeats,
        "control_seed": arguments.control_seed,
        "pre_measurement_cleanup": BENCHMARK_RESET_PROTOCOL,
        "smoke": arguments.smoke,
    }
    prepare_raw_output(output, experiment=EXPERIMENT_NAME, configuration=configuration)
    raw_path = output / "raw_data.npz"
    if not should_run_raw_result(
        raw_path,
        existing_result=arguments.existing_result,
    ):
        return output

    print("B3: evaluating block-sparse ODE reference", flush=True)
    reference_measurement = benchmark_method(
        "block-sparse",
        chain=chain,
        problem=problem,
        robustness=robustness,
        controls=controls,
        ode_method=arguments.ode_method,
        ode_rtol=arguments.ode_rtol,
        ode_atol=arguments.ode_atol,
        splitting_steps=1,
        warmups=warmups,
        repeats=repeats,
    )
    reference_loss, reference_gradient, reference_rhs = evaluate_method_once(
        "block-sparse",
        chain=chain,
        problem=problem,
        robustness=robustness,
        controls=controls,
        ode_method=arguments.ode_method,
        ode_rtol=arguments.ode_rtol,
        ode_atol=arguments.ode_atol,
    )

    st_measurements = []
    st_losses: list[float] = []
    loss_absolute_errors: list[float] = []
    loss_relative_errors: list[float] = []
    gradient_relative_errors: list[float] = []
    reference_scale = max(abs(reference_loss), np.finfo(float).tiny)
    gradient_scale = max(float(np.linalg.norm(reference_gradient)), np.finfo(float).tiny)
    for steps in splitting_steps:
        print(f"B3: benchmarking ST with q={steps}", flush=True)
        st_measurements.append(
            benchmark_method(
                "ST",
                chain=chain,
                problem=problem,
                robustness=robustness,
                controls=controls,
                ode_method=arguments.ode_method,
                ode_rtol=arguments.ode_rtol,
                ode_atol=arguments.ode_atol,
                splitting_steps=steps,
                warmups=warmups,
                repeats=repeats,
            )
        )
        loss, gradient, _ = evaluate_method_once(
            "ST",
            chain=chain,
            problem=problem,
            robustness=robustness,
            controls=controls,
            ode_method=arguments.ode_method,
            ode_rtol=arguments.ode_rtol,
            ode_atol=arguments.ode_atol,
            splitting_steps=steps,
        )
        absolute_error = float(abs(loss - reference_loss))
        st_losses.append(loss)
        loss_absolute_errors.append(absolute_error)
        loss_relative_errors.append(float(absolute_error / reference_scale))
        gradient_relative_errors.append(
            float(float(np.linalg.norm(gradient - reference_gradient)) / gradient_scale)
        )

    np.savez_compressed(
        raw_path,
        splitting_steps=np.asarray(splitting_steps, dtype=np.int64),
        n_qubits=n_qubits,
        hilbert_dimension=2**n_qubits,
        source_count=source_count,
        source_family=arguments.source_family,
        pauli_catalog_seed=(
            arguments.pauli_catalog_seed if arguments.source_family == "pauli" else -1
        ),
        robustness_order=order,
        block_count=actual_blocks,
        nonzero_operator_blocks=nonzero_blocks,
        time_step_us=time_step,
        n_time_steps=n_time_steps,
        repeats=repeats,
        warmups=warmups,
        pre_measurement_cleanup=BENCHMARK_RESET_PROTOCOL,
        st_build_times_s=np.asarray([measurement.build_time_s for measurement in st_measurements]),
        st_forward_times_s=np.asarray(
            [measurement.forward_times_s for measurement in st_measurements]
        ),
        st_objective_gradient_times_s=np.asarray(
            [measurement.objective_gradient_times_s for measurement in st_measurements]
        ),
        st_persistent_operator_bytes=np.asarray(
            [measurement.persistent_operator_bytes for measurement in st_measurements],
            dtype=np.int64,
        ),
        st_peak_traced_allocation_bytes=np.asarray(
            [measurement.peak_traced_allocation_bytes for measurement in st_measurements],
            dtype=np.int64,
        ),
        st_losses=np.asarray(st_losses),
        reference_loss=reference_loss,
        loss_absolute_errors=np.asarray(loss_absolute_errors),
        loss_relative_errors=np.asarray(loss_relative_errors),
        gradient_relative_errors=np.asarray(gradient_relative_errors),
        reference_build_time_s=reference_measurement.build_time_s,
        reference_forward_times_s=reference_measurement.forward_times_s,
        reference_objective_gradient_times_s=(reference_measurement.objective_gradient_times_s),
        reference_persistent_operator_bytes=(reference_measurement.persistent_operator_bytes),
        reference_peak_traced_allocation_bytes=(reference_measurement.peak_traced_allocation_bytes),
        reference_rhs_evaluations=reference_rhs,
    )
    print(f"Saved B3 raw data to {output}", flush=True)
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-problem splitting ablation configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--existing-result",
        choices=("skip", "overwrite"),
        default="skip",
        help="skip the completed ablation result or replace it (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in ExperimentTask),
        default=ExperimentTask.GHZ.value,
    )
    parser.add_argument("--coupling", choices=("ZZ", "XX+YY"), default="XX+YY")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--source-count", type=int, default=3)
    parser.add_argument(
        "--source-family",
        choices=("coupling-matched", "pauli"),
        default="coupling-matched",
    )
    parser.add_argument("--pauli-catalog-seed", type=int, default=42)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument(
        "--splitting-steps",
        nargs="+",
        type=int,
        default=DEFAULT_SPLITTING_STEPS,
    )
    parser.add_argument("--time-step", type=float, default=0.0005)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument("--ode-method", choices=("dop853",), default="dop853")
    parser.add_argument("--ode-rtol", type=float)
    parser.add_argument("--ode-atol", type=float)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run from terminal arguments or an editable in-file argument tuple."""

    run_experiment(parse_arguments(arguments))


if __name__ == "__main__":
    main()
