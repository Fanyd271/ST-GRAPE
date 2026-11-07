"""Experiment B2: objective/gradient scaling with Hilbert dimension.

Two local quasistatic ``Z`` sources and Taylor order one keep the augmented
block count fixed at ``B=3`` while the qubit count is increased.  Only raw
timings and memory measurements are saved; analysis lives in the notebook.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stgrape.benchmarks.common import (
    BENCHMARK_RESET_PROTOCOL,
    COMPARISON_METHODS,
    ExperimentTask,
    augmented_block_count,
    benchmark_method,
    block_structure,
    build_quasistatic_problem,
    measurements_to_arrays,
    prepare_raw_output,
    random_controls,
    save_method_benchmark_data,
    select_pending_methods,
    selected_methods,
    unique_output_directory,
)

EXPERIMENT_NAME = "B2 Hilbert-dimension scaling"
DEFAULT_GHZ_QUBITS = tuple(range(3, 11))
DEFAULT_QFT_QUBITS = tuple(range(2, 7))


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Collect raw timing data while holding the Taylor block count fixed."""

    task = ExperimentTask(arguments.task)
    methods = selected_methods(arguments.method)
    qubits = (
        tuple(arguments.qubits)
        if arguments.qubits is not None
        else (DEFAULT_GHZ_QUBITS if task == ExperimentTask.GHZ else DEFAULT_QFT_QUBITS)
    )
    time_step = arguments.time_step
    n_time_steps = arguments.time_steps
    repeats = arguments.repeats
    warmups = arguments.warmups
    if arguments.smoke:
        qubits = (2, 3)
        time_step = 0.0025
        n_time_steps = 2
        repeats = 1
        warmups = 0
    if len(set(qubits)) != len(qubits) or any(n_qubits < 2 for n_qubits in qubits):
        raise ValueError("qubits must be unique integers of at least two.")

    source_count = 2
    order = 1
    fixed_blocks = augmented_block_count(source_count, order)
    output = arguments.output or unique_output_directory("scaling_dimension", task)
    configuration = {
        "task": task.value,
        "coupling": arguments.coupling,
        "qubits": list(qubits),
        "source_count": source_count,
        "source_family": "local-z",
        "robustness_order": order,
        "augmented_blocks": fixed_blocks,
        "time_step_us": time_step,
        "n_time_steps": n_time_steps,
        "splitting_steps": arguments.splitting_steps,
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
    methods = select_pending_methods(
        output,
        methods,
        existing_result=arguments.existing_result,
        required_files=("raw_data.npz",),
    )
    if not methods:
        print(f"All requested B2 method results already exist in {output}", flush=True)
        return output

    measurements = []
    nonzero_blocks: list[int] = []
    for n_qubits in qubits:
        chain, problem, robustness = build_quasistatic_problem(
            task,
            n_qubits,
            time_step=time_step,
            n_time_steps=n_time_steps,
            source_count=source_count,
            order=order,
            source_family="local-z",
            coupling=arguments.coupling,
        )
        actual_blocks, stored_blocks = block_structure(problem, robustness)
        if actual_blocks != fixed_blocks:
            raise RuntimeError(f"Expected B={fixed_blocks}, constructed B={actual_blocks}.")
        nonzero_blocks.append(stored_blocks)
        controls = random_controls(problem, seed=arguments.control_seed + n_qubits)
        print(
            f"B2: task={task.value}, n={n_qubits}, d={2**n_qubits}, B={actual_blocks}",
            flush=True,
        )
        row = []
        for method in methods:
            print(f"  benchmarking {method}", flush=True)
            row.append(
                benchmark_method(
                    method,
                    chain=chain,
                    problem=problem,
                    robustness=robustness,
                    controls=controls,
                    ode_method=arguments.ode_method,
                    ode_rtol=arguments.ode_rtol,
                    ode_atol=arguments.ode_atol,
                    splitting_steps=arguments.splitting_steps,
                    warmups=warmups,
                    repeats=repeats,
                )
            )
        measurements.append(row)

    arrays = measurements_to_arrays(measurements)
    save_method_benchmark_data(
        output,
        methods=methods,
        shared_arrays={
            "n_qubits": np.asarray(qubits, dtype=np.int64),
            "hilbert_dimensions": 2 ** np.asarray(qubits, dtype=np.int64),
            "source_count": source_count,
            "robustness_order": order,
            "block_count": fixed_blocks,
            "nonzero_operator_blocks": np.asarray(nonzero_blocks, dtype=np.int64),
            "time_step_us": time_step,
            "n_time_steps": n_time_steps,
            "splitting_steps": arguments.splitting_steps,
            "pre_measurement_cleanup": BENCHMARK_RESET_PROTOCOL,
            "repeats": repeats,
            "warmups": warmups,
        },
        measurement_arrays=arrays,
    )
    print(f"Saved B2 raw data to {output}", flush=True)
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-block variable-dimension benchmark configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--existing-result",
        choices=("skip", "overwrite"),
        default="skip",
        help="skip completed method benchmarks or replace them (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in ExperimentTask),
        default=ExperimentTask.GHZ.value,
    )
    parser.add_argument("--coupling", choices=("ZZ", "XX+YY"), default="XX+YY")
    parser.add_argument("--qubits", nargs="+", type=int)
    parser.add_argument("--time-step", type=float, default=0.0005)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--splitting-steps", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument(
        "--method",
        nargs="+",
        choices=(*COMPARISON_METHODS, "all"),
        default=("all",),
    )
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
