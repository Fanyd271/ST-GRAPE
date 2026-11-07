"""Experiment B1: objective/gradient scaling with augmented block count.

The Hilbert dimension, PWC grid, Taylor order, and splitting precision are
fixed.  The number of independent quasistatic sources is varied, giving
``B = binomial(m + r, r)`` augmented Taylor blocks.  This is a controlled
computational benchmark; it does not optimize a pulse or produce figures.
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
    coupling_matched_catalog_labels,
    measurements_to_arrays,
    pauli_catalog_labels,
    prepare_raw_output,
    random_controls,
    save_method_benchmark_data,
    select_pending_methods,
    selected_methods,
    unique_output_directory,
)

EXPERIMENT_NAME = "B1 augmented-block scaling"
DEFAULT_SOURCE_COUNTS = (1, 2, 3, 4, 6, 8, 10, 12)
DEFAULT_DENSE_MAX_BLOCKS = 190


def run_experiment(arguments: argparse.Namespace) -> Path:
    """Collect raw fixed-order timing data for every requested source count."""

    task = ExperimentTask(arguments.task)
    methods = selected_methods(arguments.method)
    source_counts = tuple(arguments.source_counts)
    n_qubits = arguments.qubits
    order = arguments.order
    time_step = arguments.time_step
    n_time_steps = arguments.time_steps
    repeats = arguments.repeats
    warmups = arguments.warmups
    if arguments.smoke:
        source_counts = (1, 2)
        n_qubits = 2
        time_step = 0.0025
        n_time_steps = 2
        repeats = 1
        warmups = 0

    if (
        not source_counts
        or len(set(source_counts)) != len(source_counts)
        or any(count < 1 for count in source_counts)
    ):
        raise ValueError("source counts must be unique positive integers.")
    if order < 1 or n_qubits < 2 or n_time_steps < 1:
        raise ValueError("order, qubits, and time steps must be positive and valid.")
    if arguments.dense_max_blocks < 1:
        raise ValueError("dense max blocks must be a positive integer.")
    if arguments.pauli_catalog_seed < 0:
        raise ValueError("pauli catalog seed must be non-negative.")
    if "dense" in methods and not any(
        augmented_block_count(source_count, order) <= arguments.dense_max_blocks
        for source_count in source_counts
    ):
        raise ValueError(
            "No requested point is eligible for dense GRAPE: every augmented-block "
            f"count exceeds --dense-max-blocks={arguments.dense_max_blocks}."
        )

    source_family = arguments.source_family
    if source_family == "pauli":
        catalog_labels = pauli_catalog_labels(
            n_qubits,
            seed=arguments.pauli_catalog_seed,
        )
        catalog_seed: int | None = arguments.pauli_catalog_seed
    else:
        catalog_labels = coupling_matched_catalog_labels(
            n_qubits,
            coupling=arguments.coupling,
        )
        catalog_seed = None
    if max(source_counts) > len(catalog_labels):
        raise ValueError(
            f"source count {max(source_counts)} exceeds the {source_family!r} "
            f"catalog size {len(catalog_labels)} for n={n_qubits}."
        )

    output = arguments.output or unique_output_directory("scaling_blocks", task)
    configuration = {
        "task": task.value,
        "coupling": arguments.coupling,
        "n_qubits": n_qubits,
        "hilbert_dimension": 2**n_qubits,
        "robustness_order": order,
        "source_counts": list(source_counts),
        "time_step_us": time_step,
        "n_time_steps": n_time_steps,
        "splitting_steps": arguments.splitting_steps,
        "ode_method": arguments.ode_method,
        "ode_rtol": arguments.ode_rtol,
        "ode_atol": arguments.ode_atol,
        "warmups": warmups,
        "repeats": repeats,
        "control_seed": arguments.control_seed,
        "source_family": source_family,
        "source_catalog_size": len(catalog_labels),
        "pauli_catalog_seed": catalog_seed,
        "source_operator_labels": list(catalog_labels[: max(source_counts)]),
        "dense_max_blocks": arguments.dense_max_blocks,
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
        print(f"All requested B1 method results already exist in {output}", flush=True)
        return output

    measurements_by_method = {method: [] for method in methods}
    source_counts_by_method = {method: [] for method in methods}
    block_counts_by_method = {method: [] for method in methods}
    nonzero_blocks_by_method = {method: [] for method in methods}
    shared_controls = None
    for source_count in source_counts:
        chain, problem, robustness = build_quasistatic_problem(
            task,
            n_qubits,
            time_step=time_step,
            n_time_steps=n_time_steps,
            source_count=source_count,
            order=order,
            source_family=source_family,
            pauli_catalog_seed=arguments.pauli_catalog_seed,
            coupling=arguments.coupling,
        )
        if shared_controls is None:
            shared_controls = random_controls(problem, seed=arguments.control_seed)
        expected_blocks = augmented_block_count(source_count, order)
        actual_blocks, stored_blocks = block_structure(problem, robustness)
        if actual_blocks != expected_blocks:
            raise RuntimeError(
                f"Expected {expected_blocks} Taylor blocks, constructed {actual_blocks}."
            )
        print(
            f"B1: task={task.value}, m={source_count}, r={order}, B={actual_blocks}",
            flush=True,
        )
        for method in methods:
            if method == "dense" and actual_blocks > arguments.dense_max_blocks:
                print(
                    "  skipping dense: "
                    f"B={actual_blocks} exceeds --dense-max-blocks "
                    f"{arguments.dense_max_blocks}",
                    flush=True,
                )
                continue
            print(f"  benchmarking {method}", flush=True)
            measurements_by_method[method].append(
                benchmark_method(
                    method,
                    chain=chain,
                    problem=problem,
                    robustness=robustness,
                    controls=shared_controls,
                    ode_method=arguments.ode_method,
                    ode_rtol=arguments.ode_rtol,
                    ode_atol=arguments.ode_atol,
                    splitting_steps=arguments.splitting_steps,
                    warmups=warmups,
                    repeats=repeats,
                )
            )
            source_counts_by_method[method].append(source_count)
            block_counts_by_method[method].append(actual_blocks)
            nonzero_blocks_by_method[method].append(stored_blocks)

    for method in methods:
        measurements = measurements_by_method[method]
        if not measurements:
            print(
                f"No {method} points satisfy the requested benchmark limits; "
                "no method result was saved.",
                flush=True,
            )
            continue
        arrays = measurements_to_arrays([[measurement] for measurement in measurements])
        save_method_benchmark_data(
            output,
            methods=(method,),
            shared_arrays={
                "source_counts": np.asarray(
                    source_counts_by_method[method],
                    dtype=np.int64,
                ),
                "block_counts": np.asarray(
                    block_counts_by_method[method],
                    dtype=np.int64,
                ),
                "nonzero_operator_blocks": np.asarray(
                    nonzero_blocks_by_method[method],
                    dtype=np.int64,
                ),
                "n_qubits": n_qubits,
                "hilbert_dimension": 2**n_qubits,
                "robustness_order": order,
                "time_step_us": time_step,
                "n_time_steps": n_time_steps,
                "splitting_steps": arguments.splitting_steps,
                "source_family": source_family,
                "pauli_catalog_seed": -1 if catalog_seed is None else catalog_seed,
                "dense_max_blocks": arguments.dense_max_blocks,
                "pre_measurement_cleanup": BENCHMARK_RESET_PROTOCOL,
                "repeats": repeats,
                "warmups": warmups,
            },
            measurement_arrays=arrays,
        )
    print(f"Saved B1 raw data to {output}", flush=True)
    return output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-``r`` variable-source block-scaling configuration."""

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
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument(
        "--source-family",
        choices=("coupling-matched", "pauli"),
        default="coupling-matched",
        help=(
            "coupling-matched uses local and pair terms (m <= n(n+5)/2); "
            "pauli uses a seeded ordering of all 4**n - 1 non-identity strings"
        ),
    )
    parser.add_argument(
        "--pauli-catalog-seed",
        type=int,
        default=42,
        help="seed for the nested Pauli-string ordering (default: %(default)s)",
    )
    parser.add_argument(
        "--source-counts",
        nargs="+",
        type=int,
        default=DEFAULT_SOURCE_COUNTS,
    )
    parser.add_argument(
        "--dense-max-blocks",
        type=int,
        default=DEFAULT_DENSE_MAX_BLOCKS,
        help=(
            "skip dense-GRAPE points above this augmented-block count; "
            "other methods keep their full source range (default: %(default)s)"
        ),
    )
    parser.add_argument("--order", type=int, default=2)
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
