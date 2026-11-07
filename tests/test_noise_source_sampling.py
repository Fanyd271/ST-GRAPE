"""Tests for source-resolved Monte Carlo validation."""

from __future__ import annotations

import numpy as np

from stgrape.benchmarks.common import ExperimentTask, build_multisource_problem
from stgrape.sampling import (
    fidelity_statistics,
    infidelity_statistics,
    sample_fidelities_by_noise_source,
    separate_noise_models,
)


def test_multisource_model_separates_and_samples_each_noise_family() -> None:
    _, problem, robustness, _ = build_multisource_problem(
        ExperimentTask.QFT,
        2,
        time_step=0.025,
        order=1,
        static_coupling_pairs=((1, 2),),
        frequency_drift_qubits=(1, 2),
        control_drift_qubits=(1,),
        non_markovian_qubit=2,
    )
    models = separate_noise_models(robustness, static_group_sizes=(1, 2))
    assert tuple(models) == ("coupling", "frequency", "control", "colored")
    assert len(models["coupling"].taylor.static_operators) == 1  # type: ignore[union-attr]
    assert len(models["frequency"].taylor.static_operators) == 2  # type: ignore[union-attr]
    assert len(models["control"].taylor.static_operators) == 0  # type: ignore[union-attr]
    assert len(models["control"].taylor.control_drifts) == 2  # type: ignore[union-attr]
    assert len(models["colored"].colored) == 1

    controls = np.zeros((problem.n_controls, problem.n_time_steps))
    samples = sample_fidelities_by_noise_source(
        problem,
        controls,
        robustness,
        n_samples=2,
        seed=17,
        workers=2,
        static_group_sizes=(1, 2),
    )
    assert tuple(samples) == ("coupling", "frequency", "control", "colored")
    assert all(values.shape == (2,) for values in samples.values())
    assert all(np.all((values >= 0.0) & (values <= 1.0)) for values in samples.values())

    summary = fidelity_statistics(samples["coupling"])
    assert summary["samples"] == 2
    assert summary["minimum_fidelity"] <= summary["mean_fidelity"]
    assert summary["mean_fidelity"] <= summary["maximum_fidelity"]

    error_summary = infidelity_statistics(samples["coupling"])
    assert error_summary["samples"] == 2
    assert error_summary["minimum_error"] <= error_summary["mean_error"]
    assert error_summary["mean_error"] <= error_summary["maximum_error"]
