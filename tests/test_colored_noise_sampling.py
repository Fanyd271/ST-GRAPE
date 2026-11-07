"""Validation tests for independent colored-noise trajectory generators."""

from __future__ import annotations

import numpy as np
import pytest

from stgrape.benchmarks.common import ExperimentTask, build_multisource_problem
from stgrape.models import (
    MULTISOURCE_GHZ_CORRELATION_WEIGHTS,
    MULTISOURCE_GHZ_DECAY_RATES,
    MULTISOURCE_QFT_CORRELATION_WEIGHTS,
    MULTISOURCE_QFT_DECAY_RATES,
    ONE_OVER_F_REFERENCE_CORRELATION,
)
from stgrape.problem import SmoothOneOverFCorrelation
from stgrape.sampling import (
    colored_noise_sampling_metadata,
    generate_noise_realizations,
    prepare_gaussian_circulant_embedding,
)


def test_smooth_one_over_f_correlation_has_expected_variance() -> None:
    correlation = SmoothOneOverFCorrelation(0.2, 20.0)
    lags = np.array([0.0, 0.1, 0.5])
    covariance = correlation.covariance(lags)

    assert covariance[0] == pytest.approx(2.0 * np.log(100.0))
    assert np.all(np.diff(covariance) < 0.0)
    assert correlation.metadata()["model"] == "smooth-one-over-f"


def test_circulant_embedding_reconstructs_requested_grid_covariance() -> None:
    correlation = SmoothOneOverFCorrelation(0.2, 20.0)
    n_time_steps = 7
    time_step = 0.05
    embedding = prepare_gaussian_circulant_embedding(
        correlation,
        n_time_steps=n_time_steps,
        time_step=time_step,
    )

    embedded_covariance = np.fft.ifft(embedding.square_root_eigenvalues**2).real
    indices = np.arange(n_time_steps)
    expected = correlation.covariance(indices * time_step)
    np.testing.assert_allclose(
        embedded_covariance[:n_time_steps],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    first = embedding.sample(np.random.default_rng(9))
    second = embedding.sample(np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (n_time_steps,)


@pytest.mark.parametrize(
    ("duration", "weights", "rates"),
    (
        (
            0.030,
            MULTISOURCE_GHZ_CORRELATION_WEIGHTS,
            MULTISOURCE_GHZ_DECAY_RATES,
        ),
        (
            0.100,
            MULTISOURCE_QFT_CORRELATION_WEIGHTS,
            MULTISOURCE_QFT_DECAY_RATES,
        ),
    ),
)
def test_paper_rtp_fit_matches_reference_covariance_over_each_pulse(
    duration: float,
    weights: tuple[float, ...],
    rates: tuple[float, ...],
) -> None:
    """The legacy RTP and Gaussian models agree at the two-point level."""

    lags = np.linspace(0.0, duration, 10_001)
    reference = ONE_OVER_F_REFERENCE_CORRELATION.covariance(lags)
    fitted = np.sum(
        np.asarray(weights)[:, np.newaxis] * np.exp(-np.asarray(rates)[:, np.newaxis] * lags),
        axis=0,
    )

    normalized_max_error = np.max(np.abs(fitted - reference)) / reference[0]
    rms_ratio = np.sqrt(fitted[0] / reference[0])
    assert normalized_max_error < 2.1e-4
    assert rms_ratio == pytest.approx(1.0, abs=1.1e-4)


def test_multisource_auto_sampling_uses_unfitted_gaussian_reference() -> None:
    _, problem, robustness, _ = build_multisource_problem(
        ExperimentTask.QFT,
        2,
        time_step=0.025,
        order=1,
        static_coupling_pairs=((1, 2),),
        include_frequency_drift=False,
        coupling_weight=1.0 / 3.0,
        frequency_drift_weight=0.0,
        control_drift_weight=1.0 / 3.0,
        colored_weight=1.0 / 3.0,
        control_drift_qubits=(1,),
        non_markovian_qubit=2,
    )
    assert robustness.colored[0].reference_correlation is not None

    automatic = generate_noise_realizations(
        problem,
        robustness,
        n_samples=3,
        rng=np.random.default_rng(17),
    )
    explicit = generate_noise_realizations(
        problem,
        robustness,
        n_samples=3,
        rng=np.random.default_rng(17),
        colored_noise_sampling="reference-gaussian",
    )
    rtp = generate_noise_realizations(
        problem,
        robustness,
        n_samples=3,
        rng=np.random.default_rng(17),
        colored_noise_sampling="rtp",
    )

    for auto_draw, explicit_draw in zip(automatic, explicit, strict=True):
        np.testing.assert_array_equal(
            auto_draw.colored_trajectories,
            explicit_draw.colored_trajectories,
        )
    assert not np.array_equal(
        automatic[0].colored_trajectories,
        rtp[0].colored_trajectories,
    )

    metadata = colored_noise_sampling_metadata(robustness, "auto")
    sources = metadata["sources"]
    assert isinstance(sources, list)
    source = sources[0]
    assert source["resolved_method"] == "reference-gaussian"
    assert source["strength"] == pytest.approx(robustness.colored[0].strength)
    assert source["optimization_process_variance"] == pytest.approx(
        sum(robustness.colored[0].correlation_weights)
    )
    assert source["reference_rms_hamiltonian_coefficient"] == pytest.approx(2.0 * np.pi)


def test_colored_noise_rms_strength_is_independent_of_correlation_shape() -> None:
    """The paper-level MHz setting maps to the independent epsilon scale."""

    _, _, robustness, _ = build_multisource_problem(
        ExperimentTask.QFT,
        2,
        time_step=0.025,
        order=1,
        static_coupling_pairs=((1, 2),),
        include_frequency_drift=False,
        coupling_weight=1.0 / 3.0,
        frequency_drift_weight=0.0,
        control_drift_weight=1.0 / 3.0,
        colored_weight=1.0 / 3.0,
        control_drift_qubits=(1,),
        non_markovian_qubit=2,
        colored_standard_deviation_mhz=2.5,
    )
    source = robustness.colored[0]
    assert source.reference_correlation is not None
    reference_variance = source.reference_correlation.covariance(np.asarray([0.0]))[0]

    assert source.strength * np.sqrt(reference_variance) == pytest.approx(2.0 * np.pi * 2.5)
