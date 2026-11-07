import numpy as np

from stgrape.pulse import ChebyshevPulse, FilteredPulse, build_chebyshev_basis


def test_pulse_pullback_matches_finite_difference() -> None:
    rng = np.random.default_rng(5)
    pulse = FilteredPulse.random(
        n_controls=2,
        n_time_steps=40,
        time_step=0.00025,
        rng=rng,
        rise_time=0.002,
        cutoff_frequency=60.0,
        filter_taps=10,
    )
    amplitude_gradient = rng.normal(size=(2, 40))
    analytic = pulse.pullback(amplitude_gradient)

    direction = rng.normal(size=pulse.coefficients.shape)
    epsilon = 1e-5
    forward = np.vdot(
        amplitude_gradient,
        pulse.amplitudes(pulse.coefficients + epsilon * direction),
    ).real
    backward = np.vdot(
        amplitude_gradient,
        pulse.amplitudes(pulse.coefficients - epsilon * direction),
    ).real
    numeric = (forward - backward) / (2 * epsilon)
    np.testing.assert_allclose(np.vdot(analytic, direction).real, numeric, rtol=1e-8)


def test_chebyshev_chain_pullback_matches_finite_difference() -> None:
    rng = np.random.default_rng(12)
    pulse = ChebyshevPulse.random(
        n_controls=2,
        n_time_steps=40,
        time_step=0.00025,
        n_basis=8,
        rng=rng,
        amplitude_limit=20.0,
        rise_time=0.002,
        cutoff_frequency=60.0,
        filter_taps=10,
    )
    trial = pulse.coefficients + rng.normal(scale=0.2, size=pulse.coefficients.shape)
    amplitude_gradient = rng.normal(size=(2, 40))
    analytic = pulse.pullback(amplitude_gradient, trial)

    direction = rng.normal(size=trial.shape)
    epsilon = 1e-6
    forward = np.vdot(
        amplitude_gradient,
        pulse.amplitudes(trial + epsilon * direction),
    ).real
    backward = np.vdot(
        amplitude_gradient,
        pulse.amplitudes(trial - epsilon * direction),
    ).real
    numeric = (forward - backward) / (2 * epsilon)
    np.testing.assert_allclose(
        np.vdot(analytic, direction).real,
        numeric,
        rtol=2e-7,
        atol=1e-7,
    )


def test_chebyshev_basis_size_and_unbounded_coefficients() -> None:
    basis = build_chebyshev_basis(32, 8, normalization="rms")
    assert basis.shape == (32, 8)
    np.testing.assert_allclose(np.sqrt(np.mean(basis**2, axis=0)), 1.0)

    pulse = ChebyshevPulse(
        n_controls=1,
        n_time_steps=32,
        time_step=0.0005,
        coefficients=np.zeros((1, 8)),
        n_basis=8,
        amplitude_limit=10.0,
        rise_time=0.002,
        cutoff_frequency=60.0,
        filter_taps=4,
    )
    assert pulse.n_parameters_per_control == 8
    assert pulse.parameter_bounds(3.0) is None
    assert pulse.metadata()["control_chain"] == [
        "Chebyshev",
        "tanh",
        "LTI",
        "envelope",
    ]
