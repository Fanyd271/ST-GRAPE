import numpy as np
import pytest
import qutip as qt

from stgrape.optimizers import GrapeSolver
from stgrape.problem import (
    ColoredNoise,
    ControlDrift,
    ControlProblem,
    ControlTask,
    RobustnessModel,
    TaylorNoise,
)
from stgrape.propagation.exact import build_dense_components
from stgrape.propagation.ode import (
    DOP853Options,
    OdePropagationOptions,
    OdeRepresentation,
    build_ode_components,
)
from stgrape.systems import build_spin_chain


def _state_problem() -> tuple[ControlProblem, RobustnessModel, np.ndarray]:
    chain = build_spin_chain(2)
    initial = qt.basis(4, 0)
    initial.dims = [[2, 2], [1, 1]]
    problem = ControlProblem(
        drift_hamiltonian=chain.drift_hamiltonian,
        control_hamiltonians=chain.control_hamiltonians,
        initial=initial,
        target=qt.ghz_state(2),
        duration=0.001,
        time_step=0.00025,
        task=ControlTask.STATE_PREPARATION,
    )
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2.0 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            order=2,
        )
    )
    controls = np.random.default_rng(12).uniform(-20.0, 20.0, size=(4, 4))
    return problem, robustness, controls


def test_ode_components_support_static_control_drift_and_colored_noise() -> None:
    problem, _, controls = _state_problem()
    chain = build_spin_chain(2)
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2.0 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            control_drifts=(ControlDrift(channel=1, scale=0.005),),
            order=2,
        ),
        colored=(
            ColoredNoise(
                operator=0.2 * chain.z_operators[1],
                correlation_weights=(0.2, 0.3),
                decay_rates=(1.0, 5.0),
            ),
        ),
    )
    exact = build_dense_components(problem, robustness)
    exact_loss, _ = GrapeSolver._evaluate(exact, problem, controls, compute_gradient=False)
    losses = []
    for representation in OdeRepresentation:
        components = build_ode_components(
            problem,
            robustness,
            OdePropagationOptions(
                DOP853Options(rtol=1e-10, atol=1e-12),
                representation,
            ),
        )
        loss, gradient = GrapeSolver._evaluate(
            components,
            problem,
            controls,
            compute_gradient=True,
        )
        assert gradient is not None
        assert np.all(np.isfinite(gradient))
        assert loss == pytest.approx(exact_loss, abs=1e-11)
        losses.append(loss)
    assert losses[0] == pytest.approx(losses[1], abs=1e-13)


def test_block_sparse_ode_gradient_matches_finite_difference_at_small_step() -> None:
    chain = build_spin_chain(2)
    initial = qt.basis(4, 0)
    initial.dims = [[2, 2], [1, 1]]
    problem = ControlProblem(
        chain.drift_hamiltonian,
        chain.control_hamiltonians,
        initial,
        qt.ghz_state(2),
        duration=4e-5,
        time_step=1e-5,
        task=ControlTask.STATE_PREPARATION,
    )
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2.0 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            order=1,
        )
    )
    controls = np.random.default_rng(2).uniform(-20.0, 20.0, size=(4, 4))
    components = build_ode_components(
        problem,
        robustness,
        OdePropagationOptions(
            DOP853Options(rtol=1e-11, atol=1e-13),
            OdeRepresentation.BLOCK_SPARSE,
        ),
    )
    _, analytic = GrapeSolver._evaluate(components, problem, controls, compute_gradient=True)

    epsilon = 1e-5
    numeric = np.zeros_like(controls)
    for channel in range(problem.n_controls):
        for time_index in range(problem.n_time_steps):
            forward = controls.copy()
            backward = controls.copy()
            forward[channel, time_index] += epsilon
            backward[channel, time_index] -= epsilon
            forward_loss, _ = GrapeSolver._evaluate(
                components, problem, forward, compute_gradient=False
            )
            backward_loss, _ = GrapeSolver._evaluate(
                components, problem, backward, compute_gradient=False
            )
            numeric[channel, time_index] = (forward_loss - backward_loss) / (2.0 * epsilon)

    assert analytic is not None
    np.testing.assert_allclose(analytic, numeric, atol=2e-11, rtol=1e-2)
