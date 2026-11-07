import numpy as np
import pytest
import qutip as qt

from stgrape.optimizers.grape import GrapeSolver
from stgrape.optimizers.st_grape import STGrapeSolver
from stgrape.problem import (
    ColoredNoise,
    ControlDrift,
    ControlProblem,
    ControlTask,
    RobustnessModel,
    RobustPenaltyMode,
    TaylorNoise,
)
from stgrape.propagation.exact import build_dense_components
from stgrape.propagation.trotter import build_trotter_components
from stgrape.systems import SpinChain, build_spin_chain


def _problem() -> tuple[SpinChain, ControlProblem]:
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
    return chain, problem


@pytest.mark.parametrize("penalty_mode", tuple(RobustPenaltyMode))
def test_trotter_objective_and_gradient_for_all_noise_classes(
    penalty_mode: RobustPenaltyMode,
) -> None:
    chain, problem = _problem()
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            control_drifts=(ControlDrift(channel=1, scale=0.005),),
            order=2,
        ),
        colored=(
            ColoredNoise(
                operator=0.2 * chain.z_operators[1],
                correlation_weights=(0.2, 0.3),
                decay_rates=(1.0, 5.0),
            ),
            ColoredNoise(
                operator=0.15 * chain.z_operators[0],
                correlation_weights=(1.0,),
                decay_rates=(2.0,),
            ),
        ),
        penalty_mode=penalty_mode,
        p1_coefficient=1.0,
        p2_coefficient=1.25,
        pc_coefficient=0.45,
        rms_smoothing=1e-10,
    )
    controls = np.random.default_rng(4).uniform(-20, 20, size=(4, 4))
    trotter = build_trotter_components(problem, robustness, chain.control_groups)
    dense = build_dense_components(problem, robustness)
    assert len(trotter) == len(dense) == 2

    trotter_loss, analytic = STGrapeSolver._evaluate_trotter(
        trotter, problem, controls, compute_gradient=True
    )
    dense_loss, _ = GrapeSolver._evaluate(dense, problem, controls, compute_gradient=False)
    assert abs(trotter_loss - dense_loss) < 1e-5

    epsilon = 1e-5
    numeric = np.zeros_like(controls)
    for channel in range(problem.n_controls):
        for time_index in range(problem.n_time_steps):
            forward = controls.copy()
            backward = controls.copy()
            forward[channel, time_index] += epsilon
            backward[channel, time_index] -= epsilon
            forward_loss, _ = STGrapeSolver._evaluate_trotter(
                trotter, problem, forward, compute_gradient=False
            )
            backward_loss, _ = STGrapeSolver._evaluate_trotter(
                trotter, problem, backward, compute_gradient=False
            )
            numeric[channel, time_index] = (forward_loss - backward_loss) / (2 * epsilon)

    assert analytic is not None
    np.testing.assert_allclose(analytic, numeric, atol=1e-9, rtol=3e-5)


def test_trotter_objective_has_second_order_global_error() -> None:
    chain = build_spin_chain(2)
    initial = qt.basis(4, 0)
    initial.dims = [[2, 2], [1, 1]]
    target = qt.ghz_state(2)
    noise = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            order=1,
        )
    )
    differences = []
    for time_step in (0.001, 0.0005, 0.00025):
        problem = ControlProblem(
            chain.drift_hamiltonian,
            chain.control_hamiltonians,
            initial,
            target,
            duration=0.004,
            time_step=time_step,
            task=ControlTask.STATE_PREPARATION,
        )
        controls = np.tile(np.array([[8.0], [-3.0], [5.0], [2.0]]), (1, problem.n_time_steps))
        trotter = build_trotter_components(problem, noise, chain.control_groups)
        dense = build_dense_components(problem, noise)
        st_loss, _ = STGrapeSolver._evaluate_trotter(
            trotter, problem, controls, compute_gradient=False
        )
        exact_loss, _ = GrapeSolver._evaluate(dense, problem, controls, compute_gradient=False)
        differences.append(abs(st_loss - exact_loss))

    # Halving dt should reduce the global second-order error by approximately four.
    assert differences[0] / differences[1] > 3.0
    assert differences[1] / differences[2] > 3.0


def test_internal_splitting_halves_the_step_without_changing_the_pwc_grid() -> None:
    """Two internal steps improve accuracy and retain one gradient per PWC bin."""

    chain, problem = _problem()
    noise = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            order=1,
        )
    )
    controls = np.random.default_rng(21).uniform(
        -20,
        20,
        size=(problem.n_controls, problem.n_time_steps),
    )
    exact = build_dense_components(problem, noise)
    exact_loss, _ = GrapeSolver._evaluate(
        exact,
        problem,
        controls,
        compute_gradient=False,
    )
    errors: list[float] = []
    for splitting_steps in (1, 2):
        trotter = build_trotter_components(
            problem,
            noise,
            chain.control_groups,
            splitting_steps=splitting_steps,
        )
        loss, gradient = STGrapeSolver._evaluate_trotter(
            trotter,
            problem,
            controls,
            compute_gradient=True,
        )
        errors.append(abs(loss - exact_loss))
        assert gradient is not None
        assert gradient.shape == controls.shape
        assert all(component.splitting_steps == splitting_steps for component in trotter)

    assert errors[1] < errors[0]


def test_gate_synthesis_gradient() -> None:
    chain = build_spin_chain(2)
    initial = qt.qeye(4)
    initial.dims = chain.drift_hamiltonian.dims
    target = qt.qft(4)
    target.dims = chain.drift_hamiltonian.dims
    problem = ControlProblem(
        chain.drift_hamiltonian,
        chain.control_hamiltonians,
        initial,
        target,
        duration=0.0005,
        time_step=0.00025,
        task=ControlTask.GATE_SYNTHESIS,
    )
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2 * np.pi * (chain.z_operators[0] @ chain.z_operators[1]),),
            order=2,
        )
    )
    controls = np.random.default_rng(9).uniform(-10, 10, size=(4, 2))
    trotter = build_trotter_components(problem, robustness, chain.control_groups)
    _, analytic = STGrapeSolver._evaluate_trotter(trotter, problem, controls, compute_gradient=True)
    epsilon = 1e-5
    numeric = np.zeros_like(controls)
    for channel in range(problem.n_controls):
        for time_index in range(problem.n_time_steps):
            forward = controls.copy()
            backward = controls.copy()
            forward[channel, time_index] += epsilon
            backward[channel, time_index] -= epsilon
            forward_loss, _ = STGrapeSolver._evaluate_trotter(
                trotter, problem, forward, compute_gradient=False
            )
            backward_loss, _ = STGrapeSolver._evaluate_trotter(
                trotter, problem, backward, compute_gradient=False
            )
            numeric[channel, time_index] = (forward_loss - backward_loss) / (2 * epsilon)
    assert analytic is not None
    np.testing.assert_allclose(analytic, numeric, atol=1e-9, rtol=3e-5)
