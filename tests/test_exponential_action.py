"""Validation of the independent block-sparse exponential-action backend."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from scipy.linalg import expm

from stgrape.augmented import BlockOperator
from stgrape.problem import ControlProblem, ControlTask, RobustnessModel, TaylorNoise
from stgrape.propagation.exponential_action import (
    BlockLinearOperator,
    ExponentialActionInterval,
    build_exponential_action_components,
    evaluate_exponential_action,
)
from stgrape.propagation.ode import (
    DOP853Options,
    OdePropagationOptions,
    OdeRepresentation,
    build_ode_components,
)
from stgrape.systems import build_spin_chain


def _operator() -> BlockOperator:
    return BlockOperator.from_entries(
        3,
        [
            (0, 0, np.array([[1.0, 0.2j], [-0.3j, -0.5]])),
            (1, 0, np.array([[0.1, 0.4], [-0.2j, 0.3]])),
            (1, 1, np.array([[0.2j, 0.1], [0.7, -0.4j]])),
            (2, 1, np.array([[0.5, -0.1j], [0.2, 0.6]])),
            (2, 2, np.array([[-0.8, 0.3j], [0.1, 0.9]])),
        ],
    )


@pytest.mark.parametrize("n_columns", [1, 4])
def test_block_linear_operator_matches_dense_for_any_column_count(n_columns: int) -> None:
    operator = _operator()
    linear = BlockLinearOperator(operator)
    values = np.random.default_rng(17).normal(size=(6, n_columns)) + 1j * np.random.default_rng(
        18
    ).normal(size=(6, n_columns))

    actual = np.asarray(
        linear.matvec(values[:, 0]) if n_columns == 1 else linear.matmat(values),
        dtype=np.complex128,
    )
    expected = operator.to_dense() @ (values[:, 0] if n_columns == 1 else values)
    adjoint_actual = np.asarray(
        linear.rmatvec(values[:, 0]) if n_columns == 1 else linear.rmatmat(values),
        dtype=np.complex128,
    )
    adjoint_expected = operator.to_dense().conj().T @ (values[:, 0] if n_columns == 1 else values)

    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(adjoint_actual, adjoint_expected, atol=2e-15, rtol=2e-15)
    assert operator.trace() == pytest.approx(np.trace(operator.to_dense()), abs=1e-15)


def test_block_linear_operator_adjoint_inner_product() -> None:
    linear = BlockLinearOperator(_operator())
    rng = np.random.default_rng(31)
    left = rng.normal(size=6) + 1j * rng.normal(size=6)
    right = rng.normal(size=6) + 1j * rng.normal(size=6)

    assert np.vdot(left, linear.matvec(right)) == pytest.approx(
        np.vdot(linear.rmatvec(left), right), abs=2e-14, rel=2e-14
    )


@pytest.mark.parametrize("n_columns", [1, 3])
def test_exponential_action_interval_matches_dense_exponential(n_columns: int) -> None:
    hamiltonian = _operator()
    duration = 0.037
    rng = np.random.default_rng(4)
    state = rng.normal(size=(3, 2, n_columns)) + 1j * rng.normal(size=(3, 2, n_columns))
    interval = ExponentialActionInterval.from_hamiltonian(hamiltonian)

    actual = interval.forward(state, duration)
    expected = expm(-1j * duration * hamiltonian.to_dense()) @ state.reshape(6, n_columns)

    np.testing.assert_allclose(actual.reshape(6, n_columns), expected, atol=2e-13, rtol=2e-13)


def test_exponential_action_forward_and_backward_are_adjoint() -> None:
    interval = ExponentialActionInterval.from_hamiltonian(_operator())
    rng = np.random.default_rng(7)
    state = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    costate = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))

    forward = interval.forward(state, 0.021)
    backward = interval.backward(costate, 0.021)

    assert np.vdot(costate, forward) == pytest.approx(
        np.vdot(backward, state), abs=3e-13, rel=3e-13
    )


def test_full_action_evaluation_matches_high_accuracy_block_sparse_dop853() -> None:
    chain = build_spin_chain(2, coupling="XX+YY")
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
            static_operators=(2.0 * np.pi * chain.z_operators[0],),
            order=2,
        )
    )
    controls = np.random.default_rng(22).uniform(-20.0, 20.0, size=(4, 4))
    action_components = build_exponential_action_components(problem, robustness)
    action_loss, action_gradient, _, action_states = evaluate_exponential_action(
        action_components,
        problem,
        controls,
        compute_gradient=True,
    )
    ode_components = build_ode_components(
        problem,
        robustness,
        OdePropagationOptions(
            DOP853Options(rtol=1e-12, atol=1e-14),
            OdeRepresentation.BLOCK_SPARSE,
        ),
    )
    ode_evaluation = ode_components[0].evaluate(
        problem,
        controls,
        fidelity_weight=1.0,
        compute_gradient=True,
    )

    assert action_loss == pytest.approx(ode_evaluation.objective.loss, abs=2e-12)
    np.testing.assert_allclose(action_states[0], ode_evaluation.final_state, atol=2e-12, rtol=2e-12)
    assert action_gradient is not None
    assert ode_evaluation.amplitude_gradient is not None
    np.testing.assert_allclose(
        action_gradient,
        ode_evaluation.amplitude_gradient,
        atol=2e-12,
        rtol=2e-10,
    )


def test_full_action_evaluation_supports_qft_gate_columns() -> None:
    chain = build_spin_chain(2, coupling="XX+YY")
    initial = qt.qeye(4)
    initial.dims = chain.drift_hamiltonian.dims
    target = qt.qft(4)
    target.dims = chain.drift_hamiltonian.dims
    problem = ControlProblem(
        chain.drift_hamiltonian,
        chain.control_hamiltonians,
        initial,
        target,
        duration=2e-5,
        time_step=1e-5,
        task=ControlTask.GATE_SYNTHESIS,
    )
    robustness = RobustnessModel(
        taylor=TaylorNoise(
            static_operators=(2.0 * np.pi * chain.z_operators[0],),
            order=1,
        )
    )
    controls = np.random.default_rng(23).uniform(-20.0, 20.0, size=(4, 2))
    action_components = build_exponential_action_components(problem, robustness)
    action_loss, action_gradient, _, action_states = evaluate_exponential_action(
        action_components,
        problem,
        controls,
        compute_gradient=True,
    )
    ode_component = build_ode_components(
        problem,
        robustness,
        OdePropagationOptions(
            DOP853Options(rtol=1e-12, atol=1e-14),
            OdeRepresentation.BLOCK_SPARSE,
        ),
    )[0]
    ode_evaluation = ode_component.evaluate(
        problem,
        controls,
        fidelity_weight=1.0,
        compute_gradient=True,
    )

    assert action_states[0].shape == (2, 4, 4)
    assert action_loss == pytest.approx(ode_evaluation.objective.loss, abs=2e-12)
    np.testing.assert_allclose(action_states[0], ode_evaluation.final_state, atol=2e-12, rtol=2e-12)
    assert action_gradient is not None
    assert ode_evaluation.amplitude_gradient is not None
    np.testing.assert_allclose(
        action_gradient,
        ode_evaluation.amplitude_gradient,
        atol=2e-12,
        rtol=2e-10,
    )
