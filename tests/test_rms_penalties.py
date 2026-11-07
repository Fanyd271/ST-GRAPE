"""Physical RMS terminal-penalty values and gradients."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from stgrape.augmented.builders import (
    build_augmented_hamiltonians,
    taylor_penalty_terms,
    taylor_rms_penalties,
)
from stgrape.augmented.indices import generate_multi_indices
from stgrape.diagnostics import taylor_sensitivity_diagnostics_from_state
from stgrape.objectives import terminal_objective
from stgrape.problem import (
    ColoredNoise,
    ControlProblem,
    ControlTask,
    RobustnessModel,
    RobustPenaltyMode,
    TaylorNoise,
)


def _one_qubit_state_problem() -> ControlProblem:
    """Return the smallest problem needed to define diagnostic normalization."""

    return ControlProblem(
        drift_hamiltonian=0.2 * qt.sigmaz(),
        control_hamiltonians=(qt.sigmax(),),
        initial=qt.basis(2, 0),
        target=qt.basis(2, 1),
        duration=0.1,
        time_step=0.1,
        task=ControlTask.STATE_PREPARATION,
    )


def test_gaussian_rms_penalty_matches_first_and_second_order_moments() -> None:
    """Independent-Gaussian diagonal and mixed blocks use exact fourth moments."""

    noise = TaylorNoise(
        static_operators=(qt.sigmax(), qt.sigmaz()),
        order=2,
    )
    robustness = RobustnessModel(
        taylor=noise,
        penalty_mode=RobustPenaltyMode.RMS,
        p1_coefficient=0.98,
        p2_coefficient=1.372,
        rms_smoothing=1e-3,
        p1_smoothing=2e-3,
        p2_smoothing=5e-3,
    )
    indices = generate_multi_indices(noise.n_sources, noise.order)
    positions = {index: position for position, index in enumerate(indices)}
    state = np.zeros((len(indices), 2, 1), dtype=np.complex128)
    state[positions[(0, 0)]] = np.asarray([[1.0], [0.0]])
    blocks = {
        (1, 0): np.asarray([[0.2 + 0.1j], [0.3 - 0.2j]]),
        (0, 1): np.asarray([[-0.1j], [0.4 + 0.2j]]),
        (2, 0): np.asarray([[0.05], [0.07j]]),
        (0, 2): np.asarray([[-0.02j], [0.03]]),
        (1, 1): np.asarray([[0.04 + 0.01j], [-0.02]]),
    }
    for index, value in blocks.items():
        state[positions[index]] = value

    penalties = taylor_rms_penalties(noise, indices, robustness)
    objective, costate = terminal_objective(
        state,
        qt.basis(2, 0),
        ControlTask.STATE_PREPARATION,
        nominal_block=positions[(0, 0)],
        penalty_blocks=(),
        fidelity_weight=0.0,
        penalty_weights=(),
        rms_penalties=penalties,
    )

    first_squared = sum(
        float(np.vdot(blocks[index], blocks[index]).real) for index in ((1, 0), (0, 1))
    )
    diagonal_sum = blocks[(2, 0)] + blocks[(0, 2)]
    second_squared = (
        float(np.vdot(diagonal_sum, diagonal_sum).real)
        + 2.0 * float(np.vdot(blocks[(2, 0)], blocks[(2, 0)]).real)
        + 2.0 * float(np.vdot(blocks[(0, 2)], blocks[(0, 2)]).real)
        + float(np.vdot(blocks[(1, 1)], blocks[(1, 1)]).real)
    )
    expected = robustness.resolved_p1_coefficient * (
        np.sqrt(first_squared + robustness.resolved_p1_smoothing**2)
        - robustness.resolved_p1_smoothing
    ) + robustness.resolved_p2_coefficient * (
        np.sqrt(second_squared + robustness.resolved_p2_smoothing**2)
        - robustness.resolved_p2_smoothing
    )
    assert objective.penalty == pytest.approx(expected)

    direction = np.random.default_rng(8).normal(size=state.shape) + 1j * np.random.default_rng(
        9
    ).normal(size=state.shape)
    direction[positions[(0, 0)]] = 0.0
    epsilon = 1e-7

    def penalty_at(trial: np.ndarray) -> float:
        value, _ = terminal_objective(
            trial,
            qt.basis(2, 0),
            ControlTask.STATE_PREPARATION,
            nominal_block=positions[(0, 0)],
            penalty_blocks=(),
            fidelity_weight=0.0,
            penalty_weights=(),
            rms_penalties=penalties,
        )
        return value.penalty

    numeric = (
        penalty_at(state + epsilon * direction) - penalty_at(state - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = float(np.vdot(costate, direction).real)
    assert analytic == pytest.approx(numeric, rel=2e-8, abs=2e-9)


def test_rms_penalty_is_smooth_and_zero_at_zero_sensitivity() -> None:
    """The smoothing convention removes the norm cusp without adding an offset."""

    noise = TaylorNoise(static_operators=(qt.sigmaz(),), order=1)
    robustness = RobustnessModel(
        taylor=noise,
        penalty_mode=RobustPenaltyMode.RMS,
        rms_smoothing=1e-4,
    )
    indices = generate_multi_indices(1, 1)
    state = np.zeros((2, 2, 1), dtype=np.complex128)
    state[-1] = np.asarray([[1.0], [0.0]])
    objective, costate = terminal_objective(
        state,
        qt.basis(2, 0),
        ControlTask.STATE_PREPARATION,
        nominal_block=1,
        penalty_blocks=(),
        fidelity_weight=0.0,
        penalty_weights=(),
        rms_penalties=taylor_rms_penalties(noise, indices, robustness),
    )

    assert objective.penalty == pytest.approx(0.0)
    np.testing.assert_array_equal(costate, np.zeros_like(costate))


def test_rms_p1_and_p2_use_independent_smoothing_scales() -> None:
    """Dedicated P1/P2 values override the shared legacy fallback separately."""

    noise = TaylorNoise(
        static_operators=(qt.sigmax(), qt.sigmaz()),
        order=2,
    )
    robustness = RobustnessModel(
        taylor=noise,
        penalty_mode=RobustPenaltyMode.RMS,
        rms_smoothing=1e-6,
        p1_smoothing=2e-4,
        p2_smoothing=3e-3,
    )
    penalties = taylor_rms_penalties(
        noise,
        generate_multi_indices(noise.n_sources, noise.order),
        robustness,
    )

    assert robustness.resolved_p1_smoothing == pytest.approx(2e-4)
    assert robustness.resolved_p2_smoothing == pytest.approx(3e-3)
    assert tuple(penalty.smoothing for penalty in penalties) == pytest.approx((2e-4, 3e-3))


def test_direct_rms_coefficients_are_resolved_independently() -> None:
    """Explicit P1/P2/Pc coefficients are the effective objective weights."""

    robustness = RobustnessModel(
        penalty_mode=RobustPenaltyMode.RMS,
        p1_coefficient=0.25,
        p2_coefficient=1.5,
        pc_coefficient=2.75,
    )

    assert robustness.resolved_p1_coefficient == pytest.approx(0.25)
    assert robustness.resolved_p2_coefficient == pytest.approx(1.5)
    assert robustness.resolved_pc_coefficient == pytest.approx(2.75)


def test_hybrid_penalty_squares_p1_and_uses_smooth_rms_p2() -> None:
    """Hybrid mode keeps both Taylor contributions at physical order sigma squared."""

    noise = TaylorNoise(
        static_operators=(qt.sigmax(), qt.sigmaz()),
        order=2,
    )
    robustness = RobustnessModel(
        taylor=noise,
        penalty_mode=RobustPenaltyMode.HYBRID,
        p1_coefficient=0.4,
        p2_coefficient=0.7,
        rms_smoothing=1e-3,
        p2_smoothing=2e-2,
    )
    indices = generate_multi_indices(noise.n_sources, noise.order)
    positions = {index: position for position, index in enumerate(indices)}
    state = np.zeros((len(indices), 2, 1), dtype=np.complex128)
    state[positions[(0, 0)]] = np.asarray([[1.0], [0.0]])
    blocks = {
        (1, 0): np.asarray([[0.2], [0.1j]]),
        (0, 1): np.asarray([[0.3j], [-0.1]]),
        (2, 0): np.asarray([[0.05], [0.02j]]),
        (0, 2): np.asarray([[-0.01j], [0.04]]),
        (1, 1): np.asarray([[0.03], [-0.02j]]),
    }
    for index, value in blocks.items():
        state[positions[index]] = value

    penalty_blocks, penalty_weights, rms_penalties = taylor_penalty_terms(
        noise,
        indices,
        robustness,
    )
    objective, _ = terminal_objective(
        state,
        qt.basis(2, 0),
        ControlTask.STATE_PREPARATION,
        nominal_block=positions[(0, 0)],
        penalty_blocks=penalty_blocks,
        fidelity_weight=0.0,
        penalty_weights=penalty_weights,
        rms_penalties=rms_penalties,
    )

    first_squared = sum(
        float(np.vdot(blocks[index], blocks[index]).real) for index in ((1, 0), (0, 1))
    )
    diagonal_sum = blocks[(2, 0)] + blocks[(0, 2)]
    second_squared = (
        float(np.vdot(diagonal_sum, diagonal_sum).real)
        + 2.0 * float(np.vdot(blocks[(2, 0)], blocks[(2, 0)]).real)
        + 2.0 * float(np.vdot(blocks[(0, 2)], blocks[(0, 2)]).real)
        + float(np.vdot(blocks[(1, 1)], blocks[(1, 1)]).real)
    )
    expected = 0.4 * first_squared + 0.7 * (np.sqrt(second_squared + 0.02**2) - 0.02)
    assert objective.penalty == pytest.approx(expected)
    assert penalty_blocks == (positions[(1, 0)], positions[(0, 1)])
    assert penalty_weights == pytest.approx((0.8, 0.8))
    assert tuple(penalty.smoothing for penalty in rms_penalties) == (0.02,)


def test_hybrid_direct_coefficients_and_independent_smoothing_fallbacks() -> None:
    """Hybrid uses independent direct coefficients and smoothing scales."""

    robustness = RobustnessModel(
        penalty_mode=RobustPenaltyMode.HYBRID,
        p1_coefficient=1.8,
        p2_coefficient=1.8,
        pc_coefficient=0.2,
        rms_smoothing=1e-6,
        p2_smoothing=2e-5,
        pc_smoothing=3e-5,
    )

    assert robustness.resolved_p1_coefficient == pytest.approx(1.8)
    assert robustness.resolved_p2_coefficient == pytest.approx(1.8)
    assert robustness.resolved_pc_coefficient == pytest.approx(0.2)
    assert robustness.resolved_p2_smoothing == pytest.approx(2e-5)
    assert robustness.resolved_pc_smoothing == pytest.approx(3e-5)


def test_multiple_colored_sources_use_one_joint_rms_penalty() -> None:
    """Independent colored sources combine in quadrature before smoothing."""

    problem = _one_qubit_state_problem()
    colored = (
        ColoredNoise(
            operator=qt.sigmax(),
            correlation_weights=(0.7, 0.3),
            decay_rates=(1.0, 3.0),
            strength=0.2,
        ),
        ColoredNoise(
            operator=qt.sigmaz(),
            correlation_weights=(1.0,),
            decay_rates=(2.0,),
            strength=0.4,
        ),
    )
    robustness = RobustnessModel(
        colored=colored,
        penalty_mode=RobustPenaltyMode.HYBRID,
        pc_coefficient=1.7,
        pc_smoothing=0.05,
    )
    components = build_augmented_hamiltonians(problem, robustness)

    assert len(components) == 1
    component = components[0]
    assert component.n_blocks == 6
    assert component.nominal_block == 5
    assert component.penalty_blocks == ()
    assert len(component.rms_penalties) == 1
    assert component.rms_penalties[0].features == (((0, 1.0),), ((3, 1.0),))

    state = np.zeros((component.n_blocks, 2, 1), dtype=np.complex128)
    state[component.nominal_block] = problem.initial.full()
    state[0] = np.asarray([[0.2 + 0.1j], [-0.3j]])
    state[3] = np.asarray([[-0.1j], [0.4 + 0.2j]])
    objective, costate = terminal_objective(
        state,
        problem.target,
        problem.task,
        nominal_block=component.nominal_block,
        penalty_blocks=component.penalty_blocks,
        fidelity_weight=0.0,
        penalty_weights=component.penalty_weights,
        rms_penalties=component.rms_penalties,
    )
    joint_squared = float(np.vdot(state[0], state[0]).real) + float(
        np.vdot(state[3], state[3]).real
    )
    expected = 1.7 * (np.sqrt(joint_squared + 0.05**2) - 0.05)
    assert objective.penalty == pytest.approx(expected)

    direction = np.random.default_rng(21).normal(size=state.shape) + 1j * np.random.default_rng(
        22
    ).normal(size=state.shape)
    direction[component.nominal_block] = 0.0
    epsilon = 1e-7

    def penalty_at(trial: np.ndarray) -> float:
        value, _ = terminal_objective(
            trial,
            problem.target,
            problem.task,
            nominal_block=component.nominal_block,
            penalty_blocks=component.penalty_blocks,
            fidelity_weight=0.0,
            penalty_weights=component.penalty_weights,
            rms_penalties=component.rms_penalties,
        )
        return value.penalty

    numeric = (
        penalty_at(state + epsilon * direction) - penalty_at(state - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = float(np.vdot(costate, direction).real)
    assert analytic == pytest.approx(numeric, rel=2e-8, abs=2e-9)


def test_saved_sensitivity_diagnostics_match_physical_p1_and_p2() -> None:
    """P1/P2 diagnostics use the same Gaussian moments as the RMS objective."""

    indices = generate_multi_indices(n_sources=2, order=2)
    positions = {index: position for position, index in enumerate(indices)}
    state = np.zeros((len(indices), 2, 1), dtype=np.complex128)
    blocks = {
        (1, 0): np.asarray([[0.2], [0.1j]]),
        (0, 1): np.asarray([[0.3j], [-0.1]]),
        (2, 0): np.asarray([[0.05], [0.02j]]),
        (0, 2): np.asarray([[-0.01j], [0.04]]),
        (1, 1): np.asarray([[0.03], [-0.02j]]),
    }
    for index, value in blocks.items():
        state[positions[index]] = value

    diagnostics = taylor_sensitivity_diagnostics_from_state(
        state,
        _one_qubit_state_problem(),
        indices,
        rhs_evaluations=17,
    )
    first_squared = sum(
        float(np.vdot(blocks[index], blocks[index]).real) for index in ((1, 0), (0, 1))
    )
    diagonal_sum = blocks[(2, 0)] + blocks[(0, 2)]
    second_squared = (
        float(np.vdot(diagonal_sum, diagonal_sum).real)
        + 2.0 * float(np.vdot(blocks[(2, 0)], blocks[(2, 0)]).real)
        + 2.0 * float(np.vdot(blocks[(0, 2)], blocks[(0, 2)]).real)
        + float(np.vdot(blocks[(1, 1)], blocks[(1, 1)]).real)
    )

    assert diagnostics.p1 == pytest.approx(np.sqrt(first_squared))
    assert diagnostics.p1_squared == pytest.approx(first_squared)
    assert diagnostics.hybrid_p1_base == pytest.approx(first_squared)
    assert diagnostics.p2 == pytest.approx(np.sqrt(second_squared))
    assert diagnostics.evaluated_order == 2
    assert diagnostics.rhs_evaluations == 17
