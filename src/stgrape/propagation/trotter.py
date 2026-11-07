"""Structure-aware second-order Suzuki--Trotter propagation."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj
from scipy.linalg import expm

from stgrape.augmented.block_sparse import BlockOperator, nilpotent_exponential
from stgrape.augmented.builders import (
    build_joint_colored_generators,
    build_taylor_generators,
    joint_colored_penalty_terms,
    taylor_penalty_terms,
)
from stgrape.objectives import ObjectiveValue, SmoothRmsPenalty, terminal_objective
from stgrape.problem import ControlProblem, RobustnessModel
from stgrape.systems.spin_chain import CommutingControlGroup

ComplexArray = NDArray[np.complex128]


class _Factor(Protocol):
    def apply(self, state: ComplexArray) -> ComplexArray: ...

    def adjoint_apply(self, costate: ComplexArray) -> ComplexArray: ...

    def add_gradient(self, costate: ComplexArray, gradient: NDArray[np.float64]) -> None: ...


@dataclass(frozen=True)
class _FixedFactor:
    operator: BlockOperator
    adjoint_operator: BlockOperator

    @classmethod
    def build(cls, operator: BlockOperator) -> _FixedFactor:
        return cls(operator, operator.adjoint())

    def apply(self, state: ComplexArray) -> ComplexArray:
        return self.operator.apply(state)

    def adjoint_apply(self, costate: ComplexArray) -> ComplexArray:
        return self.adjoint_operator.apply(costate)

    def add_gradient(self, costate: ComplexArray, gradient: NDArray[np.float64]) -> None:
        return None


@dataclass(frozen=True)
class _NilpotentAction:
    """Cached powers used to apply ``exp(coefficient * generator)``."""

    powers: tuple[BlockOperator, ...]
    adjoint_powers: tuple[BlockOperator, ...]

    @classmethod
    def build(cls, generator: BlockOperator, order: int) -> _NilpotentAction:
        powers: list[BlockOperator] = []
        current = BlockOperator.identity(generator.n_blocks, generator.block_dimension)
        for _ in range(order):
            current = generator.compose(current)
            powers.append(current)
        return cls(tuple(powers), tuple(power.adjoint() for power in powers))

    def apply(self, state: ComplexArray, coefficient: complex) -> ComplexArray:
        result = state.copy()
        for exponent, power in enumerate(self.powers, start=1):
            result += coefficient**exponent / factorial(exponent) * power.apply(state)
        return result

    def adjoint_apply(self, costate: ComplexArray, coefficient: complex) -> ComplexArray:
        result = costate.copy()
        conjugate = coefficient.conjugate()
        for exponent, power in enumerate(self.adjoint_powers, start=1):
            result += conjugate**exponent / factorial(exponent) * power.apply(costate)
        return result


@dataclass(frozen=True)
class _ControlGroupRuntime:
    channels: tuple[int, ...]
    diagonal_values: dict[int, NDArray[np.complex128]]
    drift_generators: dict[int, BlockOperator]
    drift_actions: dict[int, _NilpotentAction]
    half_time_step: float


@dataclass
class _ControlFactor:
    runtime: _ControlGroupRuntime
    amplitudes: NDArray[np.float64]
    time_index: int
    output_state: ComplexArray | None = None
    phase: ComplexArray | None = None

    def _phase(self) -> ComplexArray:
        diagonal = np.zeros_like(next(iter(self.runtime.diagonal_values.values())))
        for channel in self.runtime.channels:
            diagonal += self.amplitudes[channel] * self.runtime.diagonal_values[channel]
        return np.exp(-1j * self.runtime.half_time_step * diagonal)

    def apply(self, state: ComplexArray) -> ComplexArray:
        self.phase = self._phase()
        result = self.phase[None, :, None] * state
        for channel, action in self.runtime.drift_actions.items():
            coefficient = -1j * self.runtime.half_time_step * self.amplitudes[channel]
            result = action.apply(result, coefficient)
        self.output_state = result
        return result

    def adjoint_apply(self, costate: ComplexArray) -> ComplexArray:
        if self.phase is None:
            raise RuntimeError("Control adjoint requested before forward propagation.")
        result = costate
        for channel, action in reversed(tuple(self.runtime.drift_actions.items())):
            coefficient = -1j * self.runtime.half_time_step * self.amplitudes[channel]
            result = action.adjoint_apply(result, coefficient)
        return self.phase.conj()[None, :, None] * result

    def add_gradient(self, costate: ComplexArray, gradient: NDArray[np.float64]) -> None:
        if self.output_state is None:
            raise RuntimeError("Control gradient requested before forward propagation.")
        for channel in self.runtime.channels:
            generator_output = (
                self.runtime.diagonal_values[channel][None, :, None] * self.output_state
            )
            drift_generator = self.runtime.drift_generators.get(channel)
            if drift_generator is not None:
                generator_output = generator_output + drift_generator.apply(self.output_state)
            derivative = -1j * self.runtime.half_time_step * generator_output
            gradient[channel, self.time_index] += float(np.vdot(costate, derivative).real)


@dataclass(frozen=True)
class TrotterEvaluation:
    """Objective and amplitude-space gradient of one component."""

    objective: ObjectiveValue
    amplitude_gradient: NDArray[np.float64] | None


@dataclass(frozen=True)
class TrotterComponent:
    """One independently propagated augmented robustness component."""

    n_blocks: int
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...]
    splitting_steps: int
    initial_transform: BlockOperator
    final_transform: BlockOperator
    static_core: BlockOperator
    group_runtimes: tuple[_ControlGroupRuntime, ...]
    forward_transitions: tuple[BlockOperator, ...]
    backward_transitions: tuple[BlockOperator, ...]

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: NDArray[np.float64],
        *,
        fidelity_weight: float,
        compute_gradient: bool = True,
    ) -> TrotterEvaluation:
        """Propagate, evaluate the loss, and optionally reverse-propagate."""

        controls = np.asarray(control_amplitudes, dtype=float)
        expected = (problem.n_controls, problem.n_time_steps)
        if controls.shape != expected:
            raise ValueError(f"control_amplitudes has shape {controls.shape}; expected {expected}.")

        columns = 1 if problem.task.value == "state" else problem.hilbert_dimension
        state = np.zeros((self.n_blocks, problem.hilbert_dimension, columns), dtype=np.complex128)
        state[self.nominal_block] = problem.initial.full()

        initial_factor = _FixedFactor.build(self.initial_transform)
        final_factor = _FixedFactor.build(self.final_transform)
        static_factor = _FixedFactor.build(self.static_core)
        forward_factors = tuple(
            _FixedFactor.build(operator) for operator in self.forward_transitions
        )
        backward_factors = tuple(
            _FixedFactor.build(operator) for operator in self.backward_transitions
        )
        factors: list[_Factor] = [initial_factor]
        for time_index in range(problem.n_time_steps):
            amplitudes = controls[:, time_index]
            # Internal splitting refines propagation without changing the PWC
            # parameterization.  Every substep shares the same time-bin
            # amplitude, so reverse-mode contributions accumulate into the
            # original ``gradient[:, time_index]`` entry.
            for _ in range(self.splitting_steps):
                for group_index, runtime in enumerate(self.group_runtimes):
                    factors.append(_ControlFactor(runtime, amplitudes, time_index))
                    if group_index < len(self.forward_transitions):
                        factors.append(forward_factors[group_index])
                factors.append(static_factor)
                for reverse_index in range(len(self.group_runtimes) - 1, -1, -1):
                    runtime = self.group_runtimes[reverse_index]
                    factors.append(_ControlFactor(runtime, amplitudes, time_index))
                    if reverse_index > 0:
                        factors.append(backward_factors[reverse_index - 1])
        factors.append(final_factor)

        for factor in factors:
            state = factor.apply(state)

        objective, costate = terminal_objective(
            state,
            problem.target,
            problem.task,
            nominal_block=self.nominal_block,
            penalty_blocks=self.penalty_blocks,
            fidelity_weight=fidelity_weight,
            penalty_weights=self.penalty_weights,
            rms_penalties=self.rms_penalties,
        )
        if not compute_gradient:
            return TrotterEvaluation(objective, None)

        gradient = np.zeros(expected, dtype=float)
        for factor in reversed(factors):
            factor.add_gradient(costate, gradient)
            costate = factor.adjoint_apply(costate)
        return TrotterEvaluation(objective, gradient)


def _validate_control_groups(
    problem: ControlProblem,
    groups: tuple[CommutingControlGroup, ...],
) -> dict[int, tuple[int, Qobj]]:
    if not groups:
        raise ValueError("ST-GRAPE requires at least one commuting control group.")
    channel_map: dict[int, tuple[int, Qobj]] = {}
    for group_index, group in enumerate(groups):
        for channel, diagonal in zip(group.channels, group.diagonal_hamiltonians, strict=True):
            if channel in channel_map:
                raise ValueError(f"Control channel {channel} occurs in multiple groups.")
            if channel >= problem.n_controls:
                raise ValueError(f"Control channel {channel} lies outside the problem.")
            transformed = (
                group.basis_transform.dag()
                @ problem.control_hamiltonians[channel]
                @ group.basis_transform
            )
            if not np.allclose(transformed.full(), diagonal.full(), atol=1e-10):
                raise ValueError(
                    f"The diagonal Hamiltonian for channel {channel} does not match its basis transform."
                )
            channel_map[channel] = (group_index, diagonal)
    if set(channel_map) != set(range(problem.n_controls)):
        raise ValueError("Commuting control groups must cover every control channel exactly once.")
    return channel_map


def _build_component(
    problem: ControlProblem,
    groups: tuple[CommutingControlGroup, ...],
    *,
    n_blocks: int,
    drift_propagator: BlockOperator,
    error_generator: BlockOperator,
    nilpotent_order: int,
    drift_generators: dict[int, BlockOperator],
    nominal_block: int,
    penalty_blocks: tuple[int, ...],
    penalty_weights: tuple[float, ...],
    rms_penalties: tuple[SmoothRmsPenalty, ...],
    splitting_steps: int,
) -> TrotterComponent:
    half_dt = problem.time_step / (2.0 * splitting_steps)
    error_half_step = nilpotent_exponential(error_generator, -1j * half_dt, nilpotent_order)

    first_basis = groups[0].basis_transform.full()
    last_basis = groups[-1].basis_transform.full()
    initial_transform = BlockOperator.physical_diagonal(n_blocks, first_basis.conj().T)
    final_transform = BlockOperator.physical_diagonal(n_blocks, first_basis)
    last_basis_operator = BlockOperator.physical_diagonal(n_blocks, last_basis)
    last_basis_adjoint = BlockOperator.physical_diagonal(n_blocks, last_basis.conj().T)
    static_core = (
        last_basis_adjoint.compose(error_half_step)
        .compose(drift_propagator)
        .compose(error_half_step)
        .compose(last_basis_operator)
    )

    runtimes: list[_ControlGroupRuntime] = []
    for group in groups:
        diagonal_values = {
            channel: np.asarray(diagonal.diag(), dtype=np.complex128)
            for channel, diagonal in zip(group.channels, group.diagonal_hamiltonians, strict=True)
        }
        group_drift = {
            channel: drift_generators[channel]
            for channel in group.channels
            if channel in drift_generators
        }
        runtimes.append(
            _ControlGroupRuntime(
                channels=group.channels,
                diagonal_values=diagonal_values,
                drift_generators=group_drift,
                drift_actions={
                    channel: _NilpotentAction.build(generator, nilpotent_order)
                    for channel, generator in group_drift.items()
                },
                half_time_step=half_dt,
            )
        )

    forward_transitions: list[BlockOperator] = []
    backward_transitions: list[BlockOperator] = []
    for current, following in zip(groups[:-1], groups[1:], strict=True):
        forward = following.basis_transform.dag() @ current.basis_transform
        backward = current.basis_transform.dag() @ following.basis_transform
        forward_transitions.append(BlockOperator.physical_diagonal(n_blocks, forward.full()))
        backward_transitions.append(BlockOperator.physical_diagonal(n_blocks, backward.full()))

    return TrotterComponent(
        n_blocks=n_blocks,
        nominal_block=nominal_block,
        penalty_blocks=penalty_blocks,
        penalty_weights=penalty_weights,
        rms_penalties=rms_penalties,
        splitting_steps=splitting_steps,
        initial_transform=initial_transform,
        final_transform=final_transform,
        static_core=static_core,
        group_runtimes=tuple(runtimes),
        forward_transitions=tuple(forward_transitions),
        backward_transitions=tuple(backward_transitions),
    )


def build_trotter_components(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
    groups: tuple[CommutingControlGroup, ...],
    *,
    splitting_steps: int = 1,
) -> tuple[TrotterComponent, ...]:
    """Precompute ST-GRAPE components at an internal splitting resolution.

    ``splitting_steps`` repeats the symmetric product formula inside each PWC
    interval while keeping its control amplitude fixed.  It therefore refines
    propagation without adding pulse parameters or changing the physical grid.
    """

    if isinstance(splitting_steps, bool) or not isinstance(splitting_steps, int):
        raise TypeError("splitting_steps must be an integer.")
    if splitting_steps < 1:
        raise ValueError("splitting_steps must be positive.")

    channel_map = _validate_control_groups(problem, groups)
    components: list[TrotterComponent] = []
    d = problem.hilbert_dimension

    if robustness is not None and robustness.taylor is not None:
        diagonal_controls = {channel: diagonal for channel, (_, diagonal) in channel_map.items()}
        indices, static_generator, drift_generators = build_taylor_generators(
            problem, robustness.taylor, diagonal_controls
        )
        drift_step = np.asarray(
            expm(-1j * problem.drift_hamiltonian.full() * problem.time_step / splitting_steps),
            dtype=np.complex128,
        )
        drift_propagator = BlockOperator.physical_diagonal(len(indices), drift_step)
        penalty_blocks, penalty_weights, rms_penalties = taylor_penalty_terms(
            robustness.taylor,
            indices,
            robustness,
        )
        components.append(
            _build_component(
                problem,
                groups,
                n_blocks=len(indices),
                drift_propagator=drift_propagator,
                error_generator=static_generator,
                nilpotent_order=robustness.taylor.order,
                drift_generators=drift_generators,
                nominal_block=len(indices) - 1,
                penalty_blocks=penalty_blocks,
                penalty_weights=penalty_weights,
                rms_penalties=rms_penalties,
                splitting_steps=splitting_steps,
            )
        )

    if robustness is not None and robustness.colored:
        colored = robustness.colored
        (
            drift_generator,
            error_generator,
            output_blocks,
            nominal_block,
        ) = build_joint_colored_generators(problem, colored)
        drift_steps = [
            np.asarray(
                expm(-1j * block * problem.time_step / splitting_steps),
                dtype=np.complex128,
            )
            for block in drift_generator.blocks
        ]
        drift_propagator = BlockOperator.block_diagonal(drift_steps)
        penalty_blocks, penalty_weights, rms_penalties = joint_colored_penalty_terms(
            colored,
            output_blocks,
            robustness,
        )
        components.append(
            _build_component(
                problem,
                groups,
                n_blocks=drift_generator.n_blocks,
                drift_propagator=drift_propagator,
                error_generator=error_generator,
                nilpotent_order=2,
                drift_generators={},
                nominal_block=nominal_block,
                penalty_blocks=penalty_blocks,
                penalty_weights=penalty_weights,
                rms_penalties=rms_penalties,
                splitting_steps=splitting_steps,
            )
        )

    if not components:
        drift_step = np.asarray(
            expm(-1j * problem.drift_hamiltonian.full() * problem.time_step / splitting_steps),
            dtype=np.complex128,
        )
        zero = BlockOperator.zeros(1, d)
        components.append(
            _build_component(
                problem,
                groups,
                n_blocks=1,
                drift_propagator=BlockOperator.physical_diagonal(1, drift_step),
                error_generator=zero,
                nilpotent_order=0,
                drift_generators={},
                nominal_block=0,
                penalty_blocks=(),
                penalty_weights=(),
                rms_penalties=(),
                splitting_steps=splitting_steps,
            )
        )
    return tuple(components)
