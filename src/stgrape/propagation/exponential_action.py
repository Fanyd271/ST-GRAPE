"""Matrix-exponential actions for block-sparse augmented dynamics.

This backend exposes :class:`~stgrape.augmented.BlockOperator` through
SciPy's public ``LinearOperator`` and ``expm_multiply`` interfaces.  It never
forms the full augmented matrix or an interval propagator.  The implementation
is intentionally independent of the production comparison-method registry: it
is a research benchmark for assessing a Krylov exponential-action baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, expm_multiply

from stgrape.augmented import BlockOperator
from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.objectives import ObjectiveValue, SmoothRmsPenalty, terminal_objective
from stgrape.problem import ControlProblem, RobustnessModel

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass
class OperatorCallCounts:
    """Optional low-level call counts reported by SciPy's action algorithm."""

    matvec: int = 0
    matmat: int = 0
    rmatvec: int = 0
    rmatmat: int = 0

    @property
    def total(self) -> int:
        """Total number of ``LinearOperator`` callback invocations."""

        return self.matvec + self.matmat + self.rmatvec + self.rmatmat

    def add(self, other: OperatorCallCounts) -> None:
        """Accumulate another evaluation's counts in place."""

        self.matvec += other.matvec
        self.matmat += other.matmat
        self.rmatvec += other.rmatvec
        self.rmatmat += other.rmatmat


class BlockLinearOperator(LinearOperator):
    """SciPy view of one block operator with complete adjoint callbacks."""

    def __init__(
        self,
        operator: BlockOperator,
        *,
        adjoint_operator: BlockOperator | None = None,
        counts: OperatorCallCounts | None = None,
    ) -> None:
        self.block_operator = operator
        self.adjoint_block_operator = adjoint_operator or operator.adjoint()
        self.counts = counts if counts is not None else OperatorCallCounts()
        dimension = operator.n_blocks * operator.block_dimension
        super().__init__(dtype=np.dtype(np.complex128), shape=(dimension, dimension))

    def _reshape_input(self, value: ComplexArray) -> tuple[ComplexArray, bool]:
        array = np.asarray(value, dtype=np.complex128)
        was_vector = array.ndim == 1
        if was_vector:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.shape[1]:
            raise ValueError(f"value must have shape ({self.shape[1]},) or ({self.shape[1]}, k).")
        return array.reshape(
            self.block_operator.n_blocks,
            self.block_operator.block_dimension,
            array.shape[1],
        ), was_vector

    @staticmethod
    def _flatten_output(value: ComplexArray, *, was_vector: bool) -> ComplexArray:
        flattened = value.reshape(value.shape[0] * value.shape[1], value.shape[2])
        return flattened[:, 0] if was_vector else flattened

    def _matvec(self, value: ComplexArray) -> ComplexArray:
        self.counts.matvec += 1
        block_value, was_vector = self._reshape_input(value)
        return self._flatten_output(self.block_operator.apply(block_value), was_vector=was_vector)

    def _matmat(self, value: ComplexArray) -> ComplexArray:
        self.counts.matmat += 1
        block_value, was_vector = self._reshape_input(value)
        return self._flatten_output(self.block_operator.apply(block_value), was_vector=was_vector)

    def _rmatvec(self, value: ComplexArray) -> ComplexArray:
        self.counts.rmatvec += 1
        block_value, was_vector = self._reshape_input(value)
        return self._flatten_output(
            self.adjoint_block_operator.apply(block_value), was_vector=was_vector
        )

    def _rmatmat(self, value: ComplexArray) -> ComplexArray:
        self.counts.rmatmat += 1
        block_value, was_vector = self._reshape_input(value)
        return self._flatten_output(
            self.adjoint_block_operator.apply(block_value), was_vector=was_vector
        )


@dataclass(frozen=True)
class ExponentialActionInterval:
    """Forward and adjoint actions for one fixed PWC Hamiltonian interval."""

    generator: BlockLinearOperator
    adjoint_generator: BlockLinearOperator
    trace: complex

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: BlockOperator,
        counts: OperatorCallCounts | None = None,
    ) -> ExponentialActionInterval:
        """Prepare ``-i H`` and its adjoint once for one control interval."""

        generator_block = hamiltonian.scaled(-1j)
        adjoint_block = generator_block.adjoint()
        shared_counts = counts if counts is not None else OperatorCallCounts()
        generator = BlockLinearOperator(
            generator_block,
            adjoint_operator=adjoint_block,
            counts=shared_counts,
        )
        adjoint_generator = BlockLinearOperator(
            adjoint_block,
            adjoint_operator=generator_block,
            counts=shared_counts,
        )
        return cls(generator, adjoint_generator, generator_block.trace())

    def forward(self, state: ComplexArray, duration: float) -> ComplexArray:
        """Apply ``exp(-i H duration)`` to a block-shaped state."""

        return _apply_action(self.generator, state, duration, trace=self.trace)

    def backward(self, costate: ComplexArray, duration: float) -> ComplexArray:
        """Apply the adjoint forward generator to a block-shaped costate."""

        return _apply_action(
            self.adjoint_generator,
            costate,
            duration,
            trace=np.conjugate(self.trace),
        )


def _apply_action(
    generator: LinearOperator,
    state: ComplexArray,
    duration: float,
    *,
    trace: complex,
) -> ComplexArray:
    """Flatten a block state, apply one exponential action, and restore it."""

    block_state = np.asarray(state, dtype=np.complex128)
    if block_state.ndim != 3:
        raise ValueError("state must have shape (n_blocks, block_dimension, state_columns).")
    rows = block_state.shape[0] * block_state.shape[1]
    if generator.shape != (rows, rows):
        raise ValueError("generator and block-state dimensions do not match.")
    flattened = block_state.reshape(rows, block_state.shape[2])
    propagated = expm_multiply(
        duration * generator,
        flattened,
        traceA=duration * trace,
    )
    return np.asarray(propagated, dtype=np.complex128).reshape(block_state.shape)


@dataclass(frozen=True)
class ExponentialActionEvaluation:
    """Objective, approximate GRAPE gradient, final state, and callback work."""

    objective: ObjectiveValue
    amplitude_gradient: RealArray | None
    final_state: ComplexArray
    operator_calls: OperatorCallCounts


@dataclass(frozen=True)
class ExponentialActionComponent:
    """One augmented component propagated by Krylov exponential actions."""

    drift_hamiltonian: BlockOperator
    control_hamiltonians: tuple[BlockOperator, ...]
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...]

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the augmented state."""

        return self.drift_hamiltonian.n_blocks

    def _effective_hamiltonian(self, amplitudes: RealArray) -> BlockOperator:
        """Assemble the control-dependent Hamiltonian for one interval."""

        return BlockOperator.linear_combination(
            (self.drift_hamiltonian, *self.control_hamiltonians),
            (1.0, *(complex(amplitude) for amplitude in amplitudes)),
        )

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: RealArray,
        *,
        fidelity_weight: float,
        compute_gradient: bool,
    ) -> ExponentialActionEvaluation:
        """Evaluate the augmented objective and optional approximate gradient."""

        controls = np.asarray(control_amplitudes, dtype=float)
        expected = (problem.n_controls, problem.n_time_steps)
        if controls.shape != expected:
            raise ValueError(f"control_amplitudes has shape {controls.shape}; expected {expected}.")

        dimension = problem.hilbert_dimension
        columns = 1 if problem.task.value == "state" else dimension
        state = np.zeros((self.n_blocks, dimension, columns), dtype=np.complex128)
        state[self.nominal_block] = problem.initial.full()
        forward_states: list[ComplexArray] = []
        intervals: list[ExponentialActionInterval] = []
        counts = OperatorCallCounts()

        for time_index in range(problem.n_time_steps):
            hamiltonian = self._effective_hamiltonian(controls[:, time_index])
            interval = ExponentialActionInterval.from_hamiltonian(hamiltonian, counts)
            state = interval.forward(state, problem.time_step)
            if compute_gradient:
                forward_states.append(state)
                intervals.append(interval)

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
            return ExponentialActionEvaluation(objective, None, state, counts)

        gradient = np.zeros_like(controls)
        for time_index in range(problem.n_time_steps - 1, -1, -1):
            for channel, control in enumerate(self.control_hamiltonians):
                derivative = -1j * problem.time_step * control.apply(forward_states[time_index])
                gradient[channel, time_index] = float(np.vdot(costate, derivative).real)
            costate = intervals[time_index].backward(costate, problem.time_step)
        return ExponentialActionEvaluation(objective, gradient, state, counts)


def build_exponential_action_components(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
) -> tuple[ExponentialActionComponent, ...]:
    """Build exponential-action components from the shared augmented model."""

    return tuple(
        ExponentialActionComponent(
            drift_hamiltonian=component.drift,
            control_hamiltonians=component.controls,
            nominal_block=component.nominal_block,
            penalty_blocks=component.penalty_blocks,
            penalty_weights=component.penalty_weights,
            rms_penalties=component.rms_penalties,
        )
        for component in build_augmented_hamiltonians(problem, robustness)
    )


def evaluate_exponential_action(
    components: tuple[ExponentialActionComponent, ...],
    problem: ControlProblem,
    controls: RealArray,
    *,
    compute_gradient: bool,
) -> tuple[float, RealArray | None, OperatorCallCounts, tuple[ComplexArray, ...]]:
    """Evaluate all components with the same aggregation used by GRAPE."""

    fidelity_weight = 1.0 / len(components)
    loss = 0.0
    gradient = np.zeros_like(controls) if compute_gradient else None
    calls = OperatorCallCounts()
    final_states: list[ComplexArray] = []
    for component in components:
        evaluation = component.evaluate(
            problem,
            controls,
            fidelity_weight=fidelity_weight,
            compute_gradient=compute_gradient,
        )
        loss += evaluation.objective.loss
        calls.add(evaluation.operator_calls)
        final_states.append(evaluation.final_state)
        if gradient is not None and evaluation.amplitude_gradient is not None:
            gradient += evaluation.amplitude_gradient
    return loss, gradient, calls, tuple(final_states)
