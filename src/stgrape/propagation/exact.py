"""Dense augmented propagation used by the GRAPE reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.objectives import ObjectiveValue, SmoothRmsPenalty, terminal_objective
from stgrape.problem import ControlProblem, RobustnessModel

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class DenseEvaluation:
    objective: ObjectiveValue
    amplitude_gradient: NDArray[np.float64] | None
    final_state: ComplexArray


@dataclass(frozen=True)
class DenseComponent:
    """One dense augmented system used for exact objective evaluation."""

    drift_generator: ComplexArray
    control_generators: tuple[ComplexArray, ...]
    n_blocks: int
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...] = ()

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: NDArray[np.float64],
        *,
        fidelity_weight: float,
        compute_gradient: bool,
    ) -> DenseEvaluation:
        """Propagate with dense matrix exponentials.

        The objective is exact for piecewise-constant controls.  The optional
        gradient uses the first-order GRAPE propagator derivative employed by
        the paper's reference implementation.
        """

        controls = np.asarray(control_amplitudes, dtype=float)
        expected = (problem.n_controls, problem.n_time_steps)
        if controls.shape != expected:
            raise ValueError(f"control_amplitudes has shape {controls.shape}; expected {expected}.")

        d = problem.hilbert_dimension
        columns = 1 if problem.task.value == "state" else d
        state = np.zeros((self.n_blocks * d, columns), dtype=np.complex128)
        state[self.nominal_block * d : (self.nominal_block + 1) * d] = problem.initial.full()
        forward_states: list[ComplexArray] = []
        propagators: list[ComplexArray] = []

        for time_index in range(problem.n_time_steps):
            generator = self.drift_generator.copy()
            for channel, control_generator in enumerate(self.control_generators):
                generator += controls[channel, time_index] * control_generator
            propagator = np.asarray(
                expm(-1j * problem.time_step * generator),
                dtype=np.complex128,
            )
            state = propagator @ state
            if compute_gradient:
                forward_states.append(state)
                propagators.append(propagator)

        block_state = state.reshape(self.n_blocks, d, columns)
        objective, block_costate = terminal_objective(
            block_state,
            problem.target,
            problem.task,
            nominal_block=self.nominal_block,
            penalty_blocks=self.penalty_blocks,
            fidelity_weight=fidelity_weight,
            penalty_weights=self.penalty_weights,
            rms_penalties=self.rms_penalties,
        )
        if not compute_gradient:
            return DenseEvaluation(objective, None, block_state)

        gradient = np.zeros(expected, dtype=float)
        costate = block_costate.reshape(self.n_blocks * d, columns)
        for time_index in range(problem.n_time_steps - 1, -1, -1):
            for channel, control_generator in enumerate(self.control_generators):
                derivative = (
                    -1j * problem.time_step * control_generator @ forward_states[time_index]
                )
                gradient[channel, time_index] = float(np.vdot(costate, derivative).real)
            costate = propagators[time_index].conj().T @ costate
        return DenseEvaluation(objective, gradient, block_state)


def build_dense_components(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
) -> tuple[DenseComponent, ...]:
    """Build reference augmented systems for exact objective evaluation."""

    return tuple(
        DenseComponent(
            drift_generator=component.drift.to_dense(),
            control_generators=tuple(control.to_dense() for control in component.controls),
            n_blocks=component.n_blocks,
            nominal_block=component.nominal_block,
            penalty_blocks=component.penalty_blocks,
            penalty_weights=component.penalty_weights,
            rms_penalties=component.rms_penalties,
        )
        for component in build_augmented_hamiltonians(problem, robustness)
    )
