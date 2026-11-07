"""SciPy DOP853 propagation for dense and block-sparse augmented systems.

Augmented states are never normalized: Taylor and Van Loan blocks are not
physical state vectors. Each piecewise-constant interval is integrated separately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from math import inf, isinf
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from stgrape.augmented.block_sparse import BlockOperator
from stgrape.augmented.builders import AugmentedHamiltonian, build_augmented_hamiltonians
from stgrape.objectives import ObjectiveValue, SmoothRmsPenalty, terminal_objective
from stgrape.problem import ControlProblem, RobustnessModel

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
OdeRightHandSide = Callable[[float, ComplexArray], ComplexArray]


class OdeRepresentation(StrEnum):
    """Linear-algebra representation used to evaluate ``G @ x``."""

    DENSE = "dense"
    BLOCK_SPARSE = "block-sparse"


@dataclass(frozen=True)
class DOP853Options:
    """High-accuracy reference settings for SciPy's DOP853 solver.

    The project uses tightened tolerances because the
    resulting objective is used as the common true-loss reference.
    """

    rtol: float = 1e-9
    atol: float = 1e-11
    first_step: float | None = None
    max_step: float = inf
    method: str = field(default="dop853", init=False)

    def __post_init__(self) -> None:
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("rtol and atol must be positive.")
        if self.first_step is not None and self.first_step <= 0:
            raise ValueError("first_step must be positive or None.")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive.")

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe solver metadata."""

        return {
            "method": self.method,
            "implementation": "scipy.integrate.solve_ivp",
            "preset": "reference-dop853",
            "rtol": self.rtol,
            "atol": self.atol,
            "first_step": self.first_step,
            "max_step": "inf" if isinf(self.max_step) else self.max_step,
        }


def default_ode_solver_options(method: str = "dop853") -> DOP853Options:
    """Return the reference integrator; obsolete solver aliases are rejected."""
    if method.lower() != "dop853":
        raise ValueError("ODE method must be 'dop853'.")
    return DOP853Options()


@dataclass(frozen=True)
class OdePropagationOptions:
    """ODE method and matrix representation selected by an optimizer."""

    solver: DOP853Options = field(default_factory=DOP853Options)
    representation: OdeRepresentation = OdeRepresentation.DENSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "representation", OdeRepresentation(self.representation))

    def metadata(self) -> dict[str, object]:
        """Return a complete JSON-safe propagation description."""

        return {
            "representation": self.representation.value,
            "solver": self.solver.metadata(),
            "normalize_output": False,
        }


@dataclass(frozen=True)
class OdeStepResult:
    """Endpoint and diagnostics from one control interval."""

    final_state: ComplexArray
    rhs_evaluations: int


class OdeIntegrationError(RuntimeError):
    """Raised when an adaptive integrator cannot reach the interval endpoint."""


def integrate_ode_interval(
    rhs: OdeRightHandSide,
    initial_state: ComplexArray,
    duration: float,
    options: DOP853Options,
) -> OdeStepResult:
    """Integrate an arbitrary-shaped complex state over one control interval."""

    if duration <= 0:
        raise ValueError("duration must be positive.")
    state = np.asarray(initial_state, dtype=np.complex128)
    if state.size == 0:
        raise ValueError("initial_state cannot be empty.")
    state_shape = state.shape
    rhs_evaluations = 0

    def flattened_rhs(time: float, flattened_state: ComplexArray) -> ComplexArray:
        nonlocal rhs_evaluations
        rhs_evaluations += 1
        shaped_state = np.asarray(flattened_state, dtype=np.complex128).reshape(state_shape)
        derivative = np.asarray(rhs(time, shaped_state), dtype=np.complex128)
        if derivative.shape != state_shape:
            raise ValueError(
                f"ODE right-hand side returned shape {derivative.shape}; expected {state_shape}."
            )
        return derivative.reshape(-1)

    flattened_initial = state.reshape(-1)
    solution = solve_ivp(
        flattened_rhs,
        (0.0, duration),
        flattened_initial,
        method="DOP853",
        t_eval=(duration,),
        rtol=options.rtol,
        atol=options.atol,
        first_step=options.first_step,
        max_step=options.max_step,
    )
    if not solution.success:
        raise OdeIntegrationError(f"DOP853 failed: {solution.message}")
    final_state = np.asarray(solution.y[:, -1], dtype=np.complex128).reshape(state_shape)

    return OdeStepResult(final_state, rhs_evaluations)


@dataclass(frozen=True)
class OdeEvaluation:
    """Objective, GRAPE gradient, and ODE work for one component."""

    objective: ObjectiveValue
    amplitude_gradient: RealArray | None
    final_state: ComplexArray
    rhs_evaluations: int


class OdeComponent(Protocol):
    """Common evaluation contract for dense and block-sparse ODE systems."""

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: RealArray,
        *,
        fidelity_weight: float,
        compute_gradient: bool,
    ) -> OdeEvaluation: ...


def _validated_controls(problem: ControlProblem, control_amplitudes: RealArray) -> RealArray:
    """Convert controls to real amplitudes and enforce the problem grid shape."""

    controls = np.asarray(control_amplitudes, dtype=float)
    expected = (problem.n_controls, problem.n_time_steps)
    if controls.shape != expected:
        raise ValueError(f"control_amplitudes has shape {controls.shape}; expected {expected}.")
    return controls


@dataclass(frozen=True)
class DenseOdeComponent:
    """One augmented system propagated through dense matrix--state products."""

    drift_hamiltonian: ComplexArray
    control_hamiltonians: tuple[ComplexArray, ...]
    n_blocks: int
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...]
    solver_options: DOP853Options

    def _effective_hamiltonian(self, amplitudes: RealArray) -> ComplexArray:
        """Assemble the dense piecewise-constant Hamiltonian for one interval."""

        hamiltonian = self.drift_hamiltonian.copy()
        for amplitude, control in zip(amplitudes, self.control_hamiltonians, strict=True):
            hamiltonian += amplitude * control
        return hamiltonian

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: RealArray,
        *,
        fidelity_weight: float,
        compute_gradient: bool,
    ) -> OdeEvaluation:
        """Propagate forward and, if requested, solve the adjoint LMDE."""

        controls = _validated_controls(problem, control_amplitudes)
        block_dimension = problem.hilbert_dimension
        columns = 1 if problem.task.value == "state" else block_dimension
        state = np.zeros((self.n_blocks * block_dimension, columns), dtype=np.complex128)
        start = self.nominal_block * block_dimension
        state[start : start + block_dimension] = problem.initial.full()
        forward_states: list[ComplexArray] = []
        rhs_evaluations = 0

        for time_index in range(problem.n_time_steps):
            hamiltonian = self._effective_hamiltonian(controls[:, time_index])
            step = integrate_ode_interval(
                lambda _time, value, matrix=hamiltonian: -1j * (matrix @ value),
                state,
                problem.time_step,
                self.solver_options,
            )
            state = step.final_state
            rhs_evaluations += step.rhs_evaluations
            if compute_gradient:
                forward_states.append(state)

        block_state = state.reshape(self.n_blocks, block_dimension, columns)
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
            return OdeEvaluation(objective, None, block_state, rhs_evaluations)

        gradient = np.zeros_like(controls)
        costate = block_costate.reshape(self.n_blocks * block_dimension, columns)
        for time_index in range(problem.n_time_steps - 1, -1, -1):
            for channel, control in enumerate(self.control_hamiltonians):
                derivative = -1j * problem.time_step * (control @ forward_states[time_index])
                gradient[channel, time_index] = float(np.vdot(costate, derivative).real)

            hamiltonian = self._effective_hamiltonian(controls[:, time_index])
            adjoint = hamiltonian.conj().T
            step = integrate_ode_interval(
                lambda _time, value, matrix=adjoint: 1j * (matrix @ value),
                costate,
                problem.time_step,
                self.solver_options,
            )
            costate = step.final_state
            rhs_evaluations += step.rhs_evaluations
        return OdeEvaluation(objective, gradient, block_state, rhs_evaluations)


@dataclass(frozen=True)
class BlockSparseOdeComponent:
    """One augmented system propagated without materializing a dense matrix."""

    drift_hamiltonian: BlockOperator
    control_hamiltonians: tuple[BlockOperator, ...]
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...]
    solver_options: DOP853Options

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the augmented state."""

        return self.drift_hamiltonian.n_blocks

    def _effective_hamiltonian(
        self,
        amplitudes: RealArray,
        *,
        adjoint: bool,
    ) -> BlockOperator:
        """Assemble one block-sparse Hamiltonian, optionally in adjoint form."""

        hamiltonian = BlockOperator.linear_combination(
            (self.drift_hamiltonian, *self.control_hamiltonians),
            (1.0, *(complex(amplitude) for amplitude in amplitudes)),
        )
        return hamiltonian.adjoint() if adjoint else hamiltonian

    def evaluate(
        self,
        problem: ControlProblem,
        control_amplitudes: RealArray,
        *,
        fidelity_weight: float,
        compute_gradient: bool,
    ) -> OdeEvaluation:
        """Propagate a block-shaped state with sparse block products."""

        controls = _validated_controls(problem, control_amplitudes)
        block_dimension = problem.hilbert_dimension
        columns = 1 if problem.task.value == "state" else block_dimension
        state = np.zeros((self.n_blocks, block_dimension, columns), dtype=np.complex128)
        state[self.nominal_block] = problem.initial.full()
        forward_states: list[ComplexArray] = []
        rhs_evaluations = 0

        for time_index in range(problem.n_time_steps):
            amplitudes = controls[:, time_index]
            hamiltonian = self._effective_hamiltonian(amplitudes, adjoint=False)
            step = integrate_ode_interval(
                lambda _time, value, operator=hamiltonian: -1j * operator.apply(value),
                state,
                problem.time_step,
                self.solver_options,
            )
            state = step.final_state
            rhs_evaluations += step.rhs_evaluations
            if compute_gradient:
                forward_states.append(state)

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
            return OdeEvaluation(objective, None, state, rhs_evaluations)

        gradient = np.zeros_like(controls)
        for time_index in range(problem.n_time_steps - 1, -1, -1):
            for channel, control in enumerate(self.control_hamiltonians):
                derivative = -1j * problem.time_step * control.apply(forward_states[time_index])
                gradient[channel, time_index] = float(np.vdot(costate, derivative).real)

            amplitudes = controls[:, time_index]
            adjoint_hamiltonian = self._effective_hamiltonian(amplitudes, adjoint=True)
            step = integrate_ode_interval(
                lambda _time, value, operator=adjoint_hamiltonian: 1j * operator.apply(value),
                costate,
                problem.time_step,
                self.solver_options,
            )
            costate = step.final_state
            rhs_evaluations += step.rhs_evaluations
        return OdeEvaluation(objective, gradient, state, rhs_evaluations)


def _dense_component(
    component: AugmentedHamiltonian,
    options: DOP853Options,
) -> DenseOdeComponent:
    """Materialize one shared augmented description as dense NumPy arrays."""

    return DenseOdeComponent(
        drift_hamiltonian=component.drift.to_dense(),
        control_hamiltonians=tuple(control.to_dense() for control in component.controls),
        n_blocks=component.n_blocks,
        nominal_block=component.nominal_block,
        penalty_blocks=component.penalty_blocks,
        penalty_weights=component.penalty_weights,
        rms_penalties=component.rms_penalties,
        solver_options=options,
    )


def _block_sparse_component(
    component: AugmentedHamiltonian,
    options: DOP853Options,
) -> BlockSparseOdeComponent:
    """Wrap one shared augmented description without dense materialization."""

    return BlockSparseOdeComponent(
        drift_hamiltonian=component.drift,
        control_hamiltonians=component.controls,
        nominal_block=component.nominal_block,
        penalty_blocks=component.penalty_blocks,
        penalty_weights=component.penalty_weights,
        rms_penalties=component.rms_penalties,
        solver_options=options,
    )


def build_ode_components(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
    options: OdePropagationOptions | None = None,
) -> tuple[OdeComponent, ...]:
    """Build dense or block-sparse components sharing identical physics."""

    propagation = options or OdePropagationOptions()
    augmented = build_augmented_hamiltonians(problem, robustness)
    if propagation.representation == OdeRepresentation.DENSE:
        return tuple(_dense_component(component, propagation.solver) for component in augmented)
    return tuple(_block_sparse_component(component, propagation.solver) for component in augmented)
