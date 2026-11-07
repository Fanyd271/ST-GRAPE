"""Dense augmented reference GRAPE implementation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from stgrape.problem import ControlProblem, RobustnessModel
from stgrape.propagation.exact import DenseComponent
from stgrape.propagation.ode import OdeComponent, OdePropagationOptions, build_ode_components
from stgrape.pulse import Pulse
from stgrape.result import OptimizationResult

from .options import LBFGSBOptions


class GrapeSolver:
    """Optimize augmented dynamics propagated by an adaptive ODE solver.

    DOP853 with dense matrix products is the default. The
    propagator gradient retains the first-order GRAPE approximation documented
    in the paper.
    """

    def __init__(
        self,
        options: LBFGSBOptions | None = None,
        *,
        propagation: OdePropagationOptions | None = None,
    ) -> None:
        self.options = options or LBFGSBOptions()
        self.propagation = propagation or OdePropagationOptions()

    @staticmethod
    def _evaluate(
        components: Sequence[DenseComponent | OdeComponent],
        problem: ControlProblem,
        controls: np.ndarray,
        *,
        compute_gradient: bool,
    ) -> tuple[float, np.ndarray | None]:
        loss, gradient, _, _ = GrapeSolver._evaluate_decomposed(
            components,
            problem,
            controls,
            compute_gradient=compute_gradient,
        )
        return loss, gradient

    @staticmethod
    def _evaluate_decomposed(
        components: Sequence[DenseComponent | OdeComponent],
        problem: ControlProblem,
        controls: np.ndarray,
        *,
        compute_gradient: bool,
    ) -> tuple[float, np.ndarray | None, float, float]:
        """Evaluate total loss and its fidelity/robustness contributions."""

        fidelity_weight = 1.0 / len(components)
        loss = 0.0
        fidelity_loss = 0.0
        robustness_penalty = 0.0
        gradient = np.zeros_like(controls) if compute_gradient else None
        for component in components:
            evaluation = component.evaluate(
                problem,
                controls,
                fidelity_weight=fidelity_weight,
                compute_gradient=compute_gradient,
            )
            loss += evaluation.objective.loss
            fidelity_loss += fidelity_weight * (1.0 - evaluation.objective.fidelity)
            robustness_penalty += evaluation.objective.penalty
            if gradient is not None and evaluation.amplitude_gradient is not None:
                gradient += evaluation.amplitude_gradient
        return loss, gradient, fidelity_loss, robustness_penalty

    def optimize(
        self,
        problem: ControlProblem,
        pulse: Pulse,
        robustness: RobustnessModel | None = None,
        *,
        time_limit: float | None = None,
        progress_callback: Callable[[int, float, float], None] | None = None,
    ) -> OptimizationResult:
        """Run L-BFGS-B from the supplied filtered pulse.

        ``progress_callback`` receives ``(iteration, loss, elapsed_seconds)``.
        Iteration zero is reported after evaluating the initial pulse.
        """

        if pulse.n_controls != problem.n_controls or pulse.n_time_steps != problem.n_time_steps:
            raise ValueError("pulse dimensions do not match the control problem.")
        components = build_ode_components(problem, robustness, self.propagation)
        initial_parameters = pulse.flattened_coefficients()
        start = time.perf_counter()
        loss_history: list[float] = []
        elapsed_history: list[float] = []
        fidelity_loss_history: list[float] = []
        robustness_penalty_history: list[float] = []
        message = ""
        cached_parameters: np.ndarray | None = None
        cached_fidelity_loss = np.nan
        cached_robustness_penalty = np.nan

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal cached_parameters, cached_fidelity_loss
            nonlocal cached_robustness_penalty
            coefficients = parameters.reshape(pulse.n_controls, pulse.n_parameters_per_control)
            controls = pulse.amplitudes(coefficients)
            loss, amplitude_gradient, fidelity_loss, robustness_penalty = self._evaluate_decomposed(
                components,
                problem,
                controls,
                compute_gradient=True,
            )
            if amplitude_gradient is None:  # pragma: no cover
                raise RuntimeError("GRAPE gradient was not computed.")
            cached_parameters = np.asarray(parameters).copy()
            cached_fidelity_loss = fidelity_loss
            cached_robustness_penalty = robustness_penalty
            return loss, pulse.pullback(amplitude_gradient, coefficients).reshape(-1)

        initial_loss = objective(initial_parameters)[0]
        initial_fidelity_loss = cached_fidelity_loss
        initial_robustness_penalty = cached_robustness_penalty
        if progress_callback is not None:
            progress_callback(0, float(initial_loss), time.perf_counter() - start)

        def callback(intermediate_result: OptimizeResult) -> None:
            nonlocal message, cached_parameters, cached_fidelity_loss
            nonlocal cached_robustness_penalty
            callback_parameters = np.asarray(intermediate_result.x)
            if cached_parameters is None or not np.array_equal(
                callback_parameters, cached_parameters
            ):
                controls = pulse.amplitudes(
                    callback_parameters.reshape(
                        pulse.n_controls,
                        pulse.n_parameters_per_control,
                    )
                )
                _, _, cached_fidelity_loss, cached_robustness_penalty = self._evaluate_decomposed(
                    components,
                    problem,
                    controls,
                    compute_gradient=False,
                )
                cached_parameters = callback_parameters.copy()
            loss_history.append(float(intermediate_result.fun))
            fidelity_loss_history.append(float(cached_fidelity_loss))
            robustness_penalty_history.append(float(cached_robustness_penalty))
            elapsed = time.perf_counter() - start
            elapsed_history.append(elapsed)
            if progress_callback is not None:
                progress_callback(len(loss_history), loss_history[-1], elapsed)
            if time_limit is not None and elapsed > time_limit:
                message = "time limit reached"
                raise StopIteration(message)

        bounds = pulse.parameter_bounds(self.options.max_control_amplitude)
        result = minimize(
            objective,
            initial_parameters,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=self.options.scipy_options(),
        )
        coefficients = result.x.reshape(pulse.n_controls, pulse.n_parameters_per_control)
        final_controls = pulse.amplitudes(coefficients)
        return OptimizationResult(
            method="GRAPE",
            coefficients=coefficients,
            control_amplitudes=final_controls,
            objective=float(result.fun),
            initial_objective=float(initial_loss),
            objective_history=np.asarray(loss_history),
            elapsed_time_history=np.asarray(elapsed_history),
            initial_fidelity_loss=float(initial_fidelity_loss),
            initial_robustness_penalty=float(initial_robustness_penalty),
            fidelity_loss_history=np.asarray(fidelity_loss_history),
            robustness_penalty_history=np.asarray(robustness_penalty_history),
            propagation=self.propagation.metadata(),
            message=message or str(result.message),
            success=bool(result.success),
        )
