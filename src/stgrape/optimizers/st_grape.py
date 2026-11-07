"""Suzuki--Trotter accelerated robust-control optimizer."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from stgrape.problem import ControlProblem, RobustnessModel
from stgrape.propagation.ode import OdeComponent, OdePropagationOptions, build_ode_components
from stgrape.propagation.trotter import TrotterComponent, build_trotter_components
from stgrape.pulse import Pulse
from stgrape.result import OptimizationResult
from stgrape.systems.spin_chain import CommutingControlGroup

from .grape import GrapeSolver
from .options import LBFGSBOptions

SplittingProgressCallback = Callable[[int, int, int, float], None]


class STGrapeSolver:
    """Optimize the Trotter landscape with true-loss safeguards.

    A true-loss value above the historical best by more than the configured
    absolute/relative tolerance counts as one deterioration. A new historical
    minimum resets the count, and optimization stops when patience is exhausted.

    When adaptive splitting is enabled, the internal Trotter grid is refined as
    soon as an exact check satisfies
    ``|J_exact - J_ST| > atol + rtol * |J_exact|``.
    """

    def __init__(
        self,
        options: LBFGSBOptions | None = None,
        *,
        exact_check_interval: int | None = 25,
        exact_increase_patience: int = 2,
        exact_increase_absolute_tolerance: float = 1e-8,
        exact_increase_relative_tolerance: float = 1e-3,
        final_exact_evaluation: bool = True,
        reference_propagation: OdePropagationOptions | None = None,
        splitting_steps: int = 1,
        adaptive_splitting: bool = False,
        max_splitting_steps: int = 1,
        splitting_relative_tolerance: float = 0.15,
        splitting_absolute_tolerance: float = 2e-5,
    ) -> None:
        if exact_check_interval is not None and exact_check_interval < 1:
            raise ValueError("exact_check_interval must be positive or None.")
        if exact_increase_patience < 1:
            raise ValueError("exact_increase_patience must be positive.")
        if exact_increase_absolute_tolerance < 0:
            raise ValueError("exact_increase_absolute_tolerance must be non-negative.")
        if exact_increase_relative_tolerance < 0:
            raise ValueError("exact_increase_relative_tolerance must be non-negative.")
        if isinstance(splitting_steps, bool) or not isinstance(splitting_steps, int):
            raise TypeError("splitting_steps must be an integer.")
        if isinstance(max_splitting_steps, bool) or not isinstance(max_splitting_steps, int):
            raise TypeError("max_splitting_steps must be an integer.")
        if splitting_steps < 1:
            raise ValueError("splitting_steps must be positive.")
        if max_splitting_steps < splitting_steps:
            raise ValueError("max_splitting_steps must be at least splitting_steps.")
        if splitting_relative_tolerance < 0:
            raise ValueError("splitting_relative_tolerance must be non-negative.")
        if splitting_absolute_tolerance < 0:
            raise ValueError("splitting_absolute_tolerance must be non-negative.")
        if splitting_relative_tolerance == 0 and splitting_absolute_tolerance == 0:
            raise ValueError("At least one splitting tolerance must be positive.")
        if adaptive_splitting and exact_check_interval is None:
            raise ValueError("adaptive_splitting requires periodic true-loss checks.")
        self.options = options or LBFGSBOptions()
        self.exact_check_interval = exact_check_interval
        self.exact_increase_patience = exact_increase_patience
        self.exact_increase_absolute_tolerance = exact_increase_absolute_tolerance
        self.exact_increase_relative_tolerance = exact_increase_relative_tolerance
        self.final_exact_evaluation = final_exact_evaluation
        self.reference_propagation = reference_propagation or OdePropagationOptions()
        self.splitting_steps = splitting_steps
        self.adaptive_splitting = adaptive_splitting
        self.max_splitting_steps = max_splitting_steps
        self.splitting_relative_tolerance = splitting_relative_tolerance
        self.splitting_absolute_tolerance = splitting_absolute_tolerance

    @staticmethod
    def _evaluate_trotter(
        components: tuple[TrotterComponent, ...],
        problem: ControlProblem,
        controls: np.ndarray,
        *,
        compute_gradient: bool,
    ) -> tuple[float, np.ndarray | None]:
        loss, gradient, _, _ = STGrapeSolver._evaluate_trotter_decomposed(
            components,
            problem,
            controls,
            compute_gradient=compute_gradient,
        )
        return loss, gradient

    @staticmethod
    def _evaluate_trotter_decomposed(
        components: tuple[TrotterComponent, ...],
        problem: ControlProblem,
        controls: np.ndarray,
        *,
        compute_gradient: bool,
    ) -> tuple[float, np.ndarray | None, float, float]:
        """Evaluate the ST surrogate and retain its fidelity/penalty split."""

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
        control_groups: tuple[CommutingControlGroup, ...],
        robustness: RobustnessModel | None = None,
        *,
        time_limit: float | None = None,
        progress_callback: Callable[[int, float, float], None] | None = None,
        exact_progress_callback: Callable[[int, float, float], None] | None = None,
        splitting_progress_callback: SplittingProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run ST-GRAPE and periodically safeguard the exact objective.

        ``progress_callback`` receives ``(iteration, loss, elapsed_seconds)``.
        Iteration zero is reported after evaluating the initial pulse.
        ``exact_progress_callback`` receives the same tuple whenever the true
        augmented-system loss is evaluated by ``reference_propagation``.
        """

        if pulse.n_controls != problem.n_controls or pulse.n_time_steps != problem.n_time_steps:
            raise ValueError("pulse dimensions do not match the control problem.")
        current_splitting_steps = self.splitting_steps
        trotter_components = build_trotter_components(
            problem,
            robustness,
            control_groups,
            splitting_steps=current_splitting_steps,
        )
        reference_components: tuple[OdeComponent, ...] | None = None
        if self.exact_check_interval is not None or self.final_exact_evaluation:
            reference_components = build_ode_components(
                problem, robustness, self.reference_propagation
            )

        initial_parameters = pulse.flattened_coefficients()
        start = time.perf_counter()
        loss_history: list[float] = []
        elapsed_history: list[float] = []
        fidelity_loss_history: list[float] = []
        robustness_penalty_history: list[float] = []
        exact_history: list[float] = []
        exact_times: list[float] = []
        exact_iterations: list[int] = []
        exact_approximate_history: list[float] = []
        exact_gap_ratio_history: list[float] = []
        exact_absolute_error_history: list[float] = []
        exact_splitting_tolerance_history: list[float] = []
        exact_splitting_tolerance_exceeded_history: list[bool] = []
        splitting_steps_history: list[int] = []
        splitting_change_iterations = [0]
        splitting_change_steps = [current_splitting_steps]
        splitting_notes: list[str] = []
        best_exact = np.inf
        best_exact_iteration: int | None = None
        best_parameters: np.ndarray | None = None
        last_exact_parameters: np.ndarray | None = None
        exact_deterioration_count = 0
        safeguard_termination_iteration: int | None = None
        stopped_by_safeguard = False
        message = ""
        cached_parameters: np.ndarray | None = None
        cached_fidelity_loss = np.nan
        cached_robustness_penalty = np.nan

        def controls_from(parameters: np.ndarray) -> np.ndarray:
            return pulse.amplitudes(
                parameters.reshape(pulse.n_controls, pulse.n_parameters_per_control)
            )

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal cached_parameters, cached_fidelity_loss
            nonlocal cached_robustness_penalty
            coefficients = parameters.reshape(
                pulse.n_controls,
                pulse.n_parameters_per_control,
            )
            controls = pulse.amplitudes(coefficients)
            (
                loss,
                amplitude_gradient,
                cached_fidelity_loss,
                cached_robustness_penalty,
            ) = self._evaluate_trotter_decomposed(
                trotter_components, problem, controls, compute_gradient=True
            )
            if amplitude_gradient is None:  # pragma: no cover
                raise RuntimeError("ST-GRAPE gradient was not computed.")
            cached_parameters = np.asarray(parameters).copy()
            return loss, pulse.pullback(amplitude_gradient, coefficients).reshape(-1)

        initial_loss = objective(initial_parameters)[0]
        initial_fidelity_loss = cached_fidelity_loss
        initial_robustness_penalty = cached_robustness_penalty
        if progress_callback is not None:
            progress_callback(0, float(initial_loss), time.perf_counter() - start)
        iteration = 0

        def evaluate_exact(
            parameters: np.ndarray,
            evaluation_iteration: int,
            approximate_value: float,
        ) -> tuple[bool, float, float, float]:
            """Evaluate true loss, its discrepancy, and the safeguard state."""

            nonlocal best_exact, best_exact_iteration, best_parameters
            nonlocal exact_deterioration_count, last_exact_parameters
            if reference_components is None:  # pragma: no cover - guarded by caller
                raise RuntimeError("Exact reference components were not constructed.")
            exact_loss, _ = GrapeSolver._evaluate(
                reference_components,
                problem,
                controls_from(parameters),
                compute_gradient=False,
            )
            exact_value = float(exact_loss)
            exact_elapsed = time.perf_counter() - start
            exact_history.append(exact_value)
            exact_times.append(exact_elapsed)
            exact_iterations.append(evaluation_iteration)
            exact_approximate_history.append(float(approximate_value))
            absolute_error = float(abs(exact_value - approximate_value))
            gap_ratio = float(absolute_error / max(abs(exact_value), np.finfo(float).tiny))
            splitting_tolerance = float(
                self.splitting_absolute_tolerance
                + self.splitting_relative_tolerance * abs(exact_value)
            )
            exact_gap_ratio_history.append(gap_ratio)
            exact_absolute_error_history.append(absolute_error)
            exact_splitting_tolerance_history.append(splitting_tolerance)
            exact_splitting_tolerance_exceeded_history.append(absolute_error > splitting_tolerance)
            last_exact_parameters = np.asarray(parameters).copy()
            if exact_progress_callback is not None:
                exact_progress_callback(evaluation_iteration, exact_value, exact_elapsed)

            if best_parameters is None or exact_value < best_exact:
                best_exact = exact_value
                best_exact_iteration = evaluation_iteration
                best_parameters = np.asarray(parameters).copy()
                exact_deterioration_count = 0
                return False, gap_ratio, absolute_error, splitting_tolerance

            tolerance = max(
                self.exact_increase_absolute_tolerance,
                self.exact_increase_relative_tolerance * abs(best_exact),
            )
            if exact_value > best_exact + tolerance:
                exact_deterioration_count += 1
            # Values inside the tolerance band neither increase the counter nor
            # replace the incumbent. Only a new true-loss minimum resets it.
            return (
                exact_deterioration_count >= self.exact_increase_patience,
                gap_ratio,
                absolute_error,
                splitting_tolerance,
            )

        def refinement_is_available(
            absolute_error: float,
            tolerance: float,
        ) -> bool:
            """Return whether the mixed-error test requests a finer ST stage."""

            if absolute_error <= tolerance:
                return False

            if not self.adaptive_splitting:
                return False
            if current_splitting_steps >= self.max_splitting_steps:
                return False
            if iteration >= self.options.max_iterations:
                return False
            return time_limit is None or time.perf_counter() - start <= time_limit

        requested_splitting_steps: int | None = None
        restart_parameters: np.ndarray | None = None

        def request_refinement(parameters: np.ndarray, gap_ratio: float) -> None:
            """Record a doubling request; the outer loop rebuilds and restarts."""

            nonlocal requested_splitting_steps, restart_parameters
            requested_splitting_steps = min(
                2 * current_splitting_steps,
                self.max_splitting_steps,
            )
            restart_parameters = np.asarray(parameters).copy()
            if splitting_progress_callback is not None:
                splitting_progress_callback(
                    iteration,
                    current_splitting_steps,
                    requested_splitting_steps,
                    gap_ratio,
                )

        # Seed the safeguard with the unmodified nominal pulse.  Without this
        # check, a first optimizer step that worsens the true objective could
        # become the incumbent simply because iteration zero was never seen by
        # the safeguard.  Final-only large-scale runs deliberately retain their
        # single terminal reference evaluation by skipping this initialization
        # when periodic checking is disabled.
        if reference_components is not None and self.exact_check_interval is not None:
            evaluate_exact(initial_parameters, 0, float(initial_loss))

        def callback(intermediate_result: OptimizeResult) -> None:
            nonlocal iteration, message, safeguard_termination_iteration
            nonlocal stopped_by_safeguard
            nonlocal cached_parameters, cached_fidelity_loss
            nonlocal cached_robustness_penalty
            iteration += 1
            callback_parameters = np.asarray(intermediate_result.x)
            if cached_parameters is None or not np.array_equal(
                callback_parameters,
                cached_parameters,
            ):
                (
                    _,
                    _,
                    cached_fidelity_loss,
                    cached_robustness_penalty,
                ) = self._evaluate_trotter_decomposed(
                    trotter_components,
                    problem,
                    controls_from(callback_parameters),
                    compute_gradient=False,
                )
                cached_parameters = callback_parameters.copy()
            elapsed = time.perf_counter() - start
            loss_history.append(float(intermediate_result.fun))
            elapsed_history.append(elapsed)
            fidelity_loss_history.append(float(cached_fidelity_loss))
            robustness_penalty_history.append(float(cached_robustness_penalty))
            splitting_steps_history.append(current_splitting_steps)
            if progress_callback is not None:
                progress_callback(iteration, loss_history[-1], elapsed)
            if time_limit is not None and elapsed > time_limit:
                message = "time limit reached"
                raise StopIteration(message)

            if (
                reference_components is not None
                and self.exact_check_interval is not None
                and iteration % self.exact_check_interval == 0
            ):
                (
                    patience_exhausted,
                    gap_ratio,
                    absolute_error,
                    splitting_tolerance,
                ) = evaluate_exact(
                    np.asarray(intermediate_result.x),
                    iteration,
                    float(intermediate_result.fun),
                )
                if refinement_is_available(absolute_error, splitting_tolerance):
                    request_refinement(np.asarray(intermediate_result.x), gap_ratio)
                    raise StopIteration("adaptive splitting refinement requested")
                if patience_exhausted:
                    stopped_by_safeguard = True
                    safeguard_termination_iteration = iteration
                    message = (
                        f"true loss deteriorated {exact_deterioration_count} times; "
                        "retained the best true-loss controls"
                    )
                    raise StopIteration(message)

        bounds = pulse.parameter_bounds(self.options.max_control_amplitude)
        stage_parameters = initial_parameters
        result: OptimizeResult | None = None
        while iteration < self.options.max_iterations:
            requested_splitting_steps = None
            restart_parameters = None
            stage_options = self.options.scipy_options()
            stage_options["maxiter"] = self.options.max_iterations - iteration
            stage_result = minimize(
                objective,
                stage_parameters,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options=stage_options,
            )
            result = stage_result

            # Match the paper-era implementation: if an optimizer stage ends
            # between scheduled checks, evaluate its final candidate once.
            if (
                requested_splitting_steps is None
                and reference_components is not None
                and self.final_exact_evaluation
                and not stopped_by_safeguard
            ):
                candidate = np.asarray(stage_result.x)
                if last_exact_parameters is None or not np.array_equal(
                    candidate, last_exact_parameters
                ):
                    (
                        patience_exhausted,
                        gap_ratio,
                        absolute_error,
                        splitting_tolerance,
                    ) = evaluate_exact(
                        candidate,
                        iteration,
                        float(stage_result.fun),
                    )
                    if refinement_is_available(absolute_error, splitting_tolerance):
                        request_refinement(candidate, gap_ratio)
                    elif patience_exhausted:
                        safeguard_termination_iteration = iteration
                        if not message:
                            message = (
                                "final true loss exhausted patience "
                                f"({exact_deterioration_count} deteriorations); "
                                "retained the best true-loss controls"
                            )

            if requested_splitting_steps is None or restart_parameters is None:
                break

            previous_steps = current_splitting_steps
            current_splitting_steps = requested_splitting_steps
            stage_parameters = restart_parameters
            trotter_components = build_trotter_components(
                problem,
                robustness,
                control_groups,
                splitting_steps=current_splitting_steps,
            )
            splitting_change_iterations.append(iteration)
            splitting_change_steps.append(current_splitting_steps)
            splitting_notes.append(
                f"splitting steps {previous_steps}->{current_splitting_steps} "
                f"at iteration {iteration}"
            )
            # The true-loss incumbent remains global, but the newly refined
            # surrogate receives a fresh deterioration-safeguard window.
            exact_deterioration_count = 0

        if result is None:  # pragma: no cover - max_iterations is validated upstream
            raise RuntimeError("ST-GRAPE did not start an optimization stage.")

        parameters = best_parameters if best_parameters is not None else result.x
        coefficients = parameters.reshape(pulse.n_controls, pulse.n_parameters_per_control)
        controls = pulse.amplitudes(coefficients)
        final_loss, _ = self._evaluate_trotter(
            trotter_components, problem, controls, compute_gradient=False
        )
        return OptimizationResult(
            method="ST-GRAPE",
            coefficients=coefficients,
            control_amplitudes=controls,
            objective=float(final_loss),
            initial_objective=float(initial_loss),
            objective_history=np.asarray(loss_history),
            elapsed_time_history=np.asarray(elapsed_history),
            initial_fidelity_loss=float(initial_fidelity_loss),
            initial_robustness_penalty=float(initial_robustness_penalty),
            fidelity_loss_history=np.asarray(fidelity_loss_history),
            robustness_penalty_history=np.asarray(robustness_penalty_history),
            exact_objective_history=np.asarray(exact_history),
            exact_evaluation_times=np.asarray(exact_times),
            exact_evaluation_iterations=np.asarray(exact_iterations, dtype=np.int64),
            exact_approximate_objective_history=np.asarray(
                exact_approximate_history,
                dtype=float,
            ),
            exact_approximation_gap_ratio_history=np.asarray(
                exact_gap_ratio_history,
                dtype=float,
            ),
            exact_approximation_absolute_error_history=np.asarray(
                exact_absolute_error_history,
                dtype=float,
            ),
            exact_splitting_tolerance_history=np.asarray(
                exact_splitting_tolerance_history,
                dtype=float,
            ),
            exact_splitting_tolerance_exceeded_history=np.asarray(
                exact_splitting_tolerance_exceeded_history,
                dtype=bool,
            ),
            trotter_splitting_steps_history=np.asarray(
                splitting_steps_history,
                dtype=np.int64,
            ),
            splitting_change_iterations=np.asarray(
                splitting_change_iterations,
                dtype=np.int64,
            ),
            splitting_change_steps=np.asarray(
                splitting_change_steps,
                dtype=np.int64,
            ),
            exact_objective=(None if best_parameters is None else float(best_exact)),
            exact_objective_iteration=best_exact_iteration,
            exact_deterioration_count=exact_deterioration_count,
            exact_safeguard_termination_iteration=safeguard_termination_iteration,
            propagation={
                "representation": "suzuki-trotter",
                "reference_check": self.reference_propagation.metadata(),
                "reference_check_interval": self.exact_check_interval,
                "final_reference_evaluation": self.final_exact_evaluation,
                "reference_increase_patience": self.exact_increase_patience,
                "reference_increase_absolute_tolerance": (self.exact_increase_absolute_tolerance),
                "reference_increase_relative_tolerance": (self.exact_increase_relative_tolerance),
                "initial_splitting_steps": self.splitting_steps,
                "final_splitting_steps": current_splitting_steps,
                "adaptive_splitting": self.adaptive_splitting,
                "maximum_splitting_steps": self.max_splitting_steps,
                "splitting_error_criterion": "absolute-plus-relative",
                "splitting_relative_tolerance": self.splitting_relative_tolerance,
                "splitting_absolute_tolerance": self.splitting_absolute_tolerance,
            },
            message="; ".join((*splitting_notes, message or str(result.message))),
            success=bool(result.success),
        )
