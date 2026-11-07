"""High-level configuration and execution of one optimization run.

The numerical kernels intentionally know nothing about paper experiments.
This module provides the small composition layer users need to combine their
own system, uncertainty model, pulse parameterization, optimizer, and ODE
configuration without editing GRAPE or ST-GRAPE internals.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from .optimizers import GrapeSolver, LBFGSBOptions, STGrapeSolver
from .problem import ControlProblem, RobustnessModel
from .propagation import (
    OdePropagationOptions,
    OdeRepresentation,
    default_ode_solver_options,
)
from .pulse import Pulse
from .result import OptimizationResult
from .systems import CommutingControlGroup

ProgressCallback = Callable[[int, float, float], None]
SplittingProgressCallback = Callable[[int, int, int, float], None]


class OptimizationAlgorithm(StrEnum):
    """Optimization landscapes exposed by the public workflow API."""

    GRAPE = "grape"
    ST_GRAPE = "st-grape"


def configure_ode_propagation(
    *,
    method: str = "dop853",
    representation: OdeRepresentation | str = OdeRepresentation.DENSE,
    rtol: float | None = None,
    atol: float | None = None,
) -> OdePropagationOptions:
    """Configure SciPy DOP853 and the augmented-state representation."""

    solver = default_ode_solver_options(method)
    if rtol is not None:
        solver = replace(solver, rtol=rtol)
    if atol is not None:
        solver = replace(solver, atol=atol)
    return OdePropagationOptions(
        solver=solver,
        representation=OdeRepresentation(representation),
    )


@dataclass(frozen=True)
class OptimizationConfig:
    """All algorithmic choices for one GRAPE or ST-GRAPE run.

    ``propagation`` controls GRAPE evolution and ST-GRAPE true-loss checks.
    The ST-GRAPE optimization landscape itself always uses the structure-aware
    Suzuki--Trotter propagator.
    """

    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.ST_GRAPE
    optimizer: LBFGSBOptions = field(default_factory=LBFGSBOptions)
    propagation: OdePropagationOptions = field(default_factory=OdePropagationOptions)
    time_limit_seconds: float | None = None
    exact_check_interval: int | None = 25
    exact_increase_patience: int = 2
    exact_increase_absolute_tolerance: float = 1e-8
    exact_increase_relative_tolerance: float = 1e-3
    final_exact_evaluation: bool = True
    splitting_steps: int = 1
    adaptive_splitting: bool = False
    max_splitting_steps: int = 1
    splitting_relative_tolerance: float = 0.15
    splitting_absolute_tolerance: float = 2e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", OptimizationAlgorithm(self.algorithm))
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None.")
        if self.exact_check_interval is not None and self.exact_check_interval < 1:
            raise ValueError("exact_check_interval must be positive or None.")
        if self.exact_increase_patience < 1:
            raise ValueError("exact_increase_patience must be positive.")
        if self.exact_increase_absolute_tolerance < 0:
            raise ValueError("exact_increase_absolute_tolerance must be non-negative.")
        if self.exact_increase_relative_tolerance < 0:
            raise ValueError("exact_increase_relative_tolerance must be non-negative.")
        if isinstance(self.splitting_steps, bool) or not isinstance(self.splitting_steps, int):
            raise TypeError("splitting_steps must be an integer.")
        if isinstance(self.max_splitting_steps, bool) or not isinstance(
            self.max_splitting_steps, int
        ):
            raise TypeError("max_splitting_steps must be an integer.")
        if self.splitting_steps < 1:
            raise ValueError("splitting_steps must be positive.")
        if self.max_splitting_steps < self.splitting_steps:
            raise ValueError("max_splitting_steps must be at least splitting_steps.")
        if self.splitting_relative_tolerance < 0:
            raise ValueError("splitting_relative_tolerance must be non-negative.")
        if self.splitting_absolute_tolerance < 0:
            raise ValueError("splitting_absolute_tolerance must be non-negative.")
        if self.splitting_relative_tolerance == 0 and self.splitting_absolute_tolerance == 0:
            raise ValueError("At least one splitting tolerance must be positive.")
        if self.adaptive_splitting and self.exact_check_interval is None:
            raise ValueError("adaptive_splitting requires periodic true-loss checks.")

    def metadata(self) -> dict[str, object]:
        """Return a complete JSON-safe record of the numerical configuration."""

        return {
            "algorithm": self.algorithm.value,
            "optimizer": {
                "method": "L-BFGS-B",
                "max_iterations": self.optimizer.max_iterations,
                "max_function_evaluations": (self.optimizer.max_function_evaluations),
                "effective_max_function_evaluations": (
                    self.optimizer.resolved_max_function_evaluations
                ),
                "function_tolerance": self.optimizer.function_tolerance,
                "gradient_tolerance": self.optimizer.gradient_tolerance,
                "max_corrections": self.optimizer.max_corrections,
                "max_control_amplitude": self.optimizer.max_control_amplitude,
            },
            "propagation": self.propagation.metadata(),
            "time_limit_seconds": self.time_limit_seconds,
            "true_loss_safeguard": {
                "check_interval": self.exact_check_interval,
                "increase_patience": self.exact_increase_patience,
                "absolute_tolerance": self.exact_increase_absolute_tolerance,
                "relative_tolerance": self.exact_increase_relative_tolerance,
                "final_evaluation": self.final_exact_evaluation,
            },
            "trotter_splitting": {
                "initial_steps": self.splitting_steps,
                "adaptive": self.adaptive_splitting,
                "maximum_steps": self.max_splitting_steps,
                "error_criterion": "absolute-plus-relative",
                "relative_tolerance": self.splitting_relative_tolerance,
                "absolute_tolerance": self.splitting_absolute_tolerance,
            },
        }


@dataclass(frozen=True)
class OptimizationRun:
    """Optimization result paired with measured end-to-end runtime and config."""

    result: OptimizationResult
    elapsed_seconds: float
    config: OptimizationConfig


def optimize(
    problem: ControlProblem,
    pulse: Pulse,
    robustness: RobustnessModel | None,
    config: OptimizationConfig,
    *,
    control_groups: tuple[CommutingControlGroup, ...] = (),
    progress_callback: ProgressCallback | None = None,
    exact_progress_callback: ProgressCallback | None = None,
    splitting_progress_callback: SplittingProgressCallback | None = None,
) -> OptimizationRun:
    """Run a configured optimization on a user-supplied control problem."""

    start = time.perf_counter()
    if config.algorithm == OptimizationAlgorithm.GRAPE:
        result = GrapeSolver(
            config.optimizer,
            propagation=config.propagation,
        ).optimize(
            problem,
            pulse,
            robustness,
            time_limit=config.time_limit_seconds,
            progress_callback=progress_callback,
        )
    else:
        if not control_groups:
            raise ValueError("ST-GRAPE requires at least one commuting control group.")
        result = STGrapeSolver(
            config.optimizer,
            exact_check_interval=config.exact_check_interval,
            exact_increase_patience=config.exact_increase_patience,
            exact_increase_absolute_tolerance=(config.exact_increase_absolute_tolerance),
            exact_increase_relative_tolerance=(config.exact_increase_relative_tolerance),
            final_exact_evaluation=config.final_exact_evaluation,
            reference_propagation=config.propagation,
            splitting_steps=config.splitting_steps,
            adaptive_splitting=config.adaptive_splitting,
            max_splitting_steps=config.max_splitting_steps,
            splitting_relative_tolerance=config.splitting_relative_tolerance,
            splitting_absolute_tolerance=config.splitting_absolute_tolerance,
        ).optimize(
            problem,
            pulse,
            control_groups,
            robustness,
            time_limit=config.time_limit_seconds,
            progress_callback=progress_callback,
            exact_progress_callback=exact_progress_callback,
            splitting_progress_callback=splitting_progress_callback,
        )
    return OptimizationRun(
        result=result,
        elapsed_seconds=time.perf_counter() - start,
        config=config,
    )
