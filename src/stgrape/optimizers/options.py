"""Optimizer configuration shared by both algorithms."""

from __future__ import annotations

from dataclasses import dataclass

PRACTICALLY_UNLIMITED_FUNCTION_EVALUATIONS = 2_147_483_647


@dataclass(frozen=True)
class LBFGSBOptions:
    """Subset of SciPy L-BFGS-B options used in the paper experiments."""

    max_iterations: int = 100_000
    max_function_evaluations: int | None = None
    function_tolerance: float = 1e-14
    gradient_tolerance: float = 1e-14
    max_corrections: int = 20
    max_control_amplitude: float = 100.0 * 2.0 * 3.141592653589793

    def __post_init__(self) -> None:
        if self.max_function_evaluations is not None and (
            isinstance(self.max_function_evaluations, bool)
            or not isinstance(self.max_function_evaluations, int)
        ):
            raise TypeError("max_function_evaluations must be an integer or None.")
        if self.max_function_evaluations is not None and self.max_function_evaluations < 1:
            raise ValueError("max_function_evaluations must be positive or None.")

    @property
    def resolved_max_function_evaluations(self) -> int:
        """Return an explicit ``maxfun`` that cannot limit practical runs."""

        if self.max_function_evaluations is None:
            return PRACTICALLY_UNLIMITED_FUNCTION_EVALUATIONS
        return self.max_function_evaluations

    def scipy_options(self) -> dict[str, int | float | bool]:
        return {
            "maxiter": self.max_iterations,
            "maxfun": self.resolved_max_function_evaluations,
            "ftol": self.function_tolerance,
            "gtol": self.gradient_tolerance,
            "maxcor": self.max_corrections,
        }
