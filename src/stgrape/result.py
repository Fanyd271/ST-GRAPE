"""Optimization results and stable on-disk serialization."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import qutip
import scipy
from numpy.typing import NDArray

from .paths import ensure_output_path


@dataclass(frozen=True)
class OptimizationResult:
    """Output shared by GRAPE and ST-GRAPE solvers."""

    method: str
    coefficients: NDArray[np.float64]
    control_amplitudes: NDArray[np.float64]
    objective: float
    initial_objective: float
    objective_history: NDArray[np.float64]
    elapsed_time_history: NDArray[np.float64]
    initial_fidelity_loss: float | None = None
    initial_robustness_penalty: float | None = None
    fidelity_loss_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    robustness_penalty_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_objective_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_evaluation_times: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_evaluation_iterations: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    exact_approximate_objective_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_approximation_gap_ratio_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_approximation_absolute_error_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_splitting_tolerance_history: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exact_splitting_tolerance_exceeded_history: NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    trotter_splitting_steps_history: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    splitting_change_iterations: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    splitting_change_steps: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    exact_objective: float | None = None
    exact_objective_iteration: int | None = None
    exact_deterioration_count: int = 0
    exact_safeguard_termination_iteration: int | None = None
    propagation: dict[str, object] = field(default_factory=dict)
    message: str = ""
    success: bool = False

    def save(self, directory: str | Path, *, metadata: dict[str, object] | None = None) -> Path:
        """Save arrays and reproducibility metadata in a dedicated run directory."""

        path = ensure_output_path(Path(directory))
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            path / "result.npz",
            coefficients=self.coefficients,
            control_amplitudes=self.control_amplitudes,
            objective=self.objective,
            initial_objective=self.initial_objective,
            objective_history=self.objective_history,
            elapsed_time_history=self.elapsed_time_history,
            initial_fidelity_loss=(
                np.nan if self.initial_fidelity_loss is None else self.initial_fidelity_loss
            ),
            initial_nominal_fidelity=(
                np.nan if self.initial_fidelity_loss is None else 1.0 - self.initial_fidelity_loss
            ),
            initial_robustness_penalty=(
                np.nan
                if self.initial_robustness_penalty is None
                else self.initial_robustness_penalty
            ),
            fidelity_loss_history=self.fidelity_loss_history,
            nominal_fidelity_history=1.0 - self.fidelity_loss_history,
            robustness_penalty_history=self.robustness_penalty_history,
            exact_objective_history=self.exact_objective_history,
            exact_evaluation_times=self.exact_evaluation_times,
            exact_evaluation_iterations=self.exact_evaluation_iterations,
            exact_approximate_objective_history=(self.exact_approximate_objective_history),
            exact_approximation_gap_ratio_history=(self.exact_approximation_gap_ratio_history),
            exact_approximation_absolute_error_history=(
                self.exact_approximation_absolute_error_history
            ),
            exact_splitting_tolerance_history=self.exact_splitting_tolerance_history,
            exact_splitting_tolerance_exceeded_history=(
                self.exact_splitting_tolerance_exceeded_history
            ),
            trotter_splitting_steps_history=self.trotter_splitting_steps_history,
            splitting_change_iterations=self.splitting_change_iterations,
            splitting_change_steps=self.splitting_change_steps,
            exact_objective=(np.nan if self.exact_objective is None else self.exact_objective),
            exact_objective_iteration=(
                -1 if self.exact_objective_iteration is None else self.exact_objective_iteration
            ),
            exact_deterioration_count=self.exact_deterioration_count,
            exact_safeguard_termination_iteration=(
                -1
                if self.exact_safeguard_termination_iteration is None
                else self.exact_safeguard_termination_iteration
            ),
        )
        information: dict[str, object] = {
            "schema_version": 1,
            "method": self.method,
            "message": self.message,
            "success": self.success,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qutip": qutip.__version__,
        }
        if self.propagation:
            information["propagation"] = self.propagation
        information.update(metadata or {})
        (path / "metadata.json").write_text(
            json.dumps(information, indent=2, sort_keys=True), encoding="utf-8"
        )
        return path
