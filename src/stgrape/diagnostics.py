"""Exact terminal sensitivity diagnostics for saved robust-control pulses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stgrape.augmented.builders import taylor_gaussian_rms_features
from stgrape.augmented.indices import generate_multi_indices
from stgrape.problem import ControlProblem, RobustnessModel
from stgrape.propagation.ode import (
    OdePropagationOptions,
    OdeRepresentation,
    build_ode_components,
)

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class TaylorSensitivityDiagnostics:
    """Physical Gaussian RMS norms extracted from an exact augmented endpoint."""

    p1: float
    p1_squared: float
    hybrid_p1_base: float
    p2: float
    evaluated_order: int
    rhs_evaluations: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe, explicitly normalized diagnostic record."""

        return {
            "p1": self.p1,
            "p1_squared": self.p1_squared,
            "hybrid_p1_base": self.hybrid_p1_base,
            "p2": self.p2,
            "evaluated_order": self.evaluated_order,
            "rhs_evaluations": self.rhs_evaluations,
            "reference": "exact-ode-block-sparse",
        }


def _feature_rms(
    state: ComplexArray,
    features: tuple[tuple[tuple[int, float], ...], ...],
    *,
    normalization: float,
) -> float:
    """Return the normalized RMS norm of terminal block combinations."""

    squared = 0.0
    for feature in features:
        combined = sum(
            (weight * state[block] for block, weight in feature),
            start=np.zeros_like(state[0]),
        )
        squared += float(np.vdot(combined, combined).real)
    return float(np.sqrt(max(0.0, normalization * squared)))


def taylor_sensitivity_diagnostics_from_state(
    state: ComplexArray,
    problem: ControlProblem,
    indices: tuple[tuple[int, ...], ...],
    *,
    rhs_evaluations: int = 0,
) -> TaylorSensitivityDiagnostics:
    """Compute P1 and Gaussian fourth-moment P2 from one augmented endpoint."""

    terminal_state = np.asarray(state, dtype=np.complex128)
    if terminal_state.ndim != 3 or terminal_state.shape[0] != len(indices):
        raise ValueError("Taylor terminal state and multi-indices are incompatible.")
    normalization = 1.0 if problem.task.value == "state" else 1.0 / problem.hilbert_dimension
    p1 = _feature_rms(
        terminal_state,
        taylor_gaussian_rms_features(indices, degree=1),
        normalization=normalization,
    )
    p2 = _feature_rms(
        terminal_state,
        taylor_gaussian_rms_features(indices, degree=2),
        normalization=normalization,
    )
    p1_squared = p1 * p1
    return TaylorSensitivityDiagnostics(
        p1=p1,
        p1_squared=p1_squared,
        hybrid_p1_base=p1_squared,
        p2=p2,
        evaluated_order=2,
        rhs_evaluations=rhs_evaluations,
    )


def evaluate_taylor_sensitivity_diagnostics(
    problem: ControlProblem,
    robustness: RobustnessModel,
    control_amplitudes: RealArray,
    *,
    propagation: OdePropagationOptions | None = None,
) -> TaylorSensitivityDiagnostics | None:
    """Evaluate P1/P2 with a common exact, order-two block-sparse ODE reference.

    An order-one optimization is deliberately lifted to an order-two diagnostic
    model so that its P2 value remains comparable with an order-two result.
    Objective weights do not affect the augmented forward dynamics.
    """

    if robustness.taylor is None:
        return None
    diagnostic_taylor = replace(
        robustness.taylor,
        order=2,
        degree_penalty_weights=(),
    )
    diagnostic_robustness = replace(
        robustness,
        taylor=diagnostic_taylor,
        colored=(),
    )
    reference = propagation or OdePropagationOptions()
    reference = OdePropagationOptions(
        solver=reference.solver,
        representation=OdeRepresentation.BLOCK_SPARSE,
    )
    component = build_ode_components(
        problem,
        diagnostic_robustness,
        reference,
    )[0]
    evaluation = component.evaluate(
        problem,
        np.asarray(control_amplitudes, dtype=float),
        fidelity_weight=0.0,
        compute_gradient=False,
    )
    indices = generate_multi_indices(diagnostic_taylor.n_sources, order=2)
    return taylor_sensitivity_diagnostics_from_state(
        evaluation.final_state,
        problem,
        indices,
        rhs_evaluations=evaluation.rhs_evaluations,
    )


def sensitivity_pair_metadata(
    initial: TaylorSensitivityDiagnostics | None,
    final: TaylorSensitivityDiagnostics | None,
) -> dict[str, Any]:
    """Return the nested metadata representation used by saved paper methods."""

    return {
        "initial": None if initial is None else initial.as_dict(),
        "final": None if final is None else final.as_dict(),
    }
