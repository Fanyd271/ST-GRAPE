"""Terminal fidelities and robustness penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj

from .problem import ControlTask

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SmoothRmsPenalty:
    r"""One smooth RMS norm of terminal-block linear combinations.

    Every entry of ``features`` describes one matrix/vector

    .. math:: Y_j=\sum_b c_{jb}X_b.

    The resulting contribution is

    .. math::

        \lambda\left[\sqrt{d_\chi^{-1}\sum_j\|Y_j\|_F^2+s^2}-s\right],

    where ``coefficient`` is :math:`\lambda` and ``smoothing`` is :math:`s`.
    Real feature coefficients are sufficient for the Gaussian RMS moments used
    by the Taylor and colored-noise objectives.
    """

    features: tuple[tuple[tuple[int, float], ...], ...]
    coefficient: float
    smoothing: float

    def __post_init__(self) -> None:
        if not self.features or any(not feature for feature in self.features):
            raise ValueError("Smooth RMS penalties require non-empty features.")
        if not np.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("RMS penalty coefficient must be finite and non-negative.")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("RMS smoothing must be finite and positive.")
        for feature in self.features:
            for block, weight in feature:
                if block < 0:
                    raise ValueError("RMS penalty block indices must be non-negative.")
                if not np.isfinite(weight):
                    raise ValueError("RMS feature coefficients must be finite.")


@dataclass(frozen=True)
class ObjectiveValue:
    """Decomposition of one augmented component's terminal loss."""

    loss: float
    fidelity: float
    penalty: float


def terminal_objective(
    state: ComplexArray,
    target: Qobj,
    task: ControlTask,
    *,
    nominal_block: int,
    penalty_blocks: tuple[int, ...],
    fidelity_weight: float,
    penalty_weights: tuple[float, ...],
    rms_penalties: tuple[SmoothRmsPenalty, ...] = (),
) -> tuple[ObjectiveValue, ComplexArray]:
    """Evaluate one augmented objective and its terminal costate.

    The returned costate is the real Hilbert--Schmidt gradient of ``loss``:
    ``d loss = Re(vdot(costate, d state))``.  Gate penalties use the
    dimension-normalized Frobenius norm employed by the original simulations.
    """

    state = np.asarray(state, dtype=np.complex128)
    if state.ndim != 3:
        raise ValueError("state must have shape (n_blocks, d, state_columns).")
    if not 0 <= nominal_block < state.shape[0]:
        raise ValueError("nominal_block lies outside the augmented state.")
    if any(index < 0 or index >= state.shape[0] for index in penalty_blocks):
        raise ValueError("A penalty block lies outside the augmented state.")
    if len(penalty_weights) != len(penalty_blocks):
        raise ValueError("penalty_weights must contain one value per penalty block.")
    if any(not np.isfinite(weight) or weight < 0.0 for weight in penalty_weights):
        raise ValueError("penalty_weights must be finite and non-negative.")
    for rms_penalty in rms_penalties:
        if any(
            block < 0 or block >= state.shape[0]
            for feature in rms_penalty.features
            for block, _ in feature
        ):
            raise ValueError("An RMS penalty block lies outside the augmented state.")

    target_array = target.full()
    nominal = state[nominal_block]
    costate = np.zeros_like(state)
    dimension = nominal.shape[0]

    if task == ControlTask.STATE_PREPARATION:
        overlap = (target_array.conj().T @ nominal).item()
        fidelity = float(abs(overlap) ** 2)
        costate[nominal_block] = -2.0 * fidelity_weight * overlap * target_array
        normalization = 1.0
    elif task == ControlTask.GATE_SYNTHESIS:
        overlap = np.trace(target_array.conj().T @ nominal)
        fidelity = float((abs(overlap) ** 2 + dimension) / (dimension * (dimension + 1)))
        costate[nominal_block] = (
            -2.0 * fidelity_weight * overlap * target_array / (dimension * (dimension + 1))
        )
        normalization = 1.0 / dimension
    else:  # pragma: no cover - ControlTask prevents this branch.
        raise ValueError(f"Unsupported control task: {task}")

    weighted_penalty_norm = sum(
        weight * float(np.vdot(state[index], state[index]).real)
        for index, weight in zip(penalty_blocks, penalty_weights, strict=True)
    )
    penalty = 0.5 * normalization * weighted_penalty_norm
    for index, weight in zip(penalty_blocks, penalty_weights, strict=True):
        costate[index] += weight * normalization * state[index]

    for rms_penalty in rms_penalties:
        features = tuple(
            sum(
                (weight * state[index] for index, weight in feature),
                start=np.zeros_like(state[0]),
            )
            for feature in rms_penalty.features
        )
        squared_rms = normalization * sum(
            float(np.vdot(feature, feature).real) for feature in features
        )
        denominator = float(np.sqrt(squared_rms + rms_penalty.smoothing * rms_penalty.smoothing))
        penalty += rms_penalty.coefficient * (denominator - rms_penalty.smoothing)
        pullback_scale = rms_penalty.coefficient * normalization / denominator
        for feature_definition, feature_state in zip(
            rms_penalty.features,
            features,
            strict=True,
        ):
            for index, weight in feature_definition:
                costate[index] += pullback_scale * weight * feature_state

    loss = fidelity_weight * (1.0 - fidelity) + penalty
    return ObjectiveValue(loss=loss, fidelity=fidelity, penalty=penalty), costate
