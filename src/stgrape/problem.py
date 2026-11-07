"""Scientific data models shared by the optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj
from scipy.special import exp1


class CorrelationModel(Protocol):
    """Pickle-safe reference correlation used for independent validation."""

    def covariance(self, lags: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the stationary covariance at non-negative time lags."""

        ...

    def metadata(self) -> dict[str, float | str]:
        """Return a JSON-compatible description of the physical model."""

        ...


@dataclass(frozen=True)
class SmoothOneOverFCorrelation:
    r"""Smoothly cut-off :math:`1/f` reference correlation.

    The angular-frequency cutoffs ``lower_cutoff`` and ``upper_cutoff`` use
    the reciprocal unit of the control problem's time variable.  The model is

    .. math::

        C(t)=2\left[E_1(\Lambda_l |t|)-E_1(\Lambda_h |t|)\right]
            =2\int_{\Lambda_l}^{\Lambda_h}\frac{e^{-\gamma |t|}}{\gamma}
              \,d\gamma.

    It is the continuous reference model approximated by the finite positive
    exponential expansion stored in :class:`ColoredNoise`.
    """

    lower_cutoff: float
    upper_cutoff: float

    def __post_init__(self) -> None:
        if self.lower_cutoff <= 0:
            raise ValueError("lower_cutoff must be positive.")
        if self.upper_cutoff <= self.lower_cutoff:
            raise ValueError("upper_cutoff must be greater than lower_cutoff.")

    @property
    def variance(self) -> float:
        """Return :math:`C(0)` for the unnormalized reference process."""

        return float(2.0 * np.log(self.upper_cutoff / self.lower_cutoff))

    def covariance(self, lags: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the exact covariance without a finite exponential fit."""

        values = np.asarray(lags, dtype=float)
        if np.any(values < 0):
            raise ValueError("lags must be non-negative.")
        covariance = np.empty_like(values)
        at_origin = values == 0.0
        covariance[at_origin] = self.variance
        positive = ~at_origin
        covariance[positive] = 2.0 * (
            exp1(self.lower_cutoff * values[positive]) - exp1(self.upper_cutoff * values[positive])
        )
        return covariance

    def metadata(self) -> dict[str, float | str]:
        """Return the parameters needed to reproduce validation samples."""

        return {
            "model": "smooth-one-over-f",
            "lower_angular_cutoff_per_time": self.lower_cutoff,
            "upper_angular_cutoff_per_time": self.upper_cutoff,
            "unnormalized_variance": self.variance,
        }


class ControlTask(StrEnum):
    """Supported terminal-control objectives."""

    STATE_PREPARATION = "state"
    GATE_SYNTHESIS = "gate"


class RobustPenaltyMode(StrEnum):
    """Supported terminal sensitivity-penalty conventions."""

    SQUARED = "squared"
    RMS = "rms"
    HYBRID = "hybrid"


def _as_qobj_tuple(values: tuple[Qobj, ...] | list[Qobj]) -> tuple[Qobj, ...]:
    return tuple(values)


@dataclass(frozen=True)
class ControlProblem:
    """Nominal piecewise-constant quantum-control problem.

    Hamiltonians are angular frequencies.  The product of a Hamiltonian and
    ``time_step`` must therefore be dimensionless.  The paper experiments use
    rad/µs for Hamiltonians and µs for time.

    Parameters
    ----------
    drift_hamiltonian
        Time-independent drift Hamiltonian :math:`H_0`.
    control_hamiltonians
        Hamiltonians :math:`H_n` driven by the real control amplitudes.
    initial, target
        State vectors for state preparation or operators for gate synthesis.
    duration, time_step
        Total evolution time and piecewise-constant time step.
    task
        State-preparation or gate-synthesis objective.
    """

    drift_hamiltonian: Qobj
    control_hamiltonians: tuple[Qobj, ...]
    initial: Qobj
    target: Qobj
    duration: float
    time_step: float
    task: ControlTask

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_hamiltonians", _as_qobj_tuple(self.control_hamiltonians))
        object.__setattr__(self, "task", ControlTask(self.task))

        if not self.control_hamiltonians:
            raise ValueError("At least one control Hamiltonian is required.")
        if self.duration <= 0 or self.time_step <= 0:
            raise ValueError("duration and time_step must be positive.")

        step_count = self.duration / self.time_step
        if not np.isclose(step_count, round(step_count), rtol=0, atol=1e-10):
            raise ValueError("duration must be an integer multiple of time_step.")

        shape = self.drift_hamiltonian.shape
        if shape[0] != shape[1]:
            raise ValueError("drift_hamiltonian must be square.")
        for index, hamiltonian in enumerate(self.control_hamiltonians):
            if hamiltonian.shape != shape:
                raise ValueError(
                    f"control_hamiltonians[{index}] has shape {hamiltonian.shape}; "
                    f"expected {shape}."
                )

        expected_initial_shape = (
            (shape[0], 1) if self.task == ControlTask.STATE_PREPARATION else shape
        )
        if self.initial.shape != expected_initial_shape:
            raise ValueError(
                f"initial has shape {self.initial.shape}; expected {expected_initial_shape}."
            )
        if self.target.shape != expected_initial_shape:
            raise ValueError(
                f"target has shape {self.target.shape}; expected {expected_initial_shape}."
            )

    @property
    def n_controls(self) -> int:
        """Number of independent control channels."""

        return len(self.control_hamiltonians)

    @property
    def n_time_steps(self) -> int:
        """Number of piecewise-constant control intervals."""

        return int(round(self.duration / self.time_step))

    @property
    def hilbert_dimension(self) -> int:
        """Dimension of the physical Hilbert space."""

        return self.drift_hamiltonian.shape[0]


@dataclass(frozen=True)
class ControlDrift:
    """Relative amplitude uncertainty on one control channel.

    ``scale`` is the standard deviation of the dimensionless relative error.
    A value of ``0.01`` represents a 1% amplitude drift.
    """

    channel: int
    scale: float

    def __post_init__(self) -> None:
        if self.channel < 0:
            raise ValueError("channel must be non-negative.")
        if self.scale <= 0:
            raise ValueError("scale must be positive.")


@dataclass(frozen=True)
class TaylorNoise:
    """Uncertainties represented by a truncated multivariate Taylor series.

    ``static_source_weights`` optionally assigns a distinct first-order weight
    to every static source. When it is empty, ``static_source_weight`` is used
    uniformly. ``control_drift_source_weight`` remains uniform within the
    control-drift family. A multi-index block receives the direct product of
    its source weights.
    ``penalty_weight`` is the default multiplier applied after that product.
    When ``degree_penalty_weights`` is provided, entry ``k - 1`` replaces the
    default multiplier for Taylor blocks of total degree ``k``.
    """

    static_operators: tuple[Qobj, ...] = field(default_factory=tuple)
    control_drifts: tuple[ControlDrift, ...] = field(default_factory=tuple)
    order: int = 2
    penalty_weight: float = 1.0
    degree_penalty_weights: tuple[float, ...] = field(default_factory=tuple)
    static_source_weight: float = 1.0
    static_source_weights: tuple[float, ...] = field(default_factory=tuple)
    control_drift_source_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "static_operators", _as_qobj_tuple(self.static_operators))
        object.__setattr__(self, "control_drifts", tuple(self.control_drifts))
        object.__setattr__(
            self,
            "static_source_weights",
            tuple(float(weight) for weight in self.static_source_weights),
        )
        object.__setattr__(
            self,
            "degree_penalty_weights",
            tuple(float(weight) for weight in self.degree_penalty_weights),
        )
        if self.order < 1:
            raise ValueError("Taylor robustness order must be at least one.")
        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")
        if self.degree_penalty_weights and len(self.degree_penalty_weights) != self.order:
            raise ValueError("degree_penalty_weights must contain one value per order.")
        if any(weight < 0 for weight in self.degree_penalty_weights):
            raise ValueError("degree_penalty_weights must be non-negative.")
        if self.static_source_weight < 0:
            raise ValueError("static_source_weight must be non-negative.")
        if self.static_source_weights and len(self.static_source_weights) != len(
            self.static_operators
        ):
            raise ValueError("static_source_weights must contain one value per static operator.")
        if any(weight < 0 for weight in self.static_source_weights):
            raise ValueError("static_source_weights must be non-negative.")
        if self.control_drift_source_weight < 0:
            raise ValueError("control_drift_source_weight must be non-negative.")
        if not self.static_operators and not self.control_drifts:
            raise ValueError("TaylorNoise requires at least one uncertainty source.")
        channels = [drift.channel for drift in self.control_drifts]
        if len(channels) != len(set(channels)):
            raise ValueError("A control channel may appear in control_drifts only once.")

    @property
    def n_sources(self) -> int:
        """Total number of independent Taylor uncertainty parameters."""

        return len(self.static_operators) + len(self.control_drifts)

    @property
    def source_weights(self) -> tuple[float, ...]:
        """Return first-order weights in augmented-source ordering."""

        return (
            *(
                self.static_source_weights
                if self.static_source_weights
                else tuple(self.static_source_weight for _ in self.static_operators)
            ),
            *(self.control_drift_source_weight for _ in self.control_drifts),
        )

    def penalty_weight_for_degree(self, degree: int) -> float:
        """Return the configured multiplier for one Taylor total degree."""

        if degree < 1 or degree > self.order:
            raise ValueError("degree must lie between one and the Taylor order.")
        if self.degree_penalty_weights:
            return self.degree_penalty_weights[degree - 1]
        return self.penalty_weight


@dataclass(frozen=True)
class ColoredNoise:
    r"""Stationary colored noise with an exponential correlation expansion.

    Following Eq. (58) of the paper, the stochastic Hamiltonian term is

    .. math:: \epsilon\,\beta(t)E,

    where ``operator`` is :math:`E`, ``strength`` is the independent scale
    :math:`\epsilon`, and :math:`\beta(t)` is a dimensionless stationary
    process.  Its optimization correlation function is

    .. math:: C(t) = \sum_k a_k \exp(-\gamma_k t).

    ``correlation_weights`` are :math:`a_k` and ``decay_rates`` are the
    positive rates :math:`\gamma_k`.  ``reference_correlation`` optionally
    records the unfitted physical covariance used for independent Monte Carlo
    validation; it does not alter the augmented optimization model.
    """

    operator: Qobj
    correlation_weights: tuple[float, ...]
    decay_rates: tuple[float, ...]
    strength: float = 1.0
    penalty_weight: float = 1.0
    reference_correlation: CorrelationModel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "correlation_weights", tuple(self.correlation_weights))
        object.__setattr__(self, "decay_rates", tuple(self.decay_rates))
        if not self.correlation_weights:
            raise ValueError("At least one exponential correlation term is required.")
        if len(self.correlation_weights) != len(self.decay_rates):
            raise ValueError("correlation_weights and decay_rates must have equal length.")
        if any(weight <= 0 for weight in self.correlation_weights):
            raise ValueError("correlation_weights must be positive.")
        if any(rate <= 0 for rate in self.decay_rates):
            raise ValueError("decay_rates must be positive.")
        if self.strength < 0:
            raise ValueError("strength must be non-negative.")
        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")

    @property
    def scaled_operator(self) -> Qobj:
        r"""Return :math:`\epsilon E` used in optimization and simulation."""

        return self.strength * self.operator

    @property
    def optimization_variance(self) -> float:
        r"""Return :math:`C_K(0)=\sum_k a_k` for the finite surrogate."""

        return float(sum(self.correlation_weights))


@dataclass(frozen=True)
class RobustnessModel:
    """Collection of independent robustness penalties.

    Taylor uncertainties are propagated in one multivariate augmented system.
    Colored-noise sources share one multi-source Van Loan system. RMS and
    hybrid modes combine their source-specific mean shifts in one joint RMS;
    the nominal fidelity is shared between robustness components and counted
    once.
    """

    taylor: TaylorNoise | None = None
    colored: tuple[ColoredNoise, ...] = field(default_factory=tuple)
    penalty_mode: RobustPenaltyMode = RobustPenaltyMode.SQUARED
    rms_smoothing: float = 1e-12
    p1_smoothing: float | None = None
    p2_smoothing: float | None = None
    pc_smoothing: float | None = None
    p1_coefficient: float = 1.0
    p2_coefficient: float = 1.0
    pc_coefficient: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "colored", tuple(self.colored))
        object.__setattr__(self, "penalty_mode", RobustPenaltyMode(self.penalty_mode))
        values = {"rms_smoothing": self.rms_smoothing}
        for name in ("p1_smoothing", "p2_smoothing", "pc_smoothing"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        for name in ("p1_coefficient", "p2_coefficient", "pc_coefficient"):
            values[name] = getattr(self, name)
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("Robust penalty scales must be finite and non-negative.")
        if self.rms_smoothing <= 0.0:
            raise ValueError("rms_smoothing must be positive.")
        if any(
            value is not None and value <= 0.0
            for value in (
                self.p1_smoothing,
                self.p2_smoothing,
                self.pc_smoothing,
            )
        ):
            raise ValueError("Optional RMS smoothing scales must be positive.")

    @property
    def resolved_p1_coefficient(self) -> float:
        """Return the direct P1 coefficient."""

        return self.p1_coefficient

    @property
    def resolved_p2_coefficient(self) -> float:
        """Return the direct P2 coefficient."""

        return self.p2_coefficient

    @property
    def resolved_p1_smoothing(self) -> float:
        """Return the dedicated P1 smoothing scale or the shared fallback."""

        return self.rms_smoothing if self.p1_smoothing is None else self.p1_smoothing

    @property
    def resolved_pc_coefficient(self) -> float:
        """Return the direct colored mean-shift coefficient."""

        return self.pc_coefficient

    @property
    def resolved_p2_smoothing(self) -> float:
        """Return the dedicated P2 smoothing scale or the shared fallback."""

        return self.rms_smoothing if self.p2_smoothing is None else self.p2_smoothing

    @property
    def resolved_pc_smoothing(self) -> float:
        """Return the dedicated colored smoothing scale or the shared fallback."""

        return self.rms_smoothing if self.pc_smoothing is None else self.pc_smoothing

    @property
    def n_components(self) -> int:
        """Number of independently propagated augmented systems."""

        return int(self.taylor is not None) + len(self.colored)
