"""Differentiable parameterizations of applied PWC control fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin
from scipy.signal.windows import blackman

from .problem import ControlProblem

RealArray = NDArray[np.float64]
ControlBasis = Literal["pwc", "chebyshev"]
ChebyshevNormalization = Literal["rms", "none"]
DEFAULT_CHEBYSHEV_BASIS_SIZE = 8
DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT = 100.0 * 2.0 * np.pi


@runtime_checkable
class Pulse(Protocol):
    """Interface accepted by all optimizers.

    A pulse parameterization only needs to map real optimization parameters to
    PWC control amplitudes and pull amplitude gradients back to parameter
    space.  Custom Fourier, spline, or hardware-aware parameterizations can
    implement this protocol without changing either optimizer.
    """

    n_controls: int
    n_time_steps: int
    time_step: float
    coefficients: RealArray

    @property
    def n_parameters_per_control(self) -> int: ...

    @property
    def n_parameters(self) -> int: ...

    def amplitudes(self, coefficients: RealArray | None = None) -> RealArray: ...

    def pullback(
        self,
        amplitude_gradient: RealArray,
        coefficients: RealArray | None = None,
    ) -> RealArray: ...

    def flattened_coefficients(self) -> RealArray: ...

    def parameter_bounds(
        self,
        max_control_amplitude: float,
    ) -> list[tuple[float, float]] | None: ...

    def metadata(self) -> dict[str, object]: ...


class PulseEnvelope(Protocol):
    """Generate one dimensionless envelope on a PWC time grid."""

    def values(self, n_time_steps: int, time_step: float) -> RealArray: ...


@dataclass(frozen=True)
class FlatEnvelope:
    """Unit envelope for controls that need no edge shaping."""

    def values(self, n_time_steps: int, time_step: float) -> RealArray:
        if n_time_steps < 1 or time_step <= 0:
            raise ValueError("n_time_steps and time_step must be positive.")
        return np.ones(n_time_steps, dtype=float)


@dataclass(frozen=True)
class BlackmanEnvelope:
    """Blackman turn-on/turn-off envelope used by the paper experiments."""

    rise_time: float = 0.01

    def values(self, n_time_steps: int, time_step: float) -> RealArray:
        if self.rise_time <= 0:
            raise ValueError("rise_time must be positive.")
        n_rise = int(round(self.rise_time / time_step))
        if n_rise < 1 or 2 * n_rise > n_time_steps:
            raise ValueError(
                "rise_time must cover at least one step and no more than half the pulse."
            )
        window = blackman(2 * n_rise)
        envelope = np.ones(n_time_steps, dtype=float)
        envelope[:n_rise] = window[:n_rise]
        envelope[-n_rise:] = window[n_rise:]
        return envelope


@dataclass(frozen=True)
class PulseConfig:
    """User-facing configuration for the paper's shaped control fields.

    ``envelope`` and ``filter_kernel`` are replaceable.  Omitting the kernel
    constructs the original low-pass FIR filter from ``cutoff_frequency`` and
    ``filter_taps``.  Initial Fourier coefficients are reproducible from
    ``seed`` and remain separate from optimizer parameter bounds.
    """

    envelope: PulseEnvelope = field(default_factory=BlackmanEnvelope)
    cutoff_frequency: float = 60.0
    filter_taps: int = 10
    filter_kernel: tuple[float, ...] | None = None
    frequency_range: tuple[float, float] = (1.0, 30.0)
    initial_max_amplitude: float = 30.0 * 2.0 * np.pi
    n_waves: int = 10
    chebyshev_normalization: ChebyshevNormalization = "rms"
    chebyshev_amplitude_limit: float = DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT

    def __post_init__(self) -> None:
        if self.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be positive.")
        if self.filter_taps < 2:
            raise ValueError("filter_taps must be at least two.")
        if self.frequency_range[0] <= 0 or self.frequency_range[0] >= self.frequency_range[1]:
            raise ValueError("frequency_range must contain two increasing positive values.")
        if self.initial_max_amplitude <= 0 or self.n_waves < 1:
            raise ValueError("initial_max_amplitude and n_waves must be positive.")
        if self.chebyshev_normalization not in {"rms", "none"}:
            raise ValueError("chebyshev_normalization must be 'rms' or 'none'.")
        if self.chebyshev_amplitude_limit <= 0:
            raise ValueError("chebyshev_amplitude_limit must be positive.")
        if self.filter_kernel is not None:
            if len(self.filter_kernel) < 2:
                raise ValueError("filter_kernel must contain at least two taps.")
            if not np.all(np.isfinite(self.filter_kernel)):
                raise ValueError("filter_kernel must contain only finite values.")

    def random_pulse(
        self,
        problem: ControlProblem,
        *,
        seed: int,
        control_basis: ControlBasis = "pwc",
        chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
    ) -> Pulse:
        """Create a deterministic pulse matching a control problem's grid."""

        rng = np.random.default_rng(seed)
        if control_basis == "pwc":
            return FilteredPulse.random(
                n_controls=problem.n_controls,
                n_time_steps=problem.n_time_steps,
                time_step=problem.time_step,
                rng=rng,
                frequency_range=self.frequency_range,
                max_amplitude=self.initial_max_amplitude,
                n_waves=self.n_waves,
                envelope=self.envelope,
                cutoff_frequency=self.cutoff_frequency,
                filter_taps=self.filter_taps,
                filter_kernel=self.filter_kernel,
            )
        if control_basis == "chebyshev":
            return ChebyshevPulse.random(
                n_controls=problem.n_controls,
                n_time_steps=problem.n_time_steps,
                time_step=problem.time_step,
                n_basis=chebyshev_k,
                rng=rng,
                frequency_range=self.frequency_range,
                initial_max_amplitude=self.initial_max_amplitude,
                n_waves=self.n_waves,
                normalization=self.chebyshev_normalization,
                amplitude_limit=self.chebyshev_amplitude_limit,
                envelope=self.envelope,
                cutoff_frequency=self.cutoff_frequency,
                filter_taps=self.filter_taps,
                filter_kernel=self.filter_kernel,
            )
        raise ValueError("control_basis must be 'pwc' or 'chebyshev'.")

    def pulse_from_coefficients(
        self,
        problem: ControlProblem,
        coefficients: RealArray,
        *,
        control_basis: ControlBasis = "pwc",
        chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
    ) -> Pulse:
        """Reconstruct a configured pulse from saved optimization parameters."""

        common = {
            "n_controls": problem.n_controls,
            "n_time_steps": problem.n_time_steps,
            "time_step": problem.time_step,
            "coefficients": coefficients,
            "envelope": self.envelope,
            "cutoff_frequency": self.cutoff_frequency,
            "filter_taps": self.filter_taps,
            "filter_kernel": self.filter_kernel,
        }
        if control_basis == "pwc":
            return FilteredPulse(**common)
        if control_basis == "chebyshev":
            return ChebyshevPulse(
                **common,
                n_basis=chebyshev_k,
                normalization=self.chebyshev_normalization,
                amplitude_limit=self.chebyshev_amplitude_limit,
            )
        raise ValueError("control_basis must be 'pwc' or 'chebyshev'.")


@dataclass
class FilteredPulse:
    """FIR-filtered controls with a smooth turn-on/turn-off envelope.

    Optimization parameters are stored as an array with shape
    ``(n_controls, n_parameters_per_control)``.  The public control amplitudes
    always have shape ``(n_controls, n_time_steps)``.
    """

    n_controls: int
    n_time_steps: int
    time_step: float
    coefficients: np.ndarray
    rise_time: float = 0.01
    envelope: PulseEnvelope | None = None
    cutoff_frequency: float = 60.0
    filter_taps: int = 10
    filter_kernel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_controls < 1 or self.n_time_steps < 2:
            raise ValueError("n_controls must be positive and n_time_steps must exceed one.")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive.")
        if self.envelope is None:
            self.envelope = BlackmanEnvelope(self.rise_time)
        if self.filter_kernel is not None:
            self.filter_kernel = tuple(float(value) for value in self.filter_kernel)
            self.filter_taps = len(self.filter_kernel)
        if not 1 < self.filter_taps <= self.n_time_steps:
            raise ValueError("filter_taps must lie in [2, n_time_steps].")

        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = (self.n_controls, self.n_parameters_per_control)
        if coefficients.shape == (self.n_controls * self.n_parameters_per_control,):
            coefficients = coefficients.reshape(expected)
        if coefficients.shape != expected:
            raise ValueError(f"coefficients has shape {coefficients.shape}; expected {expected}.")
        self.coefficients = coefficients.copy()

    @property
    def n_parameters_per_control(self) -> int:
        """Number of unconvolved parameters in each control channel."""

        return self.n_time_steps + 1 - self.filter_taps

    @property
    def n_parameters(self) -> int:
        """Total number of optimization parameters."""

        return self.n_controls * self.n_parameters_per_control

    @cached_property
    def parameterization_matrix(self) -> np.ndarray:
        """Linear map from one channel's coefficients to its amplitudes."""

        if self.envelope is None:  # pragma: no cover - normalized in __post_init__.
            raise RuntimeError("Pulse envelope was not initialized.")
        envelope = np.asarray(
            self.envelope.values(self.n_time_steps, self.time_step),
            dtype=float,
        )
        if envelope.shape != (self.n_time_steps,) or not np.all(np.isfinite(envelope)):
            raise ValueError("envelope must return one finite value per time step.")

        total_time = self.n_time_steps * self.time_step
        normalized_cutoff = self.cutoff_frequency * total_time / (0.5 * self.n_time_steps)
        if not 0 < normalized_cutoff < 1:
            raise ValueError(
                "cutoff_frequency is outside the range supported by the chosen time grid."
            )
        kernel = (
            firwin(self.filter_taps, normalized_cutoff)
            if self.filter_kernel is None
            else np.asarray(self.filter_kernel, dtype=float)
        )

        convolution = np.zeros((self.n_time_steps, self.n_parameters_per_control))
        for column in range(self.n_parameters_per_control):
            convolution[column : column + self.filter_taps, column] = kernel
        return envelope[:, None] * convolution

    def amplitudes(self, coefficients: np.ndarray | None = None) -> np.ndarray:
        """Map pulse coefficients to control amplitudes.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_controls, n_time_steps)``.
        """

        values = (
            self.coefficients if coefficients is None else np.asarray(coefficients, dtype=float)
        )
        values = values.reshape(self.n_controls, self.n_parameters_per_control)
        return values @ self.parameterization_matrix.T

    def pullback(
        self,
        amplitude_gradient: np.ndarray,
        coefficients: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the transpose pulse map to an amplitude-space gradient."""

        gradient = np.asarray(amplitude_gradient, dtype=float)
        expected = (self.n_controls, self.n_time_steps)
        if gradient.shape != expected:
            raise ValueError(f"amplitude_gradient has shape {gradient.shape}; expected {expected}.")
        return gradient @ self.parameterization_matrix

    def flattened_coefficients(self) -> np.ndarray:
        """Return a copy suitable for SciPy optimizers."""

        return self.coefficients.reshape(-1).copy()

    def parameter_bounds(
        self,
        max_control_amplitude: float,
    ) -> list[tuple[float, float]]:
        """Retain the original box bounds for the filtered-PWC parameters."""

        return [(-max_control_amplitude, max_control_amplitude)] * self.n_parameters

    def metadata(self) -> dict[str, object]:
        """Describe the complete coefficient-to-applied-control map."""

        return {
            "control_basis": "pwc",
            "parameter_count_per_control": self.n_parameters_per_control,
            "control_chain": ["PWC", "LTI", "envelope"],
            "envelope": type(self.envelope).__name__,
            "rise_time_us": self.rise_time,
            "cutoff_frequency_mhz": self.cutoff_frequency,
            "filter_taps": self.filter_taps,
            "filter_kernel": (None if self.filter_kernel is None else list(self.filter_kernel)),
        }

    @classmethod
    def random(
        cls,
        *,
        n_controls: int,
        n_time_steps: int,
        time_step: float,
        rng: np.random.Generator,
        frequency_range: tuple[float, float] = (1.0, 30.0),
        max_amplitude: float = 30.0 * 2.0 * np.pi,
        n_waves: int = 10,
        rise_time: float = 0.01,
        envelope: PulseEnvelope | None = None,
        cutoff_frequency: float = 60.0,
        filter_taps: int = 10,
        filter_kernel: tuple[float, ...] | None = None,
    ) -> FilteredPulse:
        """Generate the randomized Fourier seeds used in the paper."""

        if filter_kernel is not None:
            filter_taps = len(filter_kernel)
        if not 1 < filter_taps <= n_time_steps:
            raise ValueError("filter_taps must lie in [2, n_time_steps].")
        n_parameters = n_time_steps + 1 - filter_taps
        if n_parameters < 2:
            raise ValueError("A random pulse requires at least two parameters per control.")
        times = np.linspace(0.0, n_parameters * time_step, n_parameters)
        coefficients = np.empty((n_controls, n_parameters))

        for channel in range(n_controls):
            frequencies = rng.uniform(*frequency_range, size=n_waves)
            amplitudes = rng.uniform(0.0, 1.0, size=n_waves)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
            waveform = np.sum(
                amplitudes[:, None]
                * np.cos(2.0 * np.pi * frequencies[:, None] * times + phases[:, None]),
                axis=0,
            )
            waveform -= waveform.mean()
            scale = np.max(np.abs(waveform))
            if scale == 0:
                raise RuntimeError("Random pulse generation produced a zero waveform.")
            coefficients[channel] = max_amplitude * waveform / scale

        return cls(
            n_controls=n_controls,
            n_time_steps=n_time_steps,
            time_step=time_step,
            coefficients=coefficients,
            rise_time=rise_time,
            envelope=envelope,
            cutoff_frequency=cutoff_frequency,
            filter_taps=filter_taps,
            filter_kernel=filter_kernel,
        )


def build_chebyshev_basis(
    n_time_steps: int,
    n_basis: int,
    *,
    normalization: ChebyshevNormalization = "rms",
) -> RealArray:
    """Evaluate orders ``0,...,n_basis-1`` on the normalized PWC grid.

    RMS normalization keeps coefficient scales comparable across polynomial
    orders.  The recurrence and normalization match the reference VNQC GRAPE
    implementation while remaining independent of that project.
    """

    if n_time_steps < 2:
        raise ValueError("n_time_steps must exceed one.")
    if not 1 <= n_basis <= n_time_steps:
        raise ValueError("n_basis must lie in [1, n_time_steps].")
    if normalization not in {"rms", "none"}:
        raise ValueError("normalization must be 'rms' or 'none'.")

    grid = np.linspace(-1.0, 1.0, n_time_steps)
    basis = np.empty((n_time_steps, n_basis), dtype=float)
    basis[:, 0] = 1.0
    if n_basis > 1:
        basis[:, 1] = grid
    for order in range(2, n_basis):
        basis[:, order] = 2.0 * grid * basis[:, order - 1] - basis[:, order - 2]
    if normalization == "rms":
        rms = np.sqrt(np.mean(basis**2, axis=0))
        basis = basis / rms
    return basis


@dataclass
class ChebyshevPulse:
    """Chebyshev controls followed by tanh, causal FIR, and an envelope.

    ``n_basis`` is the number of basis functions, so ``K=8`` represents
    Chebyshev orders zero through seven.  Propagation still receives one PWC
    amplitude per physical time step; only the optimization coordinates are
    reduced.
    """

    n_controls: int
    n_time_steps: int
    time_step: float
    coefficients: np.ndarray
    n_basis: int = DEFAULT_CHEBYSHEV_BASIS_SIZE
    normalization: ChebyshevNormalization = "rms"
    amplitude_limit: float = DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT
    rise_time: float = 0.01
    envelope: PulseEnvelope | None = None
    cutoff_frequency: float = 60.0
    filter_taps: int = 10
    filter_kernel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_controls < 1 or self.n_time_steps < 2:
            raise ValueError("n_controls must be positive and n_time_steps must exceed one.")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive.")
        if not 1 <= self.n_basis <= self.n_time_steps:
            raise ValueError("n_basis must lie in [1, n_time_steps].")
        if self.normalization not in {"rms", "none"}:
            raise ValueError("normalization must be 'rms' or 'none'.")
        if self.amplitude_limit <= 0:
            raise ValueError("amplitude_limit must be positive.")
        if self.envelope is None:
            self.envelope = BlackmanEnvelope(self.rise_time)
        if self.filter_kernel is not None:
            self.filter_kernel = tuple(float(value) for value in self.filter_kernel)
            self.filter_taps = len(self.filter_kernel)
        if not 1 < self.filter_taps <= self.n_time_steps:
            raise ValueError("filter_taps must lie in [2, n_time_steps].")

        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = (self.n_controls, self.n_basis)
        if coefficients.shape == (self.n_controls * self.n_basis,):
            coefficients = coefficients.reshape(expected)
        if coefficients.shape != expected:
            raise ValueError(f"coefficients has shape {coefficients.shape}; expected {expected}.")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must contain only finite values.")
        self.coefficients = coefficients.copy()

    @property
    def n_parameters_per_control(self) -> int:
        """Return the configurable number of Chebyshev coefficients ``K``."""

        return self.n_basis

    @property
    def n_parameters(self) -> int:
        """Return the total number of optimization parameters."""

        return self.n_controls * self.n_basis

    @cached_property
    def basis(self) -> RealArray:
        """Return the normalized Chebyshev evaluation matrix."""

        return build_chebyshev_basis(
            self.n_time_steps,
            self.n_basis,
            normalization=self.normalization,
        )

    @cached_property
    def envelope_values(self) -> RealArray:
        """Return and validate the post-LTI boundary envelope."""

        if self.envelope is None:  # pragma: no cover - normalized in __post_init__.
            raise RuntimeError("Pulse envelope was not initialized.")
        values = np.asarray(
            self.envelope.values(self.n_time_steps, self.time_step),
            dtype=float,
        )
        if values.shape != (self.n_time_steps,) or not np.all(np.isfinite(values)):
            raise ValueError("envelope must return one finite value per time step.")
        return values

    @cached_property
    def fir_kernel(self) -> RealArray:
        """Return the causal FIR kernel in the paper's MHz/us convention."""

        if self.filter_kernel is not None:
            return np.asarray(self.filter_kernel, dtype=float)
        normalized_cutoff = 2.0 * self.cutoff_frequency * self.time_step
        if not 0 < normalized_cutoff < 1:
            raise ValueError(
                "cutoff_frequency is outside the range supported by the chosen time grid."
            )
        return np.asarray(firwin(self.filter_taps, normalized_cutoff), dtype=float)

    @cached_property
    def lti_matrix(self) -> RealArray:
        """Return ``F`` such that each channel is filtered as ``F @ raw``."""

        matrix = np.zeros((self.n_time_steps, self.n_time_steps), dtype=float)
        for row in range(self.n_time_steps):
            max_tap = min(self.filter_taps - 1, row)
            for tap in range(max_tap + 1):
                matrix[row, row - tap] = self.fir_kernel[tap]
        return matrix

    def _unbounded(self, coefficients: np.ndarray | None = None) -> RealArray:
        values = (
            self.coefficients if coefficients is None else np.asarray(coefficients, dtype=float)
        )
        values = values.reshape(self.n_controls, self.n_basis)
        return values @ self.basis.T

    def amplitudes(self, coefficients: np.ndarray | None = None) -> RealArray:
        """Apply Chebyshev, tanh, LTI, and envelope maps in that order."""

        unbounded = self._unbounded(coefficients)
        limited = self.amplitude_limit * np.tanh(unbounded / self.amplitude_limit)
        filtered = limited @ self.lti_matrix.T
        return filtered * self.envelope_values[None, :]

    def pullback(
        self,
        amplitude_gradient: np.ndarray,
        coefficients: np.ndarray | None = None,
    ) -> RealArray:
        """Back-propagate an applied-control gradient through the full chain."""

        gradient = np.asarray(amplitude_gradient, dtype=float)
        expected = (self.n_controls, self.n_time_steps)
        if gradient.shape != expected:
            raise ValueError(f"amplitude_gradient has shape {gradient.shape}; expected {expected}.")
        grad_filtered = gradient * self.envelope_values[None, :]
        grad_limited = grad_filtered @ self.lti_matrix
        scaled = self._unbounded(coefficients) / self.amplitude_limit
        grad_unbounded = grad_limited * (1.0 - np.tanh(scaled) ** 2)
        return grad_unbounded @ self.basis

    def flattened_coefficients(self) -> RealArray:
        """Return a copy suitable for SciPy optimizers."""

        return self.coefficients.reshape(-1).copy()

    def parameter_bounds(
        self,
        max_control_amplitude: float,
    ) -> None:
        """Leave Chebyshev coefficients unbounded; tanh limits the waveform."""

        return None

    def metadata(self) -> dict[str, object]:
        """Describe the complete coefficient-to-applied-control map."""

        return {
            "control_basis": "chebyshev",
            "chebyshev_basis_size": self.n_basis,
            "chebyshev_orders": [0, self.n_basis - 1],
            "chebyshev_normalization": self.normalization,
            "tanh_amplitude_limit_rad_per_us": self.amplitude_limit,
            "tanh_amplitude_limit_mhz": self.amplitude_limit / (2.0 * np.pi),
            "parameter_count_per_control": self.n_parameters_per_control,
            "control_chain": ["Chebyshev", "tanh", "LTI", "envelope"],
            "envelope": type(self.envelope).__name__,
            "rise_time_us": self.rise_time,
            "cutoff_frequency_mhz": self.cutoff_frequency,
            "filter_taps": self.filter_taps,
            "filter_kernel": (None if self.filter_kernel is None else list(self.filter_kernel)),
        }

    @classmethod
    def random(
        cls,
        *,
        n_controls: int,
        n_time_steps: int,
        time_step: float,
        n_basis: int,
        rng: np.random.Generator,
        frequency_range: tuple[float, float] = (1.0, 30.0),
        initial_max_amplitude: float = 30.0 * 2.0 * np.pi,
        n_waves: int = 10,
        normalization: ChebyshevNormalization = "rms",
        amplitude_limit: float = DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT,
        rise_time: float = 0.01,
        envelope: PulseEnvelope | None = None,
        cutoff_frequency: float = 60.0,
        filter_taps: int = 10,
        filter_kernel: tuple[float, ...] | None = None,
    ) -> ChebyshevPulse:
        """Project reproducible random Fourier seeds into the Chebyshev space."""

        basis = build_chebyshev_basis(
            n_time_steps,
            n_basis,
            normalization=normalization,
        )
        times = np.linspace(0.0, (n_time_steps - 1) * time_step, n_time_steps)
        coefficients = np.empty((n_controls, n_basis), dtype=float)
        for channel in range(n_controls):
            frequencies = rng.uniform(*frequency_range, size=n_waves)
            weights = rng.uniform(0.0, 1.0, size=n_waves)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=n_waves)
            target = np.sum(
                weights[:, None]
                * np.cos(2.0 * np.pi * frequencies[:, None] * times + phases[:, None]),
                axis=0,
            )
            target -= target.mean()
            projected, *_ = np.linalg.lstsq(basis, target, rcond=None)
            reconstructed = basis @ projected
            scale = np.max(np.abs(reconstructed))
            if scale == 0:
                raise RuntimeError("Random pulse generation produced a zero waveform.")
            coefficients[channel] = initial_max_amplitude * projected / scale

        return cls(
            n_controls=n_controls,
            n_time_steps=n_time_steps,
            time_step=time_step,
            coefficients=coefficients,
            n_basis=n_basis,
            normalization=normalization,
            amplitude_limit=amplitude_limit,
            rise_time=rise_time,
            envelope=envelope,
            cutoff_frequency=cutoff_frequency,
            filter_taps=filter_taps,
            filter_kernel=filter_kernel,
        )
