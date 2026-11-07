"""Side-effect-free Monte Carlo validation of optimized controls."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .problem import (
    ColoredNoise,
    ControlProblem,
    ControlTask,
    CorrelationModel,
    RobustnessModel,
    TaylorNoise,
)


class ColoredNoiseSamplingMethod(StrEnum):
    """Available colored-noise generators for Monte Carlo validation."""

    AUTO = "auto"
    REFERENCE_GAUSSIAN = "reference-gaussian"
    RTP = "rtp"


@dataclass(frozen=True)
class NoiseRealization:
    """One standardized realization of all configured uncertainty sources."""

    static_amplitudes: NDArray[np.float64]
    control_amplitudes: NDArray[np.float64]
    colored_trajectories: NDArray[np.float64]


@dataclass(frozen=True)
class GaussianCirculantEmbedding:
    """FFT factorization for exact stationary Gaussian samples on one grid.

    The requested Toeplitz covariance is embedded in a positive semidefinite
    circulant covariance.  Once constructed, each sample costs one forward and
    one inverse FFT.  Tiny negative eigenvalues caused by roundoff are clipped;
    a genuinely indefinite embedding is enlarged instead.
    """

    n_time_steps: int
    square_root_eigenvalues: NDArray[np.float64]

    @property
    def embedding_size(self) -> int:
        """Length of the real circulant Gaussian vector generated per draw."""

        return int(self.square_root_eigenvalues.size)

    def sample(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate one exact real Gaussian trajectory on the requested grid."""

        white_noise = rng.normal(size=self.embedding_size)
        embedded = np.fft.ifft(self.square_root_eigenvalues * np.fft.fft(white_noise)).real
        return np.asarray(embedded[: self.n_time_steps], dtype=float)


def prepare_gaussian_circulant_embedding(
    correlation: CorrelationModel,
    *,
    n_time_steps: int,
    time_step: float,
    max_expansions: int = 12,
) -> GaussianCirculantEmbedding:
    """Factor a stationary covariance for exact Gaussian grid sampling.

    The first ``n_time_steps`` entries have covariance
    ``correlation.covariance(abs(i-j) * time_step)``.  The embedding length is
    doubled until its FFT eigenvalues are non-negative up to roundoff.
    """

    if n_time_steps < 1:
        raise ValueError("n_time_steps must be positive.")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")
    if max_expansions < 0:
        raise ValueError("max_expansions must be non-negative.")

    minimum_size = max(1, 2 * (n_time_steps - 1))
    embedding_size = 1 << (minimum_size - 1).bit_length()
    minimum_eigenvalue = float("nan")
    for _ in range(max_expansions + 1):
        indices = np.arange(embedding_size)
        circular_lags = np.minimum(indices, embedding_size - indices) * time_step
        covariance = np.asarray(correlation.covariance(circular_lags), dtype=float)
        if covariance.shape != (embedding_size,):
            raise ValueError("correlation.covariance returned an incompatible shape.")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("correlation.covariance returned non-finite values.")

        eigenvalues_complex = np.fft.fft(covariance)
        imaginary_scale = float(np.max(np.abs(np.imag(eigenvalues_complex))))
        real_scale = max(1.0, float(np.max(np.abs(np.real(eigenvalues_complex)))))
        if imaginary_scale > 1e-11 * real_scale:
            raise ValueError("The embedded stationary covariance is not symmetric.")
        eigenvalues = np.real(eigenvalues_complex)
        minimum_eigenvalue = float(np.min(eigenvalues))
        tolerance = 1e-12 * real_scale
        if minimum_eigenvalue >= -tolerance:
            return GaussianCirculantEmbedding(
                n_time_steps=n_time_steps,
                square_root_eigenvalues=np.sqrt(np.clip(eigenvalues, 0.0, None)),
            )
        embedding_size *= 2

    raise ValueError(
        "Could not find a positive circulant embedding after "
        f"{max_expansions} expansions; final minimum eigenvalue was "
        f"{minimum_eigenvalue:.6g}."
    )


def simulate_gaussian_correlated_noise(
    correlation: CorrelationModel,
    *,
    n_time_steps: int,
    time_step: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample the unfitted Gaussian reference process on a uniform grid."""

    embedding = prepare_gaussian_circulant_embedding(
        correlation,
        n_time_steps=n_time_steps,
        time_step=time_step,
    )
    return embedding.sample(rng)


def simulate_random_telegraph_noise(
    correlation_weights: tuple[float, ...],
    decay_rates: tuple[float, ...],
    *,
    n_time_steps: int,
    time_step: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample a process with ``C(t)=sum(a_k exp(-gamma_k |t|))``.

    Each term is represented by an independent stationary random telegraph
    process.  The exact discrete-time flip probability is
    ``(1 - exp(-gamma_k * dt)) / 2``.
    """

    weights = np.asarray(correlation_weights, dtype=float)
    rates = np.asarray(decay_rates, dtype=float)
    if weights.shape != rates.shape or weights.ndim != 1:
        raise ValueError("correlation_weights and decay_rates must be equal 1D arrays.")
    if np.any(weights <= 0) or np.any(rates <= 0):
        raise ValueError("Correlation weights and decay rates must be positive.")

    flip_probability = 0.5 * (1.0 - np.exp(-rates * time_step))
    signs = rng.choice((-1.0, 1.0), size=len(rates))
    trajectory = np.empty(n_time_steps, dtype=float)
    for time_index in range(n_time_steps):
        if time_index:
            signs *= np.where(rng.random(len(rates)) < flip_probability, -1.0, 1.0)
        trajectory[time_index] = np.dot(np.sqrt(weights), signs)
    return trajectory


def generate_noise_realizations(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
    *,
    n_samples: int,
    rng: np.random.Generator,
    colored_noise_sampling: ColoredNoiseSamplingMethod | str = (ColoredNoiseSamplingMethod.AUTO),
) -> tuple[NoiseRealization, ...]:
    """Generate standardized static, control, and colored realizations.

    ``auto`` uses an exact Gaussian sample of a colored source's unfitted
    reference correlation when one is available and otherwise falls back to
    the legacy RTP expansion.  ``reference-gaussian`` requires every colored
    source to provide a reference correlation.  ``rtp`` always samples the
    finite exponential expansion used by the optimizer.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    sampling_method = ColoredNoiseSamplingMethod(colored_noise_sampling)
    n_static = (
        len(robustness.taylor.static_operators)
        if robustness is not None and robustness.taylor is not None
        else 0
    )
    n_control = (
        len(robustness.taylor.control_drifts)
        if robustness is not None and robustness.taylor is not None
        else 0
    )
    n_colored = len(robustness.colored) if robustness is not None else 0
    colored_sources = robustness.colored if robustness is not None else ()
    prepared_sources = tuple(
        _prepare_colored_noise_source(
            source,
            sampling_method,
            n_time_steps=problem.n_time_steps,
            time_step=problem.time_step,
        )
        for source in colored_sources
    )
    realizations: list[NoiseRealization] = []
    for _ in range(n_samples):
        colored = np.empty((n_colored, problem.n_time_steps), dtype=float)
        for index, (source, resolved_method, embedding) in enumerate(prepared_sources):
            if resolved_method == ColoredNoiseSamplingMethod.REFERENCE_GAUSSIAN:
                if embedding is None:  # Defensive invariant for static type checkers.
                    raise RuntimeError("Missing Gaussian circulant embedding.")
                colored[index] = embedding.sample(rng)
            else:
                colored[index] = simulate_random_telegraph_noise(
                    source.correlation_weights,
                    source.decay_rates,
                    n_time_steps=problem.n_time_steps,
                    time_step=problem.time_step,
                    rng=rng,
                )
        realizations.append(
            NoiseRealization(
                static_amplitudes=rng.normal(size=n_static),
                control_amplitudes=rng.normal(size=n_control),
                colored_trajectories=colored,
            )
        )
    return tuple(realizations)


def _prepare_colored_noise_source(
    source: ColoredNoise,
    requested_method: ColoredNoiseSamplingMethod,
    *,
    n_time_steps: int,
    time_step: float,
) -> tuple[
    ColoredNoise,
    ColoredNoiseSamplingMethod,
    GaussianCirculantEmbedding | None,
]:
    """Resolve and pre-factor one colored source before drawing trajectories."""

    resolved_method = requested_method
    if requested_method == ColoredNoiseSamplingMethod.AUTO:
        resolved_method = (
            ColoredNoiseSamplingMethod.REFERENCE_GAUSSIAN
            if source.reference_correlation is not None
            else ColoredNoiseSamplingMethod.RTP
        )
    if resolved_method == ColoredNoiseSamplingMethod.REFERENCE_GAUSSIAN:
        if source.reference_correlation is None:
            raise ValueError(
                "reference-gaussian sampling requires reference_correlation "
                "on every colored-noise source."
            )
        embedding = prepare_gaussian_circulant_embedding(
            source.reference_correlation,
            n_time_steps=n_time_steps,
            time_step=time_step,
        )
    else:
        embedding = None
    return source, resolved_method, embedding


def colored_noise_sampling_metadata(
    robustness: RobustnessModel,
    requested_method: ColoredNoiseSamplingMethod | str,
) -> dict[str, object]:
    """Describe the requested and resolved validation model for result files."""

    requested = ColoredNoiseSamplingMethod(requested_method)
    sources: list[dict[str, object]] = []
    for index, source in enumerate(robustness.colored):
        resolved = requested
        if requested == ColoredNoiseSamplingMethod.AUTO:
            resolved = (
                ColoredNoiseSamplingMethod.REFERENCE_GAUSSIAN
                if source.reference_correlation is not None
                else ColoredNoiseSamplingMethod.RTP
            )
        reference = source.reference_correlation
        reference_variance = (
            None
            if reference is None
            else float(reference.covariance(np.asarray([0.0], dtype=float))[0])
        )
        sources.append(
            {
                "source_index": index,
                "resolved_method": resolved.value,
                "strength": source.strength,
                "optimization_process_variance": source.optimization_variance,
                "optimization_rms_hamiltonian_coefficient": (
                    float(source.strength * np.sqrt(source.optimization_variance))
                ),
                "reference_rms_hamiltonian_coefficient": (
                    None
                    if reference_variance is None
                    else float(source.strength * np.sqrt(reference_variance))
                ),
                "reference_correlation": (None if reference is None else reference.metadata()),
            }
        )
    return {"requested_method": requested.value, "sources": sources}


def _fidelity(problem: ControlProblem, terminal: NDArray[np.complex128]) -> float:
    target = problem.target.full()
    if problem.task == ControlTask.STATE_PREPARATION:
        return float(abs((target.conj().T @ terminal).item()) ** 2)
    dimension = problem.hilbert_dimension
    overlap = np.trace(target.conj().T @ terminal)
    return float((abs(overlap) ** 2 + dimension) / (dimension * (dimension + 1)))


def evaluate_realization(
    realization: NoiseRealization,
    problem: ControlProblem,
    controls: NDArray[np.float64],
    robustness: RobustnessModel | None,
) -> float:
    """Evaluate one noisy trajectory without modifying problem Hamiltonians."""

    amplitudes = np.asarray(controls, dtype=float)
    expected = (problem.n_controls, problem.n_time_steps)
    if amplitudes.shape != expected:
        raise ValueError(f"controls has shape {amplitudes.shape}; expected {expected}.")

    drift = problem.drift_hamiltonian.full().copy()
    control_hamiltonians = [
        hamiltonian.full().copy() for hamiltonian in problem.control_hamiltonians
    ]
    if robustness is not None and robustness.taylor is not None:
        taylor = robustness.taylor
        if len(realization.static_amplitudes) != len(taylor.static_operators):
            raise ValueError("static realization has the wrong number of amplitudes.")
        if len(realization.control_amplitudes) != len(taylor.control_drifts):
            raise ValueError("control realization has the wrong number of amplitudes.")
        for value, operator in zip(
            realization.static_amplitudes, taylor.static_operators, strict=True
        ):
            drift += value * operator.full()
        for value, uncertainty in zip(
            realization.control_amplitudes, taylor.control_drifts, strict=True
        ):
            control_hamiltonians[uncertainty.channel] *= 1.0 + value * uncertainty.scale

    colored_sources = robustness.colored if robustness is not None else ()
    if realization.colored_trajectories.shape != (
        len(colored_sources),
        problem.n_time_steps,
    ):
        raise ValueError("colored realization has the wrong shape.")

    state = np.asarray(problem.initial.full(), dtype=np.complex128).copy()
    for time_index in range(problem.n_time_steps):
        generator = drift.copy()
        for channel, hamiltonian in enumerate(control_hamiltonians):
            generator += amplitudes[channel, time_index] * hamiltonian
        for source_index, source in enumerate(colored_sources):
            generator += (
                realization.colored_trajectories[source_index, time_index]
                * source.scaled_operator.full()
            )
        propagator = np.asarray(
            expm(-1j * problem.time_step * generator),
            dtype=np.complex128,
        )
        state = propagator @ state
    return _fidelity(problem, state)


def sample_fidelities(
    problem: ControlProblem,
    controls: NDArray[np.float64],
    robustness: RobustnessModel | None,
    *,
    n_samples: int,
    seed: int,
    workers: int = 1,
    colored_noise_sampling: ColoredNoiseSamplingMethod | str = (ColoredNoiseSamplingMethod.AUTO),
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Generate deterministic samples and optionally report completed counts."""

    realizations = generate_noise_realizations(
        problem,
        robustness,
        n_samples=n_samples,
        rng=np.random.default_rng(seed),
        colored_noise_sampling=colored_noise_sampling,
    )
    evaluator = partial(
        evaluate_realization,
        problem=problem,
        controls=np.asarray(controls, dtype=float),
        robustness=robustness,
    )
    if workers < 1:
        raise ValueError("workers must be positive.")
    if workers == 1:
        evaluated = map(evaluator, realizations)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        evaluated = executor.map(evaluator, realizations)
    try:
        values = []
        for completed, value in enumerate(evaluated, start=1):
            values.append(value)
            if progress_callback is not None:
                progress_callback(completed, n_samples)
    finally:
        if executor is not None:
            executor.shutdown()
    return np.asarray(values, dtype=float)


def separate_noise_models(
    robustness: RobustnessModel,
    *,
    static_group_sizes: tuple[int, int] | None = None,
) -> dict[str, RobustnessModel]:
    """Return models containing independently sampled physical noise groups.

    By default all static operators remain in one ``static`` group. Supplying
    ``static_group_sizes=(n_coupling, n_frequency)`` separates their ordered
    operators into ``coupling`` and ``frequency`` groups. ``control`` and
    ``colored`` are always kept separate. Each model preserves physical scales
    and is intended for source-resolved Monte Carlo validation of one pulse.
    """

    separated: dict[str, RobustnessModel] = {}
    taylor = robustness.taylor
    if taylor is not None and taylor.static_operators:
        static_weights = taylor.source_weights[: len(taylor.static_operators)]
        if static_group_sizes is None:
            static_groups = (("static", 0, len(taylor.static_operators)),)
        else:
            if (
                len(static_group_sizes) != 2
                or any(size < 0 for size in static_group_sizes)
                or sum(static_group_sizes) != len(taylor.static_operators)
            ):
                raise ValueError(
                    "static_group_sizes must contain coupling/frequency counts "
                    "whose sum matches the static operators."
                )
            coupling_count, frequency_count = static_group_sizes
            static_groups = (
                ("coupling", 0, coupling_count),
                (
                    "frequency",
                    coupling_count,
                    coupling_count + frequency_count,
                ),
            )
        for name, start, stop in static_groups:
            if stop == start:
                continue
            separated[name] = RobustnessModel(
                taylor=TaylorNoise(
                    static_operators=taylor.static_operators[start:stop],
                    order=taylor.order,
                    penalty_weight=taylor.penalty_weight,
                    degree_penalty_weights=taylor.degree_penalty_weights,
                    static_source_weight=taylor.static_source_weight,
                    static_source_weights=tuple(static_weights[start:stop]),
                    control_drift_source_weight=(taylor.control_drift_source_weight),
                )
            )
    if taylor is not None and taylor.control_drifts:
        separated["control"] = RobustnessModel(
            taylor=TaylorNoise(
                control_drifts=taylor.control_drifts,
                order=taylor.order,
                penalty_weight=taylor.penalty_weight,
                degree_penalty_weights=taylor.degree_penalty_weights,
                static_source_weight=taylor.static_source_weight,
                control_drift_source_weight=taylor.control_drift_source_weight,
            )
        )
    if robustness.colored:
        separated["colored"] = RobustnessModel(colored=robustness.colored)
    return separated


def sample_fidelities_by_noise_source(
    problem: ControlProblem,
    controls: NDArray[np.float64],
    robustness: RobustnessModel,
    *,
    n_samples: int,
    seed: int,
    workers: int = 1,
    colored_noise_sampling: ColoredNoiseSamplingMethod | str = (ColoredNoiseSamplingMethod.AUTO),
    static_group_sizes: tuple[int, int] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Sample each configured noise family independently for one pulse.

    The same Monte Carlo seed and sample count are used for every family so
    source-resolved comparisons are deterministic and equally sized.  Noise
    families are never mixed in these samples.
    """

    return {
        source: sample_fidelities(
            problem,
            controls,
            model,
            n_samples=n_samples,
            seed=seed,
            workers=workers,
            colored_noise_sampling=colored_noise_sampling,
        )
        for source, model in separate_noise_models(
            robustness,
            static_group_sizes=static_group_sizes,
        ).items()
    }


def fidelity_statistics(values: NDArray[np.float64]) -> dict[str, float | int]:
    """Summarize one fidelity sample with central and tail statistics."""

    fidelities = np.asarray(values, dtype=float)
    if fidelities.ndim != 1 or fidelities.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    return {
        "samples": int(fidelities.size),
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity": float(np.std(fidelities)),
        "minimum_fidelity": float(np.min(fidelities)),
        "p05_fidelity": float(np.quantile(fidelities, 0.05)),
        "median_fidelity": float(np.median(fidelities)),
        "p95_fidelity": float(np.quantile(fidelities, 0.95)),
        "maximum_fidelity": float(np.max(fidelities)),
    }


def infidelity_statistics(values: NDArray[np.float64]) -> dict[str, float | int]:
    """Summarize fidelity samples using the error ``1 - fidelity``."""

    fidelities = np.asarray(values, dtype=float)
    if fidelities.ndim != 1 or fidelities.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    errors = np.clip(1.0 - fidelities, 0.0, None)
    return {
        "samples": int(errors.size),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "minimum_error": float(np.min(errors)),
        "p05_error": float(np.quantile(errors, 0.05)),
        "median_error": float(np.median(errors)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "maximum_error": float(np.max(errors)),
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity": float(np.std(fidelities)),
    }
