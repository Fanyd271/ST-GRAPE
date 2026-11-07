"""Builders for the QFT and GHZ experiments reported in the paper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qutip import basis, ghz_state, qeye, qft

from .problem import (
    ColoredNoise,
    ControlDrift,
    ControlProblem,
    ControlTask,
    CorrelationModel,
    RobustnessModel,
    RobustPenaltyMode,
    SmoothOneOverFCorrelation,
    TaylorNoise,
)
from .systems import CouplingModel, SpinChain, build_spin_chain

PAPER_GHZ_DURATION_US = 0.080
PAPER_QFT_DURATION_US = 0.300


def default_paper_duration(task: ControlTask | str) -> float:
    """Return the published GHZ/QFT pulse duration in microseconds."""

    selected_task = ControlTask(task)
    return (
        PAPER_GHZ_DURATION_US
        if selected_task == ControlTask.STATE_PREPARATION
        else PAPER_QFT_DURATION_US
    )


def build_paper_problem(
    n_qubits: int,
    task: ControlTask,
    *,
    time_step: float = 0.0005,
    duration: float | None = None,
    coupling: CouplingModel | str = "XX+YY",
) -> tuple[SpinChain, ControlProblem]:
    """Build a nominal ZZ- or XX+YY-chain QFT/GHZ control problem.

    ``duration`` is expressed in microseconds.  Omitting it preserves the
    published 80 ns GHZ and 300 ns QFT settings.
    """

    chain = build_spin_chain(n_qubits, coupling=coupling)
    dimension = 2**n_qubits
    task = ControlTask(task)
    if task == ControlTask.STATE_PREPARATION:
        initial = basis(dimension, 0)
        initial.dims = [[2] * n_qubits, [1] * n_qubits]
        target = ghz_state(n_qubits)
    else:
        initial = qeye(dimension)
        initial.dims = chain.drift_hamiltonian.dims
        target = qft(dimension)
        target.dims = chain.drift_hamiltonian.dims
    resolved_duration = default_paper_duration(task) if duration is None else duration
    if not np.isfinite(resolved_duration) or resolved_duration <= 0.0:
        raise ValueError("duration must be finite and positive.")
    return chain, ControlProblem(
        drift_hamiltonian=chain.drift_hamiltonian,
        control_hamiltonians=chain.control_hamiltonians,
        initial=initial,
        target=target,
        duration=resolved_duration,
        time_step=time_step,
        task=task,
    )


def build_static_coupling_robustness(
    chain: SpinChain,
    *,
    order: int = 2,
    standard_deviation_mhz: float = 1.0,
    penalty_weight: float = 1.0,
    coupling_link_indices: Sequence[int] | None = None,
) -> RobustnessModel:
    """Build independent fluctuations on selected nominal nearest-neighbour links."""

    links = (
        tuple(range(len(chain.coupling_operators)))
        if coupling_link_indices is None
        else tuple(coupling_link_indices)
    )
    if (
        not links
        or len(set(links)) != len(links)
        or any(index < 0 or index >= len(chain.coupling_operators) for index in links)
    ):
        raise ValueError("coupling_link_indices must contain distinct valid link indices.")

    operators = tuple(
        standard_deviation_mhz * 2.0 * np.pi * chain.coupling_operators[index] for index in links
    )
    return RobustnessModel(
        taylor=TaylorNoise(
            static_operators=operators,
            order=order,
            penalty_weight=penalty_weight,
        )
    )


# Five positive exponentials used in the paper to approximate the 1/f-noise
# correlation function over the 30 ns state-preparation interval.  The old
# implementation stored the rates with a negative sign as diagonal generator
# coefficients; ``ColoredNoise`` stores the corresponding positive rates.
MULTISOURCE_GHZ_CORRELATION_WEIGHTS = (
    0.15456171,
    14.31035717,
    12.33658720,
    3.48120990,
    1.94791393,
)
MULTISOURCE_GHZ_DECAY_RATES = (
    1.42901055,
    3.77212369e-5,
    5.01679482,
    108.253319,
    411.875086,
)

# Five-exponential fit of the same 1/f-like process over the longer 100 ns QFT
# gate interval.  These values come from Sec. V B and the original
# ``multi_noise_sources.py`` implementation; they must not be replaced by the
# 30 ns GHZ fit above.
MULTISOURCE_QFT_CORRELATION_WEIGHTS = (
    3.59903953,
    4.92895840,
    19.32652322,
    2.51723057,
    1.85804726,
)
MULTISOURCE_QFT_DECAY_RATES = (
    31.2909634,
    9.82324153e-6,
    0.871971430,
    134.182283,
    415.489942,
)

ONE_OVER_F_LOWER_CUTOFF = 2.0 * np.pi * 1e-5
ONE_OVER_F_UPPER_CUTOFF = 2.0 * np.pi * 1e2
ONE_OVER_F_REFERENCE_CORRELATION = SmoothOneOverFCorrelation(
    lower_cutoff=ONE_OVER_F_LOWER_CUTOFF,
    upper_cutoff=ONE_OVER_F_UPPER_CUTOFF,
)


def _multisource_penalty_configuration(
    *,
    order: int,
    n_coupling: int,
    n_frequency: int,
    n_control: int,
    has_colored: bool,
    first_order_penalty: float,
    second_order_penalty: float,
    coupling_weight: float,
    frequency_drift_weight: float,
    control_drift_weight: float,
    colored_weight: float,
) -> tuple[tuple[float, ...], float, tuple[float, ...], float]:
    """Resolve normalized group budgets into Taylor and colored coefficients."""

    if order not in (1, 2):
        raise ValueError("Normalized multi-source weights currently support order 1 or 2.")
    penalties = np.asarray(
        [first_order_penalty, second_order_penalty],
        dtype=float,
    )
    group_weights = np.asarray(
        [
            coupling_weight,
            frequency_drift_weight,
            control_drift_weight,
            colored_weight,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(penalties)) or np.any(penalties < 0.0):
        raise ValueError("Order penalties must be finite and non-negative.")
    if not np.all(np.isfinite(group_weights)) or np.any(group_weights < 0.0):
        raise ValueError("Noise-family weights must be finite and non-negative.")
    if not np.isclose(float(np.sum(group_weights)), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "coupling_weight + frequency_drift_weight + "
            "control_drift_weight + colored_weight must equal 1."
        )
    source_counts = np.asarray(
        [n_coupling, n_frequency, n_control, int(has_colored)],
        dtype=int,
    )
    if np.any(source_counts < 0) or not np.any(source_counts > 0):
        raise ValueError("At least one physical noise source is required.")
    if np.any((source_counts == 0) & (group_weights > 1e-15)):
        raise ValueError("A disabled noise family must have zero group weight.")

    coupling_source_weight = 0.0 if n_coupling == 0 else coupling_weight / n_coupling
    frequency_source_weight = 0.0 if n_frequency == 0 else frequency_drift_weight / n_frequency
    static_source_weights = (
        *(coupling_source_weight for _ in range(n_coupling)),
        *(frequency_source_weight for _ in range(n_frequency)),
    )
    control_source_weight = 0.0 if n_control == 0 else control_drift_weight / n_control
    degree_weights = (first_order_penalty,)
    if order == 2 and n_coupling + n_frequency + n_control > 0:
        source_weights = np.asarray(
            [
                *static_source_weights,
                *(control_source_weight for _ in range(n_control)),
            ],
            dtype=float,
        )
        raw_second_order_sum = 0.5 * (
            float(np.sum(source_weights)) ** 2 + float(np.dot(source_weights, source_weights))
        )
        if raw_second_order_sum == 0.0:
            if second_order_penalty > 0.0:
                raise ValueError(
                    "A positive second_order_penalty requires a positive coupling, "
                    "frequency-drift, or control-drift weight."
                )
            second_order_multiplier = 0.0
        else:
            second_order_multiplier = second_order_penalty / raw_second_order_sum
        degree_weights = (first_order_penalty, second_order_multiplier)
    colored_penalty_weight = first_order_penalty * colored_weight
    return (
        static_source_weights,
        control_source_weight,
        degree_weights,
        colored_penalty_weight,
    )


def build_multisource_spin_chain_robustness(
    chain: SpinChain,
    *,
    order: int = 2,
    static_standard_deviation_mhz: float = 1.0,
    control_drift_standard_deviation: float = 0.01,
    colored_standard_deviation_mhz: float = 1.0,
    first_order_penalty: float = 1.0,
    second_order_penalty: float = 1.0,
    coupling_weight: float = 1.0 / 3.0,
    frequency_drift_weight: float = 0.0,
    control_drift_weight: float = 1.0 / 3.0,
    colored_weight: float = 1.0 / 3.0,
    correlation_weights: tuple[float, ...] = MULTISOURCE_GHZ_CORRELATION_WEIGHTS,
    decay_rates: tuple[float, ...] = MULTISOURCE_GHZ_DECAY_RATES,
    reference_correlation: CorrelationModel | None = ONE_OVER_F_REFERENCE_CORRELATION,
    static_coupling_pairs: Sequence[tuple[int, int]] | None = None,
    frequency_drift_qubits: Sequence[int] = (),
    frequency_drift_standard_deviation_mhz: float = 1.0,
    control_drift_qubits: Sequence[int] = (1, 2),
    colored_noise_qubit: int | None = 3,
    penalty_mode: RobustPenaltyMode | str = RobustPenaltyMode.SQUARED,
    rms_smoothing: float = 1e-12,
    p1_smoothing: float | None = None,
    p2_smoothing: float | None = None,
    pc_smoothing: float | None = None,
    p1_coefficient: float = 1.0,
    p2_coefficient: float = 1.0,
    pc_coefficient: float = 1.0,
) -> RobustnessModel:
    r"""Build a heterogeneous nearest-neighbour spin-chain noise model.

    By default, every nearest-neighbour nominal coupling has an independent
    quasistatic strength fluctuation. ``frequency_drift_qubits`` adds
    independent local-Z frequency offsets, while ``control_drift_qubits`` adds
    independent X/Y amplitude drifts. When ``colored_noise_qubit`` is not
    ``None``, a separate Van Loan component describes 1/f-like frequency noise
    on that site. Qubit indices in this low-level builder are zero-based.

    ``static_coupling_pairs`` can select a reviewed subset of adjacent links.
    This is used by the fixed four-qubit QFT reproduction, while scalable GHZ
    experiments normally leave it as ``None`` to cover the complete chain.

    ``coupling_weight``, ``frequency_drift_weight``,
    ``control_drift_weight``, and ``colored_weight`` are non-negative group
    weights that must sum to one. The complete first-order budget sums to
    ``first_order_penalty``. Degree-two Taylor product weights are normalized
    among themselves and sum to ``second_order_penalty``.
    For gate synthesis, the objective layer applies the
    dimension-normalized Frobenius norm :math:`\|X_p\|_F^2/d`; state
    preparation uses the ordinary vector norm.
    """

    n_qubits = len(chain.z_operators)
    pairs = (
        tuple((site, site + 1) for site in range(n_qubits - 1))
        if static_coupling_pairs is None
        else tuple(static_coupling_pairs)
    )
    if len(set(pairs)) != len(pairs):
        raise ValueError("static_coupling_pairs cannot contain duplicates.")
    for left, right in pairs:
        if left < 0 or right >= n_qubits or right != left + 1:
            raise ValueError(
                "static_coupling_pairs must contain valid adjacent, zero-based qubit indices."
            )

    drift_qubits = tuple(control_drift_qubits)
    if len(set(drift_qubits)) != len(drift_qubits):
        raise ValueError("control_drift_qubits cannot contain duplicates.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in drift_qubits):
        raise ValueError("control_drift_qubits contains an out-of-range qubit.")
    frequency_qubits = tuple(frequency_drift_qubits)
    if len(set(frequency_qubits)) != len(frequency_qubits):
        raise ValueError("frequency_drift_qubits cannot contain duplicates.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in frequency_qubits):
        raise ValueError("frequency_drift_qubits contains an out-of-range qubit.")
    if colored_noise_qubit is not None and (
        colored_noise_qubit < 0 or colored_noise_qubit >= n_qubits
    ):
        raise ValueError("colored_noise_qubit is out of range.")

    noise_strengths = (
        (
            "static_standard_deviation_mhz",
            static_standard_deviation_mhz,
            bool(pairs),
        ),
        (
            "frequency_drift_standard_deviation_mhz",
            frequency_drift_standard_deviation_mhz,
            bool(frequency_qubits),
        ),
        (
            "control_drift_standard_deviation",
            control_drift_standard_deviation,
            bool(drift_qubits),
        ),
        (
            "colored_standard_deviation_mhz",
            colored_standard_deviation_mhz,
            colored_noise_qubit is not None,
        ),
    )
    for name, strength, enabled in noise_strengths:
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        if enabled and strength == 0.0:
            raise ValueError(f"{name} must be positive when its noise family is enabled.")
    if colored_noise_qubit is None and not np.isclose(
        colored_weight,
        0.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("colored_weight must be zero when colored noise is disabled.")

    static_scale = 2.0 * np.pi * static_standard_deviation_mhz
    coupling_operators = tuple(static_scale * chain.coupling_operators[left] for left, _ in pairs)
    frequency_scale = 2.0 * np.pi * frequency_drift_standard_deviation_mhz
    frequency_operators = tuple(
        frequency_scale * chain.z_operators[qubit] for qubit in frequency_qubits
    )
    static_operators = coupling_operators + frequency_operators

    # Control channels are ordered as X_1,...,X_n,Y_1,...,Y_n.  Zero-based
    # channel q and n+q are therefore the X/Y controls on physical site q.
    drift_channels = drift_qubits + tuple(n_qubits + qubit for qubit in drift_qubits)
    control_drifts = tuple(
        ControlDrift(
            channel=channel,
            scale=control_drift_standard_deviation,
        )
        for channel in drift_channels
    )

    (
        static_source_weights,
        control_drift_source_weight,
        degree_penalty_weights,
        colored_penalty_weight,
    ) = _multisource_penalty_configuration(
        order=order,
        n_coupling=len(coupling_operators),
        n_frequency=len(frequency_operators),
        n_control=len(control_drifts),
        has_colored=colored_noise_qubit is not None,
        first_order_penalty=first_order_penalty,
        second_order_penalty=second_order_penalty,
        coupling_weight=coupling_weight,
        frequency_drift_weight=frequency_drift_weight,
        control_drift_weight=control_drift_weight,
        colored_weight=colored_weight,
    )

    colored: tuple[ColoredNoise, ...] = ()
    if colored_noise_qubit is not None:
        normalization_correlation = (
            reference_correlation if reference_correlation is not None else None
        )
        correlation_variance = (
            float(sum(correlation_weights))
            if normalization_correlation is None
            else float(normalization_correlation.covariance(np.asarray([0.0], dtype=float))[0])
        )
        if not np.isfinite(correlation_variance) or correlation_variance <= 0:
            raise ValueError("The colored-noise correlation variance must be positive.")
        colored_strength = (
            2.0 * np.pi * colored_standard_deviation_mhz / np.sqrt(correlation_variance)
        )
        colored = (
            ColoredNoise(
                operator=chain.z_operators[colored_noise_qubit],
                correlation_weights=correlation_weights,
                decay_rates=decay_rates,
                strength=colored_strength,
                penalty_weight=colored_penalty_weight,
                reference_correlation=reference_correlation,
            ),
        )
    taylor = None
    if static_operators or control_drifts:
        taylor = TaylorNoise(
            static_operators=static_operators,
            control_drifts=control_drifts,
            order=order,
            penalty_weight=first_order_penalty,
            degree_penalty_weights=degree_penalty_weights,
            static_source_weights=static_source_weights,
            control_drift_source_weight=control_drift_source_weight,
        )
    return RobustnessModel(
        taylor=taylor,
        colored=colored,
        penalty_mode=RobustPenaltyMode(penalty_mode),
        rms_smoothing=rms_smoothing,
        p1_smoothing=p1_smoothing,
        p2_smoothing=p2_smoothing,
        pc_smoothing=pc_smoothing,
        p1_coefficient=p1_coefficient,
        p2_coefficient=p2_coefficient,
        pc_coefficient=pc_coefficient,
    )


def build_multisource_qft_robustness(
    chain: SpinChain,
    *,
    order: int = 2,
    static_standard_deviation_mhz: float = 1.0,
    control_drift_standard_deviation: float = 0.01,
    colored_standard_deviation_mhz: float = 1.0,
    first_order_penalty: float = 1.0,
    second_order_penalty: float = 1.0,
    coupling_weight: float = 1.0 / 3.0,
    frequency_drift_weight: float = 0.0,
    control_drift_weight: float = 1.0 / 3.0,
    colored_weight: float = 1.0 / 3.0,
    static_coupling_pairs: Sequence[tuple[int, int]] = ((0, 1), (1, 2)),
    frequency_drift_qubits: Sequence[int] = (),
    frequency_drift_standard_deviation_mhz: float = 1.0,
    control_drift_qubits: Sequence[int] = (1, 2),
    colored_noise_qubit: int | None = 3,
    penalty_mode: RobustPenaltyMode | str = RobustPenaltyMode.SQUARED,
    rms_smoothing: float = 1e-12,
    p1_smoothing: float | None = None,
    p2_smoothing: float | None = None,
    pc_smoothing: float | None = None,
    p1_coefficient: float = 1.0,
    p2_coefficient: float = 1.0,
    pc_coefficient: float = 1.0,
) -> RobustnessModel:
    """Build a QFT multi-source model using the 100 ns correlation fit.

    Defaults reproduce the four-qubit Sec. V B model.  Larger systems can pass
    their nearest-neighbour links and affected qubits explicitly.  Indices are
    zero-based at this reusable builder boundary.
    """

    return build_multisource_spin_chain_robustness(
        chain,
        order=order,
        static_standard_deviation_mhz=static_standard_deviation_mhz,
        control_drift_standard_deviation=control_drift_standard_deviation,
        colored_standard_deviation_mhz=colored_standard_deviation_mhz,
        first_order_penalty=first_order_penalty,
        second_order_penalty=second_order_penalty,
        coupling_weight=coupling_weight,
        frequency_drift_weight=frequency_drift_weight,
        control_drift_weight=control_drift_weight,
        colored_weight=colored_weight,
        correlation_weights=MULTISOURCE_QFT_CORRELATION_WEIGHTS,
        decay_rates=MULTISOURCE_QFT_DECAY_RATES,
        static_coupling_pairs=static_coupling_pairs,
        frequency_drift_qubits=frequency_drift_qubits,
        frequency_drift_standard_deviation_mhz=(frequency_drift_standard_deviation_mhz),
        control_drift_qubits=control_drift_qubits,
        colored_noise_qubit=colored_noise_qubit,
        penalty_mode=penalty_mode,
        rms_smoothing=rms_smoothing,
        p1_smoothing=p1_smoothing,
        p2_smoothing=p2_smoothing,
        pc_smoothing=pc_smoothing,
        p1_coefficient=p1_coefficient,
        p2_coefficient=p2_coefficient,
        pc_coefficient=pc_coefficient,
    )
