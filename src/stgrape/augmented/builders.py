"""Construction of Taylor and colored-noise augmented generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qutip import Qobj

from stgrape.objectives import SmoothRmsPenalty
from stgrape.problem import (
    ColoredNoise,
    ControlProblem,
    RobustnessModel,
    RobustPenaltyMode,
    TaylorNoise,
)

from .block_sparse import BlockOperator
from .indices import decrement, generate_multi_indices

RmsFeature = tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class AugmentedHamiltonian:
    """One augmented Hamiltonian and its objective-block bookkeeping.

    The stored operators are Hamiltonians.  Propagation backends convert them
    to the LMDE generator ``G = -1j * H`` at the right-hand-side boundary.
    Keeping the factor of ``-1j`` out of this data object avoids mixing the
    Hamiltonian convention used by the paper with the LMDE convention used by
    generic ODE solvers.
    """

    drift: BlockOperator
    controls: tuple[BlockOperator, ...]
    nominal_block: int
    penalty_blocks: tuple[int, ...]
    penalty_weights: tuple[float, ...]
    rms_penalties: tuple[SmoothRmsPenalty, ...] = ()

    @property
    def n_blocks(self) -> int:
        """Number of physical-space blocks in the augmented system."""

        return self.drift.n_blocks


def build_taylor_generators(
    problem: ControlProblem,
    noise: TaylorNoise,
    diagonal_controls: dict[int, Qobj],
) -> tuple[
    tuple[tuple[int, ...], ...],
    BlockOperator,
    dict[int, BlockOperator],
]:
    """Construct static and control-drift recursion generators.

    ``diagonal_controls`` contains each drift-affected control Hamiltonian in
    the basis of its commuting control group.
    """

    indices = generate_multi_indices(noise.n_sources, noise.order)
    index_map = {index: position for position, index in enumerate(indices)}
    n_blocks = len(indices)
    d = problem.hilbert_dimension

    static_entries: list[tuple[int, int, np.ndarray]] = []
    drift_entries: dict[int, list[tuple[int, int, np.ndarray]]] = {
        drift.channel: [] for drift in noise.control_drifts
    }
    n_static = len(noise.static_operators)

    for higher_index in indices:
        row = index_map[higher_index]
        for source in range(noise.n_sources):
            lower_index = decrement(higher_index, source)
            if lower_index is None:
                continue
            column = index_map[lower_index]
            if source < n_static:
                static_entries.append((row, column, noise.static_operators[source].full()))
            else:
                drift = noise.control_drifts[source - n_static]
                if drift.channel >= problem.n_controls:
                    raise ValueError(
                        f"Control-drift channel {drift.channel} exceeds the problem's "
                        f"{problem.n_controls} control channels."
                    )
                try:
                    diagonal = diagonal_controls[drift.channel]
                except KeyError as error:
                    raise ValueError(
                        f"No diagonal control representation for channel {drift.channel}."
                    ) from error
                drift_entries[drift.channel].append((row, column, drift.scale * diagonal.full()))

    static_generator = (
        BlockOperator.from_entries(n_blocks, static_entries)
        if static_entries
        else BlockOperator.zeros(n_blocks, d)
    )
    drift_generators = {
        channel: BlockOperator.from_entries(n_blocks, entries)
        for channel, entries in drift_entries.items()
    }
    return indices, static_generator, drift_generators


def taylor_block_penalty_weights(
    noise: TaylorNoise,
    indices: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[float, ...]:
    """Return direct-product weights for all non-nominal Taylor blocks.

    For a multi-index ``alpha`` of total degree ``k``, the block coefficient is
    ``degree_weight[k] * product(source_weight[i] ** alpha[i])``.  When no
    degree-specific weights are configured, ``penalty_weight`` is used for
    every degree.  The nominal zero multi-index is omitted.  Default unit
    source weights reproduce the original uniform Taylor penalty exactly.
    """

    resolved_indices = indices or generate_multi_indices(noise.n_sources, noise.order)
    source_weights = noise.source_weights
    weights: list[float] = []
    for multi_index in resolved_indices:
        if len(multi_index) != noise.n_sources:
            raise ValueError("Taylor multi-index has an incompatible source count.")
        if not any(multi_index):
            continue
        product = 1.0
        for source_weight, exponent in zip(source_weights, multi_index, strict=True):
            product *= source_weight**exponent
        degree = sum(multi_index)
        weights.append(noise.penalty_weight_for_degree(degree) * product)
    return tuple(weights)


def taylor_rms_penalties(
    noise: TaylorNoise,
    indices: tuple[tuple[int, ...], ...],
    robustness: RobustnessModel,
    *,
    degrees: frozenset[int] = frozenset((1, 2)),
) -> tuple[SmoothRmsPenalty, ...]:
    """Build physically scaled Gaussian RMS terms for Taylor orders one and two.

    Perturbation generators already include their one-standard-deviation
    physical scales.  First-order blocks are combined in quadrature.  The
    degree-two features implement the exact independent-Gaussian fourth moments
    for the repository's Taylor-coefficient convention.
    """

    if noise.order > 2:
        raise ValueError("RMS sensitivity penalties currently support Taylor order <= 2.")
    if not degrees.issubset({1, 2}):
        raise ValueError("RMS Taylor penalty degrees must be drawn from {1, 2}.")
    penalties: list[SmoothRmsPenalty] = []
    first_order_features = taylor_gaussian_rms_features(indices, degree=1)
    if 1 in degrees and first_order_features:
        penalties.append(
            SmoothRmsPenalty(
                features=first_order_features,
                coefficient=robustness.resolved_p1_coefficient,
                smoothing=robustness.resolved_p1_smoothing,
            )
        )

    if 2 in degrees and noise.order >= 2:
        second_order_features = taylor_gaussian_rms_features(indices, degree=2)
        if second_order_features:
            penalties.append(
                SmoothRmsPenalty(
                    features=second_order_features,
                    coefficient=robustness.resolved_p2_coefficient,
                    smoothing=robustness.resolved_p2_smoothing,
                )
            )
    return tuple(penalties)


def taylor_gaussian_rms_features(
    indices: tuple[tuple[int, ...], ...],
    *,
    degree: int,
) -> tuple[RmsFeature, ...]:
    """Return block combinations for one independent-Gaussian RMS moment."""

    positions = {index: position for position, index in enumerate(indices)}
    if degree == 1:
        return tuple(((position, 1.0),) for index, position in positions.items() if sum(index) == 1)
    if degree != 2:
        raise ValueError("Gaussian RMS features currently support degree one or two.")

    diagonal_positions = tuple(
        position for index, position in positions.items() if sum(index) == 2 and 2 in index
    )
    mixed_positions = tuple(
        position for index, position in positions.items() if sum(index) == 2 and 2 not in index
    )
    features: list[RmsFeature] = []
    if diagonal_positions:
        # ||sum_a X_aa||^2 is the squared second-order mean shift.
        features.append(tuple((position, 1.0) for position in diagonal_positions))
        # Two additional copies give E[xi_a^4] = 3 for each diagonal block.
        features.extend(((position, np.sqrt(2.0)),) for position in diagonal_positions)
    features.extend(((position, 1.0),) for position in mixed_positions)
    return tuple(features)


def colored_rms_penalties(
    robustness: RobustnessModel,
    output_blocks: tuple[int, ...] = (0,),
) -> tuple[SmoothRmsPenalty, ...]:
    """Return one joint smooth RMS norm over colored-noise mean shifts."""

    if not output_blocks:
        return ()

    return (
        SmoothRmsPenalty(
            features=tuple(((block, 1.0),) for block in output_blocks),
            coefficient=robustness.resolved_pc_coefficient,
            smoothing=robustness.resolved_pc_smoothing,
        ),
    )


def taylor_penalty_terms(
    noise: TaylorNoise,
    indices: tuple[tuple[int, ...], ...],
    robustness: RobustnessModel,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[SmoothRmsPenalty, ...]]:
    """Return terminal Taylor penalties for squared, RMS, or hybrid mode.

    Hybrid mode defines its first-order contribution as ``lambda_1 * P1**2``
    and applies a smooth RMS norm only to the Gaussian-moment degree-two
    features. The generic quadratic objective contains a factor of one half,
    so its internal block weights are twice the public P1 coefficient.
    """

    if robustness.penalty_mode == RobustPenaltyMode.SQUARED:
        return (
            tuple(range(len(indices) - 1)),
            taylor_block_penalty_weights(noise, indices),
            (),
        )
    if robustness.penalty_mode == RobustPenaltyMode.RMS:
        return (), (), taylor_rms_penalties(noise, indices, robustness)

    first_order_blocks = tuple(
        position for position, index in enumerate(indices) if sum(index) == 1
    )
    return (
        first_order_blocks,
        tuple(2.0 * robustness.resolved_p1_coefficient for _ in first_order_blocks),
        taylor_rms_penalties(
            noise,
            indices,
            robustness,
            degrees=frozenset((2,)),
        ),
    )


def colored_penalty_terms(
    colored: ColoredNoise,
    robustness: RobustnessModel,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[SmoothRmsPenalty, ...]]:
    """Return one colored component's terminal penalty convention."""

    if robustness.penalty_mode == RobustPenaltyMode.SQUARED:
        return (0,), (colored.penalty_weight,), ()
    return (), (), colored_rms_penalties(robustness)


def joint_colored_penalty_terms(
    colored: tuple[ColoredNoise, ...],
    output_blocks: tuple[int, ...],
    robustness: RobustnessModel,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[SmoothRmsPenalty, ...]]:
    """Return terminal penalties for one joint multi-colored-noise system.

    RMS and hybrid modes use equal source weights and apply one smooth norm to
    all source-specific mean-shift blocks.  Physical noise strengths already
    reside in each source's scaled operator.  Squared mode retains the legacy
    per-source ``penalty_weight`` multipliers.
    """

    if len(colored) != len(output_blocks):
        raise ValueError("Each colored-noise source requires one output block.")
    if robustness.penalty_mode == RobustPenaltyMode.SQUARED:
        return (
            output_blocks,
            tuple(source.penalty_weight for source in colored),
            (),
        )
    return (), (), colored_rms_penalties(robustness, output_blocks)


def build_colored_generators(
    problem: ControlProblem,
    noise: ColoredNoise,
) -> tuple[BlockOperator, BlockOperator]:
    """Construct the Van Loan drift propagator inputs for one colored source.

    Returns the block-diagonal drift generator and the strictly off-diagonal
    noise generator corresponding to Eqs. (74)--(80) of the paper.
    """

    n_terms = len(noise.correlation_weights)
    n_blocks = n_terms + 2
    d = problem.hilbert_dimension
    identity = np.eye(d, dtype=np.complex128)
    drift = problem.drift_hamiltonian.full()

    drift_blocks = [drift]
    drift_blocks.extend(drift - 1j * rate * identity for rate in noise.decay_rates)
    drift_blocks.append(drift)
    drift_generator = BlockOperator.block_diagonal(drift_blocks)

    error = noise.scaled_operator.full()
    entries: list[tuple[int, int, np.ndarray]] = []
    for term, weight in enumerate(noise.correlation_weights):
        entries.append((0, term + 1, weight * error))
        entries.append((term + 1, n_blocks - 1, error))
    error_generator = BlockOperator.from_entries(n_blocks, entries)
    return drift_generator, error_generator


def build_joint_colored_generators(
    problem: ControlProblem,
    noises: tuple[ColoredNoise, ...],
) -> tuple[BlockOperator, BlockOperator, tuple[int, ...], int]:
    """Construct one Van Loan system sharing the nominal block across sources.

    Every independent colored-noise source keeps its own output and exponential
    correlation blocks.  A shared nominal block avoids duplicated nominal
    propagation and lets the objective form one joint RMS over all outputs.
    """

    if not noises:
        raise ValueError("At least one colored-noise source is required.")

    d = problem.hilbert_dimension
    identity = np.eye(d, dtype=np.complex128)
    drift = problem.drift_hamiltonian.full()
    n_blocks = 1 + sum(len(noise.decay_rates) + 1 for noise in noises)
    nominal_block = n_blocks - 1
    drift_blocks: list[np.ndarray] = []
    error_entries: list[tuple[int, int, np.ndarray]] = []
    output_blocks: list[int] = []
    cursor = 0

    for noise in noises:
        output_block = cursor
        output_blocks.append(output_block)
        drift_blocks.append(drift)
        drift_blocks.extend(drift - 1j * rate * identity for rate in noise.decay_rates)

        error = noise.scaled_operator.full()
        for term, weight in enumerate(noise.correlation_weights):
            intermediate_block = output_block + term + 1
            error_entries.append((output_block, intermediate_block, weight * error))
            error_entries.append((intermediate_block, nominal_block, error))
        cursor += len(noise.decay_rates) + 1

    drift_blocks.append(drift)
    return (
        BlockOperator.block_diagonal(drift_blocks),
        BlockOperator.from_entries(n_blocks, error_entries),
        tuple(output_blocks),
        nominal_block,
    )


def _build_taylor_augmented_hamiltonian(
    problem: ControlProblem,
    noise: TaylorNoise,
    robustness: RobustnessModel,
) -> AugmentedHamiltonian:
    """Build the full Taylor-augmented Hamiltonian in the physical basis."""

    indices = generate_multi_indices(noise.n_sources, noise.order)
    positions = {index: position for position, index in enumerate(indices)}
    n_blocks = len(indices)
    drift = BlockOperator.physical_diagonal(n_blocks, problem.drift_hamiltonian.full())
    controls = [
        BlockOperator.physical_diagonal(n_blocks, hamiltonian.full())
        for hamiltonian in problem.control_hamiltonians
    ]

    static_entries: list[tuple[int, int, np.ndarray]] = []
    control_entries: dict[int, list[tuple[int, int, np.ndarray]]] = {
        control_drift.channel: [] for control_drift in noise.control_drifts
    }
    n_static = len(noise.static_operators)
    for higher_index in indices:
        row = positions[higher_index]
        for source in range(noise.n_sources):
            lower_index = decrement(higher_index, source)
            if lower_index is None:
                continue
            column = positions[lower_index]
            if source < n_static:
                static_entries.append((row, column, noise.static_operators[source].full()))
                continue

            control_drift = noise.control_drifts[source - n_static]
            if control_drift.channel >= problem.n_controls:
                raise ValueError(
                    f"Control-drift channel {control_drift.channel} exceeds the problem's "
                    f"{problem.n_controls} control channels."
                )
            control_entries[control_drift.channel].append(
                (
                    row,
                    column,
                    control_drift.scale
                    * problem.control_hamiltonians[control_drift.channel].full(),
                )
            )

    if static_entries:
        drift = drift.add(BlockOperator.from_entries(n_blocks, static_entries))
    for channel, entries in control_entries.items():
        if entries:
            controls[channel] = controls[channel].add(BlockOperator.from_entries(n_blocks, entries))

    penalty_blocks, penalty_weights, rms_penalties = taylor_penalty_terms(
        noise,
        indices,
        robustness,
    )
    return AugmentedHamiltonian(
        drift=drift,
        controls=tuple(controls),
        nominal_block=n_blocks - 1,
        penalty_blocks=penalty_blocks,
        penalty_weights=penalty_weights,
        rms_penalties=rms_penalties,
    )


def build_augmented_hamiltonians(
    problem: ControlProblem,
    robustness: RobustnessModel | None,
) -> tuple[AugmentedHamiltonian, ...]:
    """Build all independently propagated augmented Hamiltonians.

    Taylor uncertainties share one multivariable augmented system. Colored
    sources share a second multi-source Van Loan system so their output blocks
    can enter one joint RMS penalty. With no robustness model, a one-block
    nominal Hamiltonian is returned.
    """

    components: list[AugmentedHamiltonian] = []
    if robustness is not None and robustness.taylor is not None:
        components.append(
            _build_taylor_augmented_hamiltonian(problem, robustness.taylor, robustness)
        )

    if robustness is not None and robustness.colored:
        colored = robustness.colored
        drift, errors, output_blocks, nominal_block = build_joint_colored_generators(
            problem,
            colored,
        )
        n_blocks = drift.n_blocks
        penalty_blocks, penalty_weights, rms_penalties = joint_colored_penalty_terms(
            colored,
            output_blocks,
            robustness,
        )
        components.append(
            AugmentedHamiltonian(
                drift=drift.add(errors),
                controls=tuple(
                    BlockOperator.physical_diagonal(n_blocks, hamiltonian.full())
                    for hamiltonian in problem.control_hamiltonians
                ),
                nominal_block=nominal_block,
                penalty_blocks=penalty_blocks,
                penalty_weights=penalty_weights,
                rms_penalties=rms_penalties,
            )
        )

    if not components:
        components.append(
            AugmentedHamiltonian(
                drift=BlockOperator.physical_diagonal(1, problem.drift_hamiltonian.full()),
                controls=tuple(
                    BlockOperator.physical_diagonal(1, hamiltonian.full())
                    for hamiltonian in problem.control_hamiltonians
                ),
                nominal_block=0,
                penalty_blocks=(),
                penalty_weights=(),
            )
        )
    return tuple(components)
