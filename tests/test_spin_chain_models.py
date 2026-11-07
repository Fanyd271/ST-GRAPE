"""Physical-model checks for the selectable spin-chain coupling."""

from __future__ import annotations

import numpy as np
import pytest

from stgrape.models import build_multisource_spin_chain_robustness
from stgrape.systems import build_spin_chain


def test_xx_yy_coupling_uses_the_two_level_exchange_hamiltonian() -> None:
    """The public XX+YY model is exactly the nearest-neighbour hopping model."""

    chain = build_spin_chain(2, coupling="XX+YY")
    x_left, x_right, y_left, y_right = chain.control_hamiltonians
    pauli_exchange = x_left @ x_right + y_left @ y_right
    hopping = (
        chain.lowering_operators[0] @ chain.lowering_operators[1].dag()
        + chain.lowering_operators[0].dag() @ chain.lowering_operators[1]
    )

    assert chain.coupling == "XX+YY"
    assert (chain.coupling_operators[0] - pauli_exchange).norm() < 1e-12
    assert (hopping - 0.5 * pauli_exchange).norm() < 1e-12
    assert (chain.drift_hamiltonian - 30.0 * 2.0 * np.pi * pauli_exchange).norm() < 1e-12


def test_zz_coupling_remains_the_default() -> None:
    """Existing experiments keep the original 30 MHz Ising drift by default."""

    chain = build_spin_chain(2)
    zz = chain.z_operators[0] @ chain.z_operators[1]

    assert chain.coupling == "ZZ"
    assert (chain.coupling_operators[0] - zz).norm() < 1e-12
    assert (chain.drift_hamiltonian - 30.0 * 2.0 * np.pi * zz).norm() < 1e-12


def test_coupling_and_local_frequency_drifts_follow_the_selected_model() -> None:
    """Quasistatic sources contain the selected link and requested local Z."""

    chain = build_spin_chain(2, coupling="XX+YY")
    robustness = build_multisource_spin_chain_robustness(
        chain,
        order=1,
        static_standard_deviation_mhz=1.5,
        frequency_drift_standard_deviation_mhz=1.0,
        frequency_drift_qubits=(1,),
        control_drift_qubits=(),
        colored_noise_qubit=None,
        coupling_weight=0.5,
        frequency_drift_weight=0.5,
        control_drift_weight=0.0,
        colored_weight=0.0,
    )

    assert robustness.taylor is not None
    coupling_drift, frequency_drift = robustness.taylor.static_operators
    assert (coupling_drift - 1.5 * 2.0 * np.pi * chain.coupling_operators[0]).norm() < 1e-12
    assert (frequency_drift - 1.0 * 2.0 * np.pi * chain.z_operators[1]).norm() < 1e-12


def test_invalid_coupling_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="ZZ.*XX\\+YY"):
        build_spin_chain(2, coupling="XX")


def test_disabled_noise_families_allow_zero_physical_strength() -> None:
    """A removed source may use zero strength without breaking model setup."""

    chain = build_spin_chain(2)
    robustness = build_multisource_spin_chain_robustness(
        chain,
        order=1,
        control_drift_standard_deviation=0.0,
        colored_standard_deviation_mhz=0.0,
        frequency_drift_qubits=(0, 1),
        control_drift_qubits=(),
        colored_noise_qubit=None,
        coupling_weight=0.5,
        frequency_drift_weight=0.5,
        control_drift_weight=0.0,
        colored_weight=0.0,
    )

    assert robustness.taylor is not None
    assert len(robustness.taylor.control_drifts) == 0
    assert not robustness.colored


def test_enabled_noise_family_rejects_zero_physical_strength() -> None:
    """An active source must retain a nonzero physical noise scale."""

    chain = build_spin_chain(2)
    with pytest.raises(ValueError, match="positive when its noise family is enabled"):
        build_multisource_spin_chain_robustness(
            chain,
            order=1,
            control_drift_standard_deviation=0.0,
            control_drift_qubits=(0,),
            colored_noise_qubit=None,
            coupling_weight=0.5,
            frequency_drift_weight=0.0,
            control_drift_weight=0.5,
            colored_weight=0.0,
        )
