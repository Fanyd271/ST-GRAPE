"""Locally controlled spin-chain model used in the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from qutip import Qobj, destroy, qeye, sigmaz, tensor

CouplingModel = Literal["ZZ", "XX+YY"]


@dataclass(frozen=True)
class CommutingControlGroup:
    """Control channels that are diagonal in a shared physical basis."""

    channels: tuple[int, ...]
    basis_transform: Qobj
    diagonal_hamiltonians: tuple[Qobj, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("A commuting control group cannot be empty.")
        if len(self.channels) != len(self.diagonal_hamiltonians):
            raise ValueError("channels and diagonal_hamiltonians must have equal length.")


@dataclass(frozen=True)
class SpinChain:
    """Hamiltonians and control decomposition of a qubit spin chain."""

    drift_hamiltonian: Qobj
    control_hamiltonians: tuple[Qobj, ...]
    control_groups: tuple[CommutingControlGroup, ...]
    lowering_operators: tuple[Qobj, ...]
    z_operators: tuple[Qobj, ...]
    coupling: CouplingModel
    coupling_operators: tuple[Qobj, ...]


def normalize_coupling_model(coupling: str) -> CouplingModel:
    """Return the canonical public name for a supported chain coupling."""

    normalized = coupling.upper().replace(" ", "")
    if normalized == "XY":
        normalized = "XX+YY"
    if normalized not in {"ZZ", "XX+YY"}:
        raise ValueError("coupling must be 'ZZ' or 'XX+YY'.")
    return cast(CouplingModel, normalized)


def build_spin_chain(n_qubits: int, coupling: str = "ZZ") -> SpinChain:
    r"""Construct the paper's nearest-neighbour spin-chain model.

    Parameters
    ----------
    n_qubits
        Number of two-level systems.
    coupling
        ``"ZZ"`` for a 30 MHz Ising coupling or ``"XX+YY"`` for the
        exchange Hamiltonian
        :math:`60\,\mathrm{MHz}\,(a_i^\dagger a_{i+1}+\mathrm{h.c.})`,
        equivalently :math:`30\,\mathrm{MHz}\,(X_iX_{i+1}+Y_iY_{i+1})`.
    """

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two.")
    coupling_model = normalize_coupling_model(coupling)

    lowering: list[Qobj] = []
    z_operators: list[Qobj] = []
    for site in range(n_qubits):
        lowering.append(
            tensor(*(destroy(2) if index == site else qeye(2) for index in range(n_qubits)))
        )
        z_operators.append(
            tensor(*(sigmaz() if index == site else qeye(2) for index in range(n_qubits)))
        )

    x_controls = tuple(operator + operator.dag() for operator in lowering)
    y_controls = tuple(1j * (operator - operator.dag()) for operator in lowering)
    controls = x_controls + y_controls

    coupling_operators: list[Qobj] = []
    drift = 0 * z_operators[0]
    for site in range(n_qubits - 1):
        if coupling_model == "ZZ":
            interaction = z_operators[site] @ z_operators[site + 1]
            coupling_strength_mhz = 30.0
        else:
            # X_i X_{i+1} + Y_i Y_{i+1} equals twice the two-level
            # excitation-hopping operator.  Keeping the public interaction in
            # this Pauli form lets the same 30 MHz nominal coefficient and the
            # same MHz-valued coupling-drift parameter apply to both models.
            interaction = 2.0 * (
                lowering[site] @ lowering[site + 1].dag()
                + lowering[site].dag() @ lowering[site + 1]
            )
            coupling_strength_mhz = 30.0
        coupling_operators.append(interaction)
        drift += coupling_strength_mhz * 2.0 * np.pi * interaction

    hadamard = Qobj(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0), dims=[[2], [2]])
    y_to_z = Qobj(np.array([[1.0, 1.0], [-1j, 1j]]) / np.sqrt(2.0), dims=[[2], [2]])
    x_basis = tensor(*([hadamard] * n_qubits))
    y_basis = tensor(*([y_to_z] * n_qubits))
    z_tuple = tuple(z_operators)

    groups = (
        CommutingControlGroup(tuple(range(n_qubits)), x_basis, z_tuple),
        CommutingControlGroup(tuple(range(n_qubits, 2 * n_qubits)), y_basis, z_tuple),
    )

    return SpinChain(
        drift_hamiltonian=drift,
        control_hamiltonians=controls,
        control_groups=groups,
        lowering_operators=tuple(lowering),
        z_operators=z_tuple,
        coupling=coupling_model,
        coupling_operators=tuple(coupling_operators),
    )
