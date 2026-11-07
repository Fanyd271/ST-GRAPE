import numpy as np
import pytest
import qutip as qt

from stgrape.problem import ControlDrift, ControlProblem, ControlTask, RobustnessModel, TaylorNoise
from stgrape.sampling import NoiseRealization, evaluate_realization, sample_fidelities
from stgrape.systems import build_spin_chain


def test_sampling_does_not_modify_control_hamiltonians() -> None:
    chain = build_spin_chain(2)
    initial = qt.basis(4, 0)
    initial.dims = [[2, 2], [1, 1]]
    problem = ControlProblem(
        chain.drift_hamiltonian,
        chain.control_hamiltonians,
        initial,
        qt.ghz_state(2),
        duration=0.0005,
        time_step=0.00025,
        task=ControlTask.STATE_PREPARATION,
    )
    robustness = RobustnessModel(
        taylor=TaylorNoise(control_drifts=(ControlDrift(0, 0.005),), order=1)
    )
    before = [operator.full().copy() for operator in problem.control_hamiltonians]
    realization = NoiseRealization(
        static_amplitudes=np.empty(0),
        control_amplitudes=np.array([1.2]),
        colored_trajectories=np.empty((0, problem.n_time_steps)),
    )
    controls = np.ones((problem.n_controls, problem.n_time_steps))
    first = evaluate_realization(realization, problem, controls, robustness)
    second = evaluate_realization(realization, problem, controls, robustness)

    assert first == second
    for original, hamiltonian in zip(before, problem.control_hamiltonians, strict=True):
        np.testing.assert_array_equal(original, hamiltonian.full())


@pytest.mark.parametrize("workers", (1, 2))
def test_sampling_reports_completed_trajectories(workers: int) -> None:
    """Serial and process-pool sampling report parent-side completion counts."""

    chain = build_spin_chain(2)
    initial = qt.basis(4, 0)
    initial.dims = [[2, 2], [1, 1]]
    problem = ControlProblem(
        chain.drift_hamiltonian,
        chain.control_hamiltonians,
        initial,
        qt.ghz_state(2),
        duration=0.0005,
        time_step=0.00025,
        task=ControlTask.STATE_PREPARATION,
    )
    progress: list[tuple[int, int]] = []
    values = sample_fidelities(
        problem,
        np.zeros((problem.n_controls, problem.n_time_steps)),
        None,
        n_samples=3,
        seed=42,
        workers=workers,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert values.shape == (3,)
    assert progress == [(1, 3), (2, 3), (3, 3)]
