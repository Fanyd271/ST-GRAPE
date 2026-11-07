"""Contract tests for the maintained E00 nominal-seed workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stgrape.benchmarks.nominal import (
    DEFAULT_PLOT_FIRST_N,
    PAPER_NOMINAL_ERROR_BOUND,
    _promote_accepted_seed,
    _qubits_for_task,
    _save_nominal_control_overview,
    _select_pending_seed_cases,
    _selected_tasks,
    build_optimizer_options,
    generate_nominal_seed_case,
    nominal_seed_is_acceptable,
    parse_arguments,
    seeds_for_case,
    validate_nominal_error_bound,
)
from stgrape.execution import (
    generated_seed_coupling,
    generated_seed_duration,
    generated_seed_path,
    generated_seed_time_step,
    validate_generated_seed,
    validate_generated_seed_root,
)
from stgrape.problem import ControlTask


def test_e00_uses_the_stable_new_result_layout() -> None:
    root = Path("results/stgrape.benchmarks/nominal/run")
    path = generated_seed_path(
        root,
        task=ControlTask.STATE_PREPARATION,
        n_qubits=4,
        seed=3,
    )
    assert path == root / "state/4q/seed3/result.npz"


def test_generated_seed_validation_rejects_bad_nominal_loss(tmp_path: Path) -> None:
    seed_directory = tmp_path / "state/4q/seed3"
    seed_directory.mkdir(parents=True)
    path = seed_directory / "result.npz"
    np.savez(path, coefficients=np.zeros((8, 10)), objective=0.2)
    (seed_directory / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "E00 nominal seeds",
                "task": "state",
                "n_qubits": 4,
                "seed": 3,
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeding the accepted bound"):
        validate_generated_seed(
            path,
            task=ControlTask.STATE_PREPARATION,
            n_qubits=4,
            seed=3,
        )


def test_generated_seed_validation_checks_chebyshev_basis(tmp_path: Path) -> None:
    seed_directory = tmp_path / "gate/2q/seed0"
    seed_directory.mkdir(parents=True)
    path = seed_directory / "result.npz"
    np.savez(path, coefficients=np.zeros((4, 8)), objective=1e-4)
    (seed_directory / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "E00 nominal seeds",
                "task": "gate",
                "n_qubits": 2,
                "seed": 0,
                "schema_version": 1,
                "pulse": {
                    "control_basis": "chebyshev",
                    "chebyshev_basis_size": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    validate_generated_seed(
        path,
        task=ControlTask.GATE_SYNTHESIS,
        n_qubits=2,
        seed=0,
        control_basis="chebyshev",
        chebyshev_k=8,
    )
    with pytest.raises(ValueError, match="control_basis"):
        validate_generated_seed(
            path,
            task=ControlTask.GATE_SYNTHESIS,
            n_qubits=2,
            seed=0,
            control_basis="pwc",
        )


def test_generated_seed_root_metadata_is_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="E00 manifest"):
        validate_generated_seed_root(tmp_path)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"experiment": "E00 nominal seeds"}),
        encoding="utf-8",
    )
    validate_generated_seed_root(tmp_path)

    manifest.write_text(
        json.dumps(
            {
                "experiment": "E00 nominal seeds",
                "configuration": {
                    "time_step_us": 0.0005,
                    "coupling": "XX+YY",
                },
            }
        ),
        encoding="utf-8",
    )
    assert generated_seed_time_step(tmp_path) == pytest.approx(0.0005)
    assert generated_seed_coupling(tmp_path) == "XX+YY"

    seed_directory = tmp_path / "state/4q/seed3"
    seed_directory.mkdir(parents=True)
    (seed_directory / "metadata.json").write_text(
        json.dumps({"duration_us": 0.05}),
        encoding="utf-8",
    )
    assert generated_seed_duration(
        tmp_path,
        task=ControlTask.STATE_PREPARATION,
        n_qubits=4,
        seed=3,
    ) == pytest.approx(0.05)


def test_nominal_seed_matrix_and_optimizer_configuration() -> None:
    assert seeds_for_case(ControlTask.GATE_SYNTHESIS, 5, None) == tuple(range(5))
    assert seeds_for_case(ControlTask.STATE_PREPARATION, 7, None) == tuple(range(3))
    assert seeds_for_case(ControlTask.STATE_PREPARATION, 10, (0, 3)) == (0, 3)

    options = build_optimizer_options(max_iterations=123, ftol=1e-10, gtol=1e-8)
    assert options.max_iterations == 123
    assert options.scipy_options()["ftol"] == 1e-10
    assert options.scipy_options()["gtol"] == 1e-8


def test_nominal_seed_acceptance_uses_one_percent_error_bound() -> None:
    assert PAPER_NOMINAL_ERROR_BOUND == 0.01
    assert nominal_seed_is_acceptable(0.01, PAPER_NOMINAL_ERROR_BOUND)
    assert not nominal_seed_is_acceptable(0.0100001, PAPER_NOMINAL_ERROR_BOUND)
    assert not nominal_seed_is_acceptable(float("nan"), PAPER_NOMINAL_ERROR_BOUND)
    for invalid in (0.0, -0.1, 1.1, float("inf")):
        with pytest.raises(ValueError):
            validate_nominal_error_bound(invalid)


def test_nominal_seed_case_reinitializes_until_slots_are_filled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[int] = []

    def fake_candidate_batch(jobs, *, workers):
        assert workers == 1
        rows = []
        for job in jobs:
            attempted.append(job.initialization_seed)
            accepted = job.initialization_seed != 0
            candidate = (
                job.output_root
                / "attempts"
                / job.task
                / f"{job.n_qubits}q"
                / f"seed{job.seed}"
                / f"attempt{job.attempt}_rng{job.initialization_seed}"
            )
            candidate.mkdir(parents=True)
            np.savez(candidate / "result.npz", objective=1e-4 if accepted else 0.97)
            (candidate / "metadata.json").write_text("{}", encoding="utf-8")
            rows.append(
                {
                    "task": job.task,
                    "n_qubits": job.n_qubits,
                    "seed": job.seed,
                    "initialization_seed": job.initialization_seed,
                    "attempt": job.attempt,
                    "nominal_loss": 1e-4 if accepted else 0.97,
                    "nominal_fidelity": 0.9999 if accepted else 0.03,
                    "error_bound": job.error_bound,
                    "accepted": accepted,
                    "wall_time_s": 0.0,
                    "iterations": 1,
                    "success": True,
                    "result": str(candidate / "result.npz"),
                }
            )
        return rows

    monkeypatch.setattr(
        "stgrape.benchmarks.nominal._run_candidate_batch",
        fake_candidate_batch,
    )
    accepted, attempts = generate_nominal_seed_case(
        task=ControlTask.GATE_SYNTHESIS,
        n_qubits=5,
        output_seeds=(0, 1),
        output_root=tmp_path,
        time_step=0.00025,
        max_iterations=1,
        ftol=1e-10,
        gtol=1e-10,
        ode_method="dop853",
        ode_representation="dense",
        ode_rtol=None,
        ode_atol=None,
        time_limit_hours=None,
        log_interval=20,
        workers=1,
    )

    assert attempted == [0, 1, 2]
    assert [row["seed"] for row in accepted] == [0, 1]
    assert accepted[0]["initialization_seed"] == 2
    assert len(attempts) == 3
    assert (tmp_path / "gate/5q/seed0/result.npz").is_file()


def test_existing_seed_policy_skips_or_overwrites_atomically(tmp_path: Path) -> None:
    case = (ControlTask.GATE_SYNTHESIS, 2, 0)
    stable = tmp_path / "gate/2q/seed0"
    stable.mkdir(parents=True)
    np.savez(
        stable / "result.npz",
        objective=1e-4,
        objective_history=np.asarray([1e-3, 1e-4]),
    )
    (stable / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "E00 nominal seeds",
                "task": "gate",
                "n_qubits": 2,
                "seed": 0,
                "initialization_seed": 0,
                "attempt": 1,
                "nominal_error_bound": 0.01,
                "end_to_end_wall_time_s": 1.0,
                "success": True,
            }
        ),
        encoding="utf-8",
    )

    pending, skipped = _select_pending_seed_cases((case,), tmp_path, existing_seed="skip")
    assert pending == ()
    assert skipped == (case,)

    pending, skipped = _select_pending_seed_cases(
        (case,),
        tmp_path,
        existing_seed="overwrite",
    )
    assert pending == (case,)
    assert skipped == ()

    candidate = tmp_path / "attempts/gate/2q/seed0/attempt1_rng0"
    candidate.mkdir(parents=True)
    np.savez(candidate / "result.npz", objective=1e-5)
    (candidate / "metadata.json").write_text("{}", encoding="utf-8")
    _promote_accepted_seed(
        {
            "task": "gate",
            "n_qubits": 2,
            "seed": 0,
            "result": str(candidate / "result.npz"),
        },
        tmp_path,
        overwrite_existing=True,
    )
    with np.load(stable / "result.npz") as result:
        assert float(result["objective"]) == 1e-5
    assert not list(stable.parent.glob(".seed0.backup-*"))


def test_nominal_cli_defaults_to_skip_and_xx_yy() -> None:
    default = parse_arguments(())
    overwrite = parse_arguments(("--existing-seed", "overwrite"))

    assert default.existing_seed == "skip"
    assert default.coupling == "XX+YY"
    assert default.plot_first_n == DEFAULT_PLOT_FIRST_N == 4
    assert default.qubits is None
    assert overwrite.existing_seed == "overwrite"
    assert parse_arguments(("--coupling", "XX+YY")).coupling == "XX+YY"
    assert parse_arguments(("--plot-first-n", "7")).plot_first_n == 7
    assert parse_arguments(("--duration", "0.05")).duration == pytest.approx(0.05)


def test_e00_uses_the_shared_ghz_qft_task_names() -> None:
    """E00 exposes physical targets rather than low-level objective names."""

    ghz = parse_arguments(("--task", "ghz", "--qubits", "3", "4"))
    qft = parse_arguments(("--task", "qft", "--qubits", "2", "5"))
    both = parse_arguments(("--task", "both", "--qubits", "3", "4"))

    assert ghz.task == "ghz"
    assert tuple(ghz.qubits) == (3, 4)
    assert _selected_tasks(ghz.task) == (ControlTask.STATE_PREPARATION,)
    assert qft.task == "qft"
    assert tuple(qft.qubits) == (2, 5)
    assert _selected_tasks(qft.task) == (ControlTask.GATE_SYNTHESIS,)
    assert tuple(both.qubits) == (3, 4)
    assert _qubits_for_task(ControlTask.GATE_SYNTHESIS, None) == (2, 3, 4, 5)
    assert _qubits_for_task(ControlTask.STATE_PREPARATION, None) == tuple(range(2, 11))
    assert _qubits_for_task(ControlTask.GATE_SYNTHESIS, (3, 4)) == (3, 4)
    assert _qubits_for_task(ControlTask.STATE_PREPARATION, (3, 4)) == (3, 4)
    with pytest.raises(SystemExit):
        parse_arguments(("--task", "gate"))


def test_e00_saves_a_multi_panel_overview_of_the_first_controls(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for seed in range(3):
        seed_directory = tmp_path / f"gate/2q/seed{seed}"
        seed_directory.mkdir(parents=True)
        controls = np.arange(20, dtype=float).reshape(4, 5) + seed
        result_path = seed_directory / "result.npz"
        np.savez(result_path, control_amplitudes=controls)
        rows.append(
            {
                "task": "gate",
                "n_qubits": 2,
                "seed": seed,
                "nominal_fidelity": 0.999 - seed * 1e-4,
                "result": str(result_path),
            }
        )

    png_path, pdf_path = _save_nominal_control_overview(
        rows,
        tmp_path,
        task=ControlTask.GATE_SYNTHESIS,
        n_qubits=2,
        first_n=2,
        time_step=0.0005,
    )

    assert png_path == tmp_path / "figures/nominal_controls_gate_2q.png"
    assert pdf_path == tmp_path / "figures/nominal_controls_gate_2q.pdf"
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
