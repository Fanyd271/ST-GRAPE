"""Contract and smoke checks for the revised GHZ/QFT experiment suite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stgrape.augmented.builders import build_augmented_hamiltonians
from stgrape.benchmarks.common import (
    ExperimentTask,
    build_multisource_problem,
    pauli_catalog_labels,
    pauli_operator_catalog,
    quasistatic_operator_catalog,
    select_pending_methods,
    should_run_raw_result,
)
from stgrape.benchmarks.nominal import default_output_directory
from stgrape.benchmarks.robustness import (
    parse_arguments as parse_exp01,
)
from stgrape.benchmarks.scaling_blocks import (
    DEFAULT_DENSE_MAX_BLOCKS,
    DEFAULT_SOURCE_COUNTS,
)
from stgrape.benchmarks.scaling_blocks import (
    parse_arguments as parse_exp02a,
)
from stgrape.benchmarks.scaling_blocks import run_experiment as run_exp02a
from stgrape.benchmarks.scaling_dimension import (
    parse_arguments as parse_exp02b,
)
from stgrape.benchmarks.splitting import (
    parse_arguments as parse_exp02c,
)
from stgrape.benchmarks.splitting import run_experiment as run_exp02c
from stgrape.benchmarks.stress import (
    DEFAULT_GHZ_QUBITS,
    _coupling_link_indices,
    _disjoint_coupling_link_indices,
    _load_file_pulse,
    _time_step,
)
from stgrape.benchmarks.stress import (
    parse_arguments as parse_exp03,
)
from stgrape.benchmarks.stress import run_experiment as run_exp03
from stgrape.benchmarks.stress_sampling import backfill_monte_carlo
from stgrape.benchmarks.stress_sampling import parse_arguments as parse_exp03_sampling
from stgrape.diagnostics import evaluate_taylor_sensitivity_diagnostics
from stgrape.execution import PAPER_PULSE_CONFIG
from stgrape.models import build_paper_problem, build_static_coupling_robustness
from stgrape.systems.spin_chain import build_spin_chain

REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK_STEMS = (
    "nominal",
    "robustness",
    "exp02_scaling",
    "splitting",
    "stress",
)
INLINE_RETINA_MAGIC = "%config InlineBackend.figure_format = 'retina'"


def test_revised_nominal_seed_default_is_classified() -> None:
    output = default_output_directory()

    assert output.parent == Path("outputs/experiments/nominal")
    assert output.name.isascii()
    assert len(output.name) == len("20260811-123456-123456")


def test_revised_experiment_defaults_encode_the_reviewed_protocol() -> None:
    """Defaults keep B1 on fixed r=2 and stress tests on r=1."""

    exp01 = parse_exp01(("--initial-source", "random"))
    exp02a = parse_exp02a(())
    exp02b = parse_exp02b(())
    exp02c = parse_exp02c(())
    exp03 = parse_exp03(("--initial-source", "random"))

    assert tuple(exp01.orders) == (1, 2)
    assert exp01.existing_result == "skip"
    assert exp01.colored_noise_sampling == "reference-gaussian"
    assert exp01.static_std_mhz == pytest.approx(1.0)
    assert exp01.frequency_drift_std_mhz == pytest.approx(1.0)
    assert exp01.coupling == "XX+YY"
    assert exp01.control_drift_std == pytest.approx(0.01)
    assert exp01.colored_std_mhz == pytest.approx(1.0)
    assert exp01.penalty_mode == "hybrid"
    assert exp01.p1_coefficient == pytest.approx(1.0)
    assert exp01.duration is None
    assert exp01.p2_coefficient == pytest.approx(1.0)
    assert exp01.pc_coefficient == pytest.approx(1.0)
    assert exp01.splitting_relative_tolerance == pytest.approx(0.15)
    assert exp01.splitting_absolute_tolerance == pytest.approx(2e-5)
    assert exp02a.order == 2
    assert exp02a.existing_result == "skip"
    assert tuple(exp02a.source_counts) == DEFAULT_SOURCE_COUNTS
    assert exp02a.dense_max_blocks == DEFAULT_DENSE_MAX_BLOCKS == 190
    assert exp02b.splitting_steps == 1
    assert exp02b.existing_result == "skip"
    assert tuple(exp02c.splitting_steps) == (1, 2, 4)
    assert exp02c.existing_result == "skip"
    assert exp03.order == 1
    assert tuple(exp03.method) == ("all",)
    assert exp03.existing_result == "skip"
    assert exp03.duration is None
    assert exp03.qubits is None
    assert DEFAULT_GHZ_QUBITS == (9,)
    assert exp03.coupling == "XX+YY"
    assert exp03.static_coupling_pairs is None
    assert exp03.static_std_mhz == pytest.approx(1.0)
    assert exp03.p1_coefficient == pytest.approx(0.5)
    assert exp03.st_true_loss_mode == "final-only"
    assert exp03.samples == 0
    assert exp03.sampling_workers == 1


def test_stress_test_records_post_run_st_reference_and_baseline_components(tmp_path) -> None:
    """Final-only ST validation stays outside the timed optimization."""

    output = tmp_path / "stress"
    run_exp03(
        parse_exp03(
            (
                "--output",
                str(output),
                "--initial-source",
                "random",
                "--duration",
                "0.03",
                "--static-coupling-pairs",
                "1-2",
                "--static-std-mhz",
                "2.0",
                "--p1-coefficient",
                "0.05",
                "--splitting-steps",
                "2",
                "--smoke",
            )
        )
    )
    case = output / "2q" / "seed0"
    metadata = json.loads((case / "ST" / "metadata.json").read_text())
    assert metadata["taylor_sources"] == 1
    assert metadata["static_coupling_pairs"] == [[1, 2]]
    assert metadata["static_standard_deviation_mhz"] == pytest.approx(2.0)
    assert metadata["p1_coefficient"] == pytest.approx(0.05)
    assert metadata["optimization"]["trotter_splitting"]["initial_steps"] == 2
    assert metadata["optimization"]["trotter_splitting"]["maximum_steps"] == 2
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["configuration"]["taylor_penalty_weight"] == pytest.approx(0.1)
    st_summary = json.loads((case / "ST" / "method_summary.json").read_text())
    assert st_summary["initial_true_loss"] is None
    assert st_summary["post_run_true_loss_wall_time_s"] > 0.0
    assert st_summary["final_nominal_fidelity"] is not None
    assert st_summary["true_loss"] == pytest.approx(
        1.0 - st_summary["final_nominal_fidelity"] + st_summary["final_robustness_penalty"]
    )
    chain, problem = build_paper_problem(
        2, "state", time_step=0.0025, duration=0.03, coupling="XX+YY"
    )
    robustness = build_static_coupling_robustness(
        chain, order=1, standard_deviation_mhz=2.0, coupling_link_indices=(0,)
    )
    with np.load(case / "ST" / "result.npz", allow_pickle=False) as result:
        final_controls = result["control_amplitudes"]
    sensitivity = evaluate_taylor_sensitivity_diagnostics(problem, robustness, final_controls)
    assert sensitivity is not None
    assert st_summary["final_robustness_penalty"] == pytest.approx(
        0.05 * sensitivity.p1_squared, rel=1e-5
    )
    with np.load(case / "ST" / "result.npz") as result:
        assert result["exact_objective_history"].size == 1
        assert result["exact_evaluation_times"][0] > st_summary["wall_time_s"]
        assert result["fidelity_loss_history"].shape == result["objective_history"].shape
        assert result["robustness_penalty_history"].shape == result["objective_history"].shape
        np.testing.assert_allclose(
            result["objective_history"],
            result["fidelity_loss_history"] + result["robustness_penalty_history"],
        )

    for method in ("dense", "block-sparse"):
        with np.load(case / method / "result.npz") as result:
            assert result["fidelity_loss_history"].shape == result["objective_history"].shape
            np.testing.assert_allclose(
                result["nominal_fidelity_history"],
                1.0 - result["fidelity_loss_history"],
            )
            assert result["robustness_penalty_history"].shape == result["objective_history"].shape
            np.testing.assert_allclose(
                result["objective_history"],
                result["fidelity_loss_history"] + result["robustness_penalty_history"],
            )


def test_stress_test_monte_carlo_backfill_preserves_optimization(
    tmp_path,
    capsys,
) -> None:
    """Existing controls can be sampled without rerunning or replacing optimization."""

    output = tmp_path / "stress"
    run_exp03(
        parse_exp03(
            (
                "--output",
                str(output),
                "--initial-source",
                "random",
                "--duration",
                "0.03",
                "--static-coupling-pairs",
                "1-2",
                "--smoke",
            )
        )
    )
    result_paths = {
        method: output / "2q" / "seed0" / method / "result.npz"
        for method in ("ST", "dense", "block-sparse")
    }
    original_results = {method: path.read_bytes() for method, path in result_paths.items()}

    validation_root = backfill_monte_carlo(
        parse_exp03_sampling(
            (
                "--result-root",
                str(output),
                "--samples",
                "4",
                "--sampling-workers",
                "1",
                "--progress-interval",
                "2",
                "--existing-result",
                "overwrite",
            )
        )
    )
    output_text = capsys.readouterr().out
    assert "[initial] 4/4 (100.0%)" in output_text
    assert "[ST] 4/4 (100.0%)" in output_text
    assert "[block-sparse] 4/4 (100.0%)" in output_text
    assert validation_root.parent == output / "monte_carlo"
    assert validation_root.name.startswith("samples-4_seed-42_")

    for method, path in result_paths.items():
        assert path.read_bytes() == original_results[method]
        fidelity_path = validation_root / "2q" / "seed0" / method / "fidelities.npz"
        with np.load(fidelity_path, allow_pickle=False) as saved:
            assert saved["values"].shape == (4,)
    with np.load(validation_root / "2q" / "seed0" / "initial_fidelities.npz") as saved:
        assert saved["values"].shape == (4,)
    validation = json.loads((validation_root / "monte_carlo_validation.json").read_text())
    assert validation["samples_per_control"] == 4
    assert validation["sampling_workers"] == 1
    assert validation["optimization_rerun"] is False

    larger_root = backfill_monte_carlo(
        parse_exp03_sampling(
            (
                "--result-root",
                str(output),
                "--samples",
                "5",
                "--sampling-workers",
                "1",
                "--existing-result",
                "skip",
            )
        )
    )
    assert larger_root != validation_root
    assert larger_root.name.startswith("samples-5_seed-42_")
    assert (validation_root / "2q" / "seed0" / "ST" / "fidelities.npz").is_file()
    with np.load(larger_root / "2q" / "seed0" / "ST" / "fidelities.npz") as saved:
        assert saved["values"].shape == (5,)
    index = json.loads((output / "monte_carlo" / "index.json").read_text())
    assert {entry["directory"] for entry in index["runs"].values()} == {
        validation_root.name,
        larger_root.name,
    }


def test_stress_test_selects_only_disjoint_coupling_fluctuations() -> None:
    """The nominal chain retains every link, but only alternate links fluctuate."""

    assert _disjoint_coupling_link_indices(9) == (0, 2, 4, 6)
    assert _coupling_link_indices(9, None) == (0, 2, 4, 6)
    assert _coupling_link_indices(9, ((2, 3), (8, 9))) == (1, 7)
    with pytest.raises(ValueError, match="distinct links"):
        _coupling_link_indices(9, ((1, 2), (1, 2)))
    with pytest.raises(ValueError, match="within the chain"):
        _coupling_link_indices(9, ((9, 10),))
    chosen = parse_exp03(("--static-coupling-pairs", "2-3", "8-9"))
    assert chosen.static_coupling_pairs == [(2, 3), (8, 9)]
    with pytest.raises(SystemExit):
        parse_exp03(("--static-coupling-pairs", "2-4"))
    assert _time_step(parse_exp03(("--initial-pulse", "result.npz"))) == pytest.approx(0.0005)
    chain = build_spin_chain(4, coupling="XX+YY")
    noise = build_static_coupling_robustness(
        chain, order=1, coupling_link_indices=_disjoint_coupling_link_indices(4)
    ).taylor
    assert noise is not None
    assert len(chain.coupling_operators) == 3
    assert noise.n_sources == 2
    for actual, index in zip(noise.static_operators, (0, 2), strict=True):
        np.testing.assert_allclose(
            actual.full(), 2.0 * np.pi * chain.coupling_operators[index].full()
        )


def test_stress_test_file_pulse_checks_model_grid_and_waveform(tmp_path) -> None:
    """A direct optimized pulse must reconstruct on the exact stress-test model."""

    _, problem = build_paper_problem(2, "state", time_step=0.0025, duration=0.03, coupling="XX+YY")
    pulse = PAPER_PULSE_CONFIG.random_pulse(problem, seed=4)
    path = tmp_path / "result.npz"
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "task": "state",
        "n_qubits": 2,
        "coupling": "XX+YY",
        "duration_us": problem.duration,
        "time_step_us": problem.time_step,
        "n_time_steps": problem.n_time_steps,
        "pulse": {"control_basis": "pwc"},
    }
    np.savez(path, coefficients=pulse.coefficients, control_amplitudes=pulse.amplitudes())
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded = _load_file_pulse(path, problem, task=ExperimentTask.GHZ, n_qubits=2, coupling="XX+YY")
    np.testing.assert_allclose(loaded.amplitudes(), pulse.amplitudes())

    metadata["time_step_us"] = 0.0005
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="time_step_us"):
        _load_file_pulse(path, problem, task=ExperimentTask.GHZ, n_qubits=2, coupling="XX+YY")

    metadata["time_step_us"] = problem.time_step
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(
        path,
        coefficients=pulse.coefficients,
        control_amplitudes=pulse.amplitudes() + 1.0,
    )
    with pytest.raises(ValueError, match="waveform"):
        _load_file_pulse(path, problem, task=ExperimentTask.GHZ, n_qubits=2, coupling="XX+YY")

    np.savez(path, coefficients=pulse.coefficients, control_amplitudes=pulse.amplitudes())
    output = tmp_path / "file_input_run"
    run_exp03(
        parse_exp03(
            (
                "--output",
                str(output),
                "--initial-pulse",
                str(path),
                "--qubits",
                "2",
                "--duration",
                "0.03",
                "--time-step",
                "0.0025",
                "--max-iterations",
                "1",
                "--method",
                "ST",
                "--samples",
                "0",
            )
        )
    )
    result_metadata = json.loads((output / "2q" / "seed0" / "ST" / "metadata.json").read_text())
    assert result_metadata["initial_source"] == "file"
    assert result_metadata["initial_seed_path"] == str(path)
    assert result_metadata["initial_pulse_path"] == str(path.resolve())


def test_quasistatic_catalog_matches_the_spin_chain_coupling() -> None:
    """Pair-noise directions follow ZZ or XX+YY without changing catalog size."""

    zz_chain = build_spin_chain(4, coupling="ZZ")
    exchange_chain = build_spin_chain(4, coupling="XX+YY")
    zz_catalog = quasistatic_operator_catalog(zz_chain)
    exchange_catalog = quasistatic_operator_catalog(exchange_chain)

    assert len(zz_catalog) == len(exchange_catalog) == 18
    for catalog, chain in (
        (zz_catalog, zz_chain),
        (exchange_catalog, exchange_chain),
    ):
        for actual, expected in zip(
            catalog[:3],
            chain.coupling_operators,
            strict=True,
        ):
            np.testing.assert_allclose(actual.full(), expected.full())
    assert not np.allclose(zz_catalog[0].full(), exchange_catalog[0].full())


def test_pauli_catalog_is_seeded_complete_and_extends_beyond_local_sources() -> None:
    """The synthetic catalog is a nested permutation of every nontrivial Pauli string."""

    labels = pauli_catalog_labels(4, seed=42)
    repeated = pauli_catalog_labels(4, seed=42)
    alternative = pauli_catalog_labels(4, seed=43)
    operators = pauli_operator_catalog(build_spin_chain(4, coupling="XX+YY"), seed=42)

    assert labels == repeated
    assert labels != alternative
    assert len(labels) == len(set(labels)) == len(operators) == 4**4 - 1
    assert "IIII" not in labels
    assert parse_exp02a(("--source-family", "pauli", "--source-counts", "20")).source_counts == [20]
    assert parse_exp02c(("--source-family", "pauli")).source_family == "pauli"


def test_revised_experiments_accept_explicit_overwrite() -> None:
    """Every revised entry point exposes the same result policy."""

    parsers = (parse_exp01, parse_exp02a, parse_exp02b, parse_exp02c, parse_exp03)
    for parser in parsers:
        assert parser(("--existing-result", "overwrite")).existing_result == "overwrite"


def test_multisource_builder_supports_both_physical_tasks() -> None:
    """GHZ and QFT share a scalable, explicitly located noise model."""

    for task in ExperimentTask:
        chain, problem, robustness, locations = build_multisource_problem(
            task,
            4,
            time_step=0.0025,
            order=2,
            static_standard_deviation_mhz=1.5,
            colored_standard_deviation_mhz=2.5,
            static_coupling_pairs=((1, 2), (2, 3)),
            control_drift_qubits=(2, 3),
            non_markovian_qubit=4,
        )
        assert robustness.taylor is not None
        # Two coupling-strength offsets, four local-frequency offsets, and
        # X/Y control drifts on two selected qubits.
        assert robustness.taylor.n_sources == 10
        components = build_augmented_hamiltonians(problem, robustness)
        assert tuple(component.n_blocks for component in components) == (
            math.comb(10 + 2, 2),
            7,
        )
        assert locations.static_coupling_pairs == ((1, 2), (2, 3))
        assert locations.frequency_drift_qubits == (1, 2, 3, 4)
        assert locations.control_drift_qubits == (2, 3)
        assert locations.non_markovian_qubit == 4
        source = robustness.colored[0]
        assert source.reference_correlation is not None
        variance = source.reference_correlation.covariance(np.asarray([0.0]))[0]
        assert source.strength * np.sqrt(variance) == pytest.approx(2.0 * np.pi * 2.5)
        assert robustness.taylor is not None
        expected_static = 2.0 * np.pi * 1.5 * chain.coupling_operators[0]
        assert (robustness.taylor.static_operators[0] - expected_static).norm() < 1e-12


def test_multisource_parser_accepts_explicit_one_based_noise_locations() -> None:
    """Every physical noise family exposes its qubit location at the CLI."""

    arguments = parse_exp01(
        (
            "--initial-source",
            "random",
            "--static-coupling-pairs",
            "1-2",
            "3-4",
            "--control-drift-qubits",
            "2",
            "4",
            "--frequency-drift-qubits",
            "1",
            "3",
            "--non-markovian-qubit",
            "3",
            "--static-std-mhz",
            "1.25",
            "--frequency-drift-std-mhz",
            "0.8",
            "--control-drift-std",
            "0.02",
            "--colored-std-mhz",
            "0.75",
            "--duration",
            "0.05",
        )
    )
    assert arguments.static_coupling_pairs == [(1, 2), (3, 4)]
    assert arguments.frequency_drift_qubits == [1, 3]
    assert arguments.control_drift_qubits == [2, 4]
    assert arguments.non_markovian_qubit == 3
    assert arguments.static_std_mhz == pytest.approx(1.25)
    assert arguments.frequency_drift_std_mhz == pytest.approx(0.8)
    assert arguments.control_drift_std == pytest.approx(0.02)
    assert arguments.colored_std_mhz == pytest.approx(0.75)
    assert arguments.duration == pytest.approx(0.05)


def test_multisource_builder_accepts_custom_duration() -> None:
    """The public multi-source builder forwards T to the control problem."""

    _, problem, _, _ = build_multisource_problem(
        ExperimentTask.GHZ,
        2,
        duration=0.05,
        time_step=0.0025,
        order=1,
    )

    assert problem.duration == pytest.approx(0.05)
    assert problem.n_time_steps == 20


def test_multisource_builder_forwards_independent_rms_coefficients() -> None:
    """The public experiment builder preserves all three direct RMS weights."""

    _, _, robustness, _ = build_multisource_problem(
        ExperimentTask.GHZ,
        2,
        time_step=0.0025,
        order=2,
        penalty_mode="rms",
        p1_coefficient=0.2,
        p2_coefficient=0.7,
        pc_coefficient=1.3,
    )

    assert robustness.resolved_p1_coefficient == pytest.approx(0.2)
    assert robustness.resolved_p2_coefficient == pytest.approx(0.7)
    assert robustness.resolved_pc_coefficient == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("include_static", "include_control", "include_colored", "expected_components"),
    (
        (False, True, True, 2),
        (True, False, True, 2),
        (True, True, False, 1),
        (True, False, False, 1),
        (False, True, False, 1),
    ),
)
def test_multisource_builder_supports_optional_noise_families(
    include_static: bool,
    include_control: bool,
    include_colored: bool,
    expected_components: int,
) -> None:
    """Every non-empty physical family subset produces a valid model."""

    active = np.asarray(
        [include_static, include_static, include_control, include_colored],
        dtype=float,
    )
    active /= np.sum(active)
    _, problem, robustness, locations = build_multisource_problem(
        ExperimentTask.GHZ,
        3,
        time_step=0.0025,
        order=2,
        coupling_weight=float(active[0]),
        frequency_drift_weight=float(active[1]),
        control_drift_weight=float(active[2]),
        colored_weight=float(active[3]),
        include_static_noise=include_static,
        include_frequency_drift=include_static,
        include_control_drift=include_control,
        include_non_markovian_noise=include_colored,
    )
    assert bool(locations.static_coupling_pairs) is include_static
    assert bool(locations.control_drift_qubits) is include_control
    assert (locations.non_markovian_qubit is not None) is include_colored
    assert len(build_augmented_hamiltonians(problem, robustness)) == expected_components


def test_block_scaling_dense_cap_preserves_longer_sparse_ranges(tmp_path: Path) -> None:
    """Dense stops at its block cap without truncating ST or block-sparse."""

    output = tmp_path / "b1_ragged"
    arguments = parse_exp02a(
        (
            "--smoke",
            "--method",
            "all",
            "--dense-max-blocks",
            "3",
            "--output",
            str(output),
        )
    )

    assert run_exp02a(arguments) == output
    for method in ("ST", "block-sparse"):
        with np.load(output / method / "raw_data.npz") as saved:
            np.testing.assert_array_equal(saved["source_counts"], (1, 2))
            np.testing.assert_array_equal(saved["block_counts"], (3, 6))
            assert saved["objective_gradient_times_s"].shape == (2, 1, 1)
            assert int(saved["dense_max_blocks"]) == 3
    with np.load(output / "dense" / "raw_data.npz") as saved:
        np.testing.assert_array_equal(saved["source_counts"], (1,))
        np.testing.assert_array_equal(saved["block_counts"], (3,))
        assert saved["objective_gradient_times_s"].shape == (1, 1, 1)


def test_splitting_ablation_smoke_saves_cost_and_accuracy(tmp_path: Path) -> None:
    """B3 remains runnable and records the fixed-q accuracy diagnostics."""

    output = tmp_path / "b3"
    arguments = parse_exp02c(("--smoke", "--output", str(output)))
    assert run_exp02c(arguments) == output
    with np.load(output / "raw_data.npz") as saved:
        np.testing.assert_array_equal(saved["splitting_steps"], (1, 2))
        assert saved["st_objective_gradient_times_s"].shape == (2, 1)
        assert saved["loss_absolute_errors"].shape == (2,)
        assert saved["gradient_relative_errors"].shape == (2,)
        assert np.all(np.isfinite(saved["loss_absolute_errors"]))
        assert np.all(np.isfinite(saved["gradient_relative_errors"]))
        assert saved["loss_absolute_errors"][1] < saved["loss_absolute_errors"][0]


def test_existing_result_helpers_reject_incomplete_skip_outputs(tmp_path: Path) -> None:
    """Skip is conservative, while overwrite explicitly schedules replacement."""

    incomplete = tmp_path / "ST"
    incomplete.mkdir()
    with np.testing.assert_raises(RuntimeError):
        select_pending_methods(
            tmp_path,
            ("ST",),
            existing_result="skip",
            required_files=("raw_data.npz",),
        )
    assert select_pending_methods(
        tmp_path,
        ("ST",),
        existing_result="overwrite",
        required_files=("raw_data.npz",),
    ) == ("ST",)

    raw_path = tmp_path / "raw_data.npz"
    np.savez(raw_path, value=1)
    assert not should_run_raw_result(raw_path, existing_result="skip")
    assert should_run_raw_result(raw_path, existing_result="overwrite")
