"""Release-only contracts: immutable data, supported presets, and optimized blocks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from stgrape.archive import verify_archive
from stgrape.augmented.block_sparse import BlockOperator
from stgrape.cli import EXPERIMENTS, main
from stgrape.execution import prepare_resumable_output
from stgrape.paths import data_root, ensure_output_path
from stgrape.workflow import configure_ode_propagation

ROOT = Path(__file__).parents[1]


def test_published_data_checksums_and_table_precision() -> None:
    report = verify_archive()
    assert report["verified_files"] == 974
    assert report["verified_table_values"] == 76
    assert report["stress_samples_per_control"] == 500


def test_numeric_archives_are_identical_to_the_extraction_source() -> None:
    manifest = json.loads((data_root() / "manifest.json").read_text())
    arrays = [r for r in manifest["files"] if r["path"].endswith(".npz")]
    assert arrays
    assert all(r["sha256"] == r["source_sha256"] for r in arrays)


def test_archive_cannot_be_used_as_experiment_output() -> None:
    with pytest.raises(ValueError, match="immutable"):
        ensure_output_path(data_root() / "paper/ghz")
    with pytest.raises(ValueError, match="immutable"):
        prepare_resumable_output(data_root() / "paper/ghz", experiment="x", configuration={})


def test_no_command_only_prints_help(capsys) -> None:
    main([])
    assert "reproduce" in capsys.readouterr().out


@pytest.mark.parametrize(
    "preset", sorted((ROOT / "experiments").glob("*.json")), ids=lambda p: p.stem
)
def test_published_presets_parse_without_launching_a_simulation(preset: Path) -> None:
    content = json.loads(preset.read_text())
    module = importlib.import_module("stgrape.benchmarks." + EXPERIMENTS[content["experiment"]])
    args = module.parse_arguments(content["arguments"])
    assert args.coupling == "XX+YY"
    if content["experiment"] == "stress":
        assert args.splitting_steps == 1
        assert args.p1_coefficient == 1.0
        assert args.st_true_loss_mode == "final-only"
    if content["experiment"] == "robustness":
        assert args.penalty_mode == "hybrid"
        assert args.samples == 10000
        assert len(args.seeds) == 10
        assert args.time_step == 0.0005
        assert args.seed_root.is_dir()


def test_retired_integrators_are_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="dop853"):
        configure_ode_propagation(method="adams")


def test_block_coalescing_does_not_alias_or_mutate_inputs() -> None:
    blocks = np.arange(3 * 4, dtype=float).reshape(3, 2, 2).astype(complex)
    before = blocks.copy()
    op = BlockOperator(2, 2, np.array([0, 0, 1]), np.array([0, 0, 1]), blocks)
    np.testing.assert_array_equal(blocks, before)
    np.testing.assert_array_equal(op.to_dense()[:2, :2], before[0] + before[1])
    op.blocks[0, 0, 0] = 999
    np.testing.assert_array_equal(blocks, before)


def test_optimized_composition_matches_dense_algebra() -> None:
    rng = np.random.default_rng(19)
    entries = [
        (i, j, rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        for i in range(4)
        for j in range(i + 1)
    ]
    left = BlockOperator.from_entries(4, entries)
    right = left.adjoint()
    first = left.to_dense().copy()
    second = right.to_dense().copy()
    np.testing.assert_allclose(left.compose(right).to_dense(), first @ second, atol=1e-12)
    np.testing.assert_array_equal(left.to_dense(), first)
    np.testing.assert_array_equal(right.to_dense(), second)


def test_demo_reports_losses_without_complementing_them(tmp_path) -> None:
    from stgrape.demo import run_demo

    report = run_demo(tmp_path / "demo", iterations=2)
    for method, values in report.items():
        with np.load(tmp_path / "demo" / method / "result.npz", allow_pickle=False) as saved:
            assert values["initial_loss"] == float(saved["initial_objective"])
            assert values["final_loss"] == float(saved["objective"])
        assert values["final_loss"] <= values["initial_loss"]


def test_experiment_help_forwards_to_the_supported_runner(capsys) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["experiment", "robustness", "--help"])
    assert stopped.value.code == 0
    assert "--p2-coefficient" in capsys.readouterr().out


def test_stress_resampling_rejects_reference_archive_before_writing() -> None:
    from stgrape.benchmarks.stress_sampling import backfill_monte_carlo, parse_arguments

    args = parse_arguments(["--result-root", str(data_root() / "paper/stress"), "--samples", "1"])
    with pytest.raises(ValueError, match="immutable"):
        backfill_monte_carlo(args)


def test_resuming_with_new_unrecorded_parameters_is_rejected(tmp_path) -> None:
    prepare_resumable_output(tmp_path / "run", experiment="test", configuration={"a": 1})
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_resumable_output(
            tmp_path / "run", experiment="test", configuration={"a": 1, "b": 2}
        )


def test_notebooks_are_clean_frontends_with_stable_cell_ids() -> None:
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        content = json.loads(path.read_text())
        assert all(cell.get("id") for cell in content["cells"])
        for cell in content["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert not cell["outputs"]
                assert "../outputs" not in "".join(cell["source"])
