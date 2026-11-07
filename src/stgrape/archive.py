"""Integrity checks and independently recomputed paper tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import data_root, ensure_output_path

METHODS = ("dense", "block-sparse", "ST")
METHOD_LABELS = {"Dense GRAPE": "dense", "Block-sparse GRAPE": "block-sparse", "ST-GRAPE": "ST"}
COLUMNS = (
    "loss",
    "mean_fidelity",
    "std_fidelity",
    "iterations",
    "wall_time_s",
    "time_per_iteration_s",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _values(path: Path, count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        values = np.asarray(saved["values"], dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected exactly {count} finite fidelities: {path}")
    if np.any(values < -1e-10) or np.any(values > 1 + 1e-10):
        raise ValueError(f"Fidelities are outside [0,1]: {path}")
    return values


def benchmark_table(task: str, archive: Path | None = None) -> list[dict]:
    """Recompute Table I/II, averaging per-seed ratios rather than their totals."""
    archive = data_root() if archive is None else archive
    n = 4 if task == "ghz" else 3
    root = archive / "paper" / task / f"{n}q"
    initial = [
        _values(root / "r1" / f"seed{seed}" / "initial_fidelities.npz", 10000) for seed in range(10)
    ]
    rows = [
        {
            "order": 0,
            "method": "Initial (nominal GRAPE)",
            "loss": None,
            "mean_fidelity": float(np.mean([np.mean(v) for v in initial])),
            "std_fidelity": float(np.mean([np.std(v) for v in initial])),
            "iterations": None,
            "wall_time_s": None,
            "time_per_iteration_s": None,
        }
    ]
    for order in (1, 2):
        for method in METHODS:
            values = []
            for seed in range(10):
                case = root / f"r{order}" / f"seed{seed}" / method
                summary = _read(case / "method_summary.json")
                f = _values(case / "fidelities.npz", 10000)
                with np.load(case / "result.npz", allow_pickle=False) as result:
                    iterations = len(result["objective_history"])
                if iterations < 1:
                    raise ValueError(f"Empty recorded optimization: {case}")
                wall = float(summary["wall_time_s"])
                values.append(
                    [
                        float(summary["true_loss"]),
                        float(np.mean(f)),
                        float(np.std(f)),
                        iterations,
                        wall,
                        wall / iterations,
                    ]
                )
            average = np.mean(values, axis=0)
            rows.append(
                {
                    "order": order,
                    "method": method,
                    **dict(zip(COLUMNS, map(float, average), strict=True)),
                }
            )
    return rows


def stress_summary(archive: Path | None = None) -> dict:
    """Summarize Fig. 4 using its five seeds and dedicated 500-sample archive."""
    archive = data_root() if archive is None else archive
    root = archive / "paper/stress"
    mc = root / "monte_carlo/samples-500_seed-42_dbe325b94dbf/9q"
    result = {}
    for method in ("initial", *METHODS):
        fidelities = []
        loss = []
        iterations = []
        times = []
        for seed in range(5):
            path = mc / f"seed{seed}"
            values = _values(
                path
                / ("initial_fidelities.npz" if method == "initial" else f"{method}/fidelities.npz"),
                500,
            )
            fidelities.append(float(np.mean(values)))
            if method != "initial":
                case = root / "9q" / f"seed{seed}" / method
                summary = _read(case / "method_summary.json")
                loss.append(float(summary["true_loss"]))
                times.append(float(summary["wall_time_s"]))
                with np.load(case / "result.npz", allow_pickle=False) as saved:
                    iterations.append(len(saved["objective_history"]))
        result[method] = {"mean_fidelity": float(np.mean(fidelities))}
        if method != "initial":
            result[method].update(
                loss=float(np.mean(loss)),
                iterations=float(np.mean(iterations)),
                wall_time_s=float(np.mean(times)),
                time_per_iteration_s=float(np.mean(np.asarray(times) / iterations)),
            )
    return result


def verify_archive(archive: Path | None = None) -> dict:
    """Check file hashes and agreement with the article's rounded table values."""
    archive = data_root() if archive is None else archive.resolve()
    manifest = _read(archive / "manifest.json")
    errors = []
    for record in manifest["files"]:
        path = (archive / record["path"]).resolve()
        if not path.is_relative_to(archive.resolve()) or not path.is_file():
            errors.append(f"Missing or invalid archive path: {record['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            errors.append(f"Checksum mismatch: {record['path']}")
    if errors:
        raise ValueError("\n".join(errors))
    expected = _read(archive / "expected_tables.json")
    compared = 0
    for task in ("ghz", "qft"):
        for observed, reference in zip(benchmark_table(task, archive), expected[task], strict=True):
            assert observed["order"] == reference["order"]
            for col, target, tolerance in zip(
                COLUMNS, reference["values"], reference["tolerances"], strict=True
            ):
                if target is not None:
                    if abs(observed[col] - target) > tolerance * 1.001:
                        errors.append(
                            f"{task} r={observed['order']} {observed['method']}: {col}={observed[col]} != printed {target} (rounding tolerance {tolerance})"
                        )
                    compared += 1
    stress = stress_summary(archive)
    for method, target in {
        "initial": 0.6095,
        "dense": 0.7822,
        "block-sparse": 0.8192,
        "ST": 0.9305,
    }.items():
        if abs(stress[method]["mean_fidelity"] - target) > 0.0000501:
            errors.append(f"Stress-test fidelity differs for {method}.")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "verified_files": len(manifest["files"]),
        "verified_table_values": compared,
        "stress_seeds": 5,
        "stress_samples_per_control": 500,
        "paper_doi": manifest["paper_doi"],
    }


def summarize_paper(output: Path) -> None:
    """Export numeric tables and per-seed provenance for external reuse."""
    target = ensure_output_path(output)
    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {task: benchmark_table(task) for task in ("ghz", "qft")}
    for task in ("ghz", "qft"):
        rows = benchmark_table(task)
        with (target / f"table_{task}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("order", "method", *COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    summary["stress"] = stress_summary()
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
