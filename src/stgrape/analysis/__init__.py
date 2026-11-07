"""Regenerate publication plots from immutable saved measurements."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

from stgrape.paths import data_root, ensure_output_path, project_root


def _render(name: str, output: str | Path, **settings: object) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    target = ensure_output_path(Path(output))
    target.mkdir(parents=True, exist_ok=True)
    return runpy.run_path(
        str(Path(__file__).with_name(f"_{name}_plot.py")),
        init_globals={
            "PROJECT_ROOT": project_root(),
            "DATA_ROOT": data_root(),
            "OUTPUT_DIR": target,
            **settings,
        },
    )


def plot_robustness(task: str, output: str | Path) -> None:
    """Render the GHZ (Fig. 2) or QFT (Fig. 5) six-panel benchmark."""
    if task not in ("ghz", "qft"):
        raise ValueError("task must be ghz or qft.")
    _render("robustness", output, TASK=task)


def plot_scaling(task: str, output: str | Path, *, supplementary: bool = False) -> None:
    """Render Fig. 3/6, optionally including the reply-only exponential action."""
    if task not in ("ghz", "qft"):
        raise ValueError("task must be ghz or qft.")
    _render("scaling", output, TASK=task, SUPPLEMENTARY=supplementary)


def plot_stress(output: str | Path) -> None:
    """Render Fig. 4 using exactly five seeds and 500 trajectories per control."""
    _render("stress", output, TASK="stress")


def plot_strength_scan(output: str | Path) -> None:
    """Render the reply-only paired uncertainty-strength scan."""
    _render("strength_scan", output)


def reproduce_paper(output: str | Path, *, supplementary: bool = False) -> None:
    """Regenerate all numerical figures and table summaries without simulation."""
    from stgrape.archive import summarize_paper, verify_archive

    verify_archive()
    target = ensure_output_path(Path(output))
    target.mkdir(parents=True, exist_ok=True)
    for task in ("ghz", "qft"):
        plot_robustness(task, target / task)
        plot_scaling(task, target / f"{task}_scaling")
        if supplementary:
            plot_scaling(task, target / f"{task}_exponential_action", supplementary=True)
    plot_stress(target / "stress")
    if supplementary:
        plot_strength_scan(target / "strength_scan")
    shutil.copy2(data_root() / "reference_figures/flow_chart.pdf", target / "fig01_flow_chart.pdf")
    summarize_paper(target)
    print(f"Paper figures and tables written to {target}")
