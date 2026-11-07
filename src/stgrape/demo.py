"""Deterministic, inexpensive installation and numerical-workflow check."""

from __future__ import annotations

import json
from pathlib import Path

from .models import build_paper_problem, build_static_coupling_robustness
from .optimizers import LBFGSBOptions
from .paths import ensure_output_path
from .problem import ControlTask
from .pulse import BlackmanEnvelope, PulseConfig
from .workflow import OptimizationAlgorithm, OptimizationConfig, configure_ode_propagation, optimize


def run_demo(output: Path, *, iterations: int = 2) -> dict[str, dict[str, float]]:
    """Compare all three methods on the same small state-preparation problem."""
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    ensure_output_path(output)
    chain, problem = build_paper_problem(
        2, ControlTask.STATE_PREPARATION, time_step=0.0025, duration=0.01, coupling="XX+YY"
    )
    robustness = build_static_coupling_robustness(chain, order=1, standard_deviation_mhz=0.5)
    pulse = PulseConfig(filter_taps=3, envelope=BlackmanEnvelope(rise_time=0.0025)).random_pulse(
        problem, seed=7
    )
    summary = {}
    for method in ("ST", "dense", "block-sparse"):
        config = OptimizationConfig(
            algorithm=OptimizationAlgorithm.ST_GRAPE
            if method == "ST"
            else OptimizationAlgorithm.GRAPE,
            optimizer=LBFGSBOptions(max_iterations=iterations),
            propagation=configure_ode_propagation(
                representation="block-sparse" if method == "ST" else method
            ),
            exact_check_interval=1,
        )
        run = optimize(problem, pulse, robustness, config, control_groups=chain.control_groups)
        run.result.save(
            output / method, metadata={"demo": True, "seed": 7, "configuration": config.metadata()}
        )
        summary[method] = {
            "initial_loss": run.result.initial_objective,
            "final_loss": run.result.objective,
            "wall_time_s": run.elapsed_seconds,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
