import numpy as np
import pytest

from stgrape import (
    OptimizationAlgorithm,
    OptimizationConfig,
    PulseConfig,
    configure_ode_propagation,
    optimize,
)
from stgrape.models import build_paper_problem, build_static_coupling_robustness
from stgrape.optimizers import LBFGSBOptions
from stgrape.problem import ControlTask
from stgrape.propagation import DOP853Options, OdeRepresentation


def test_public_workflow_composes_problem_pulse_and_optimizer() -> None:
    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = PulseConfig(filter_taps=2).random_pulse(problem, seed=7)
    initial_coefficients = pulse.coefficients.copy()
    config = OptimizationConfig(
        algorithm=OptimizationAlgorithm.ST_GRAPE,
        optimizer=LBFGSBOptions(max_iterations=1),
        propagation=configure_ode_propagation(
            method="dop853",
            representation="block-sparse",
        ),
        exact_check_interval=1,
    )

    run = optimize(
        problem,
        pulse,
        robustness,
        config,
        control_groups=chain.control_groups,
    )

    assert run.result.method == "ST-GRAPE"
    assert run.elapsed_seconds >= 0.0
    assert run.result.control_amplitudes.shape == (
        problem.n_controls,
        problem.n_time_steps,
    )
    np.testing.assert_array_equal(pulse.coefficients, initial_coefficients)
    assert config.metadata()["algorithm"] == "st-grape"


def test_ode_configuration_uses_named_preset_and_representation() -> None:
    propagation = configure_ode_propagation(
        method="dop853",
        representation=OdeRepresentation.BLOCK_SPARSE,
        rtol=2e-7,
        atol=3e-9,
    )

    assert isinstance(propagation.solver, DOP853Options)
    assert propagation.solver.rtol == 2e-7
    assert propagation.solver.atol == 3e-9
    assert propagation.representation == OdeRepresentation.BLOCK_SPARSE


def test_paper_problem_accepts_a_custom_total_duration() -> None:
    """Changing T preserves dt and rebuilds the complete time grid."""

    _, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        duration=0.05,
        time_step=0.0005,
    )

    assert problem.duration == pytest.approx(0.05)
    assert problem.time_step == pytest.approx(0.0005)
    assert problem.n_time_steps == 100

    with pytest.raises(ValueError, match="integer multiple"):
        build_paper_problem(
            2,
            ControlTask.STATE_PREPARATION,
            duration=0.0501,
            time_step=0.0005,
        )
