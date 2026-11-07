import json
from contextlib import suppress

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import stgrape.optimizers.st_grape as st_grape_module
from stgrape.io import pulse_from_result
from stgrape.models import build_paper_problem, build_static_coupling_robustness
from stgrape.optimizers import LBFGSBOptions, STGrapeSolver
from stgrape.problem import ControlTask
from stgrape.pulse import ChebyshevPulse, FilteredPulse


def test_current_result_can_be_reused_as_an_experiment_seed(tmp_path) -> None:
    """Maintained experiments can consume the stable result schema from E00."""

    path = tmp_path / "result.npz"
    coefficients = np.arange(12.0).reshape(2, 6)
    np.savez(path, coefficients=coefficients)

    pulse = pulse_from_result(
        path,
        n_controls=2,
        n_time_steps=7,
        time_step=0.001,
        rise_time=0.001,
        cutoff_frequency=60.0,
        filter_taps=2,
    )

    np.testing.assert_array_equal(pulse.coefficients, coefficients)


def test_chebyshev_result_can_be_reused_with_its_basis_size(tmp_path) -> None:
    path = tmp_path / "result.npz"
    coefficients = np.arange(16.0).reshape(2, 8)
    np.savez(path, coefficients=coefficients)

    pulse = pulse_from_result(
        path,
        n_controls=2,
        n_time_steps=20,
        time_step=0.001,
        rise_time=0.002,
        cutoff_frequency=60.0,
        filter_taps=4,
        control_basis="chebyshev",
        chebyshev_k=8,
    )

    assert isinstance(pulse, ChebyshevPulse)
    np.testing.assert_array_equal(pulse.coefficients, coefficients)
    assert pulse.parameter_bounds(1.0) is None


def test_current_result_loader_rejects_unknown_arrays(tmp_path) -> None:
    """Only the current result schema can enter E00-based workflows."""

    path = tmp_path / "legacy.npz"
    np.savez(path, x=np.arange(12.0).reshape(2, 6))
    with pytest.raises(ValueError, match="current schema"):
        pulse_from_result(
            path,
            n_controls=2,
            n_time_steps=7,
            time_step=0.001,
            filter_taps=2,
        )


def test_short_st_grape_run_and_serialization(tmp_path):
    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(7),
    )
    options = LBFGSBOptions(max_iterations=1)
    progress: list[tuple[int, float, float]] = []
    result = STGrapeSolver(options, exact_check_interval=1).optimize(
        problem,
        pulse,
        chain.control_groups,
        robustness,
        progress_callback=lambda iteration, loss, elapsed: progress.append(
            (iteration, loss, elapsed)
        ),
    )

    assert result.method == "ST-GRAPE"
    assert progress[0][0] == 0
    assert progress[0][1] == pytest.approx(result.initial_objective)
    assert all(elapsed >= 0.0 for _, _, elapsed in progress)
    assert result.exact_objective is not None
    assert result.exact_objective_iteration is not None
    assert result.exact_evaluation_iterations.size == result.exact_objective_history.size
    assert np.isfinite(result.objective)
    assert result.coefficients.shape == pulse.coefficients.shape
    assert result.control_amplitudes.shape == (
        problem.n_controls,
        problem.n_time_steps,
    )

    output = result.save(tmp_path / "run", metadata={"seed": 7})
    with np.load(output / "result.npz") as saved:
        assert float(saved["objective"]) == pytest.approx(result.objective)
        assert float(saved["exact_objective"]) == pytest.approx(result.exact_objective)
        np.testing.assert_array_equal(
            saved["exact_evaluation_iterations"],
            result.exact_evaluation_iterations,
        )
        np.testing.assert_array_equal(
            saved["trotter_splitting_steps_history"],
            result.trotter_splitting_steps_history,
        )
        np.testing.assert_array_equal(saved["splitting_change_steps"], [1])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["method"] == "ST-GRAPE"
    assert metadata["seed"] == 7
    assert metadata["propagation"]["representation"] == "suzuki-trotter"
    assert metadata["propagation"]["reference_check"]["solver"]["method"] == "dop853"

    with pytest.raises(FileExistsError):
        result.save(output)


def test_grape_records_fidelity_and_robustness_loss_components() -> None:
    """The Fig.-7 decomposition is recorded without an extra callback propagation."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(17),
    )
    result = st_grape_module.GrapeSolver(LBFGSBOptions(max_iterations=1)).optimize(
        problem, pulse, robustness
    )

    assert result.initial_fidelity_loss is not None
    assert result.initial_robustness_penalty is not None
    assert result.initial_objective == pytest.approx(
        result.initial_fidelity_loss + result.initial_robustness_penalty
    )
    assert result.fidelity_loss_history.shape == result.objective_history.shape
    assert result.robustness_penalty_history.shape == result.objective_history.shape
    np.testing.assert_allclose(
        result.objective_history,
        result.fidelity_loss_history + result.robustness_penalty_history,
    )


def test_st_grape_can_skip_periodic_but_keep_final_true_loss() -> None:
    """The n>=9 paper policy performs exactly one final ODE reference check."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(18),
    )
    result = STGrapeSolver(
        LBFGSBOptions(max_iterations=1),
        exact_check_interval=None,
        final_exact_evaluation=True,
    ).optimize(problem, pulse, chain.control_groups, robustness)

    assert result.exact_objective is not None
    assert result.exact_objective_history.shape == (1,)
    assert result.exact_evaluation_iterations.shape == (1,)
    assert result.propagation["reference_check_interval"] is None
    assert result.propagation["final_reference_evaluation"] is True


def test_st_grape_true_loss_safeguard_retains_best_controls(monkeypatch) -> None:
    """Patience stops persistent true-loss deterioration and restores the best pulse."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(8),
    )
    candidates: list[np.ndarray] = []

    def fake_minimize(_function, initial_parameters, *, callback, **_options):
        for index in range(5):
            candidate = np.asarray(initial_parameters) + (index + 1) * 1e-4
            candidates.append(candidate)
            intermediate = OptimizeResult(x=candidate, fun=0.8 - 0.1 * index)
            try:
                callback(intermediate)
            except StopIteration:
                return OptimizeResult(
                    x=candidate,
                    fun=intermediate.fun,
                    message="stopped by callback",
                    success=False,
                )
        raise AssertionError("The fifth true-loss evaluation should exhaust patience.")

    # The new minimum at iteration 3 must reset the first deterioration.
    true_losses = iter((0.5, 0.4, 0.45, 0.3, 0.35, 0.36))

    def fake_true_evaluation(*_args, **_kwargs):
        return next(true_losses), None

    monkeypatch.setattr(st_grape_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        st_grape_module.GrapeSolver,
        "_evaluate",
        staticmethod(fake_true_evaluation),
    )
    exact_progress: list[tuple[int, float, float]] = []
    result = STGrapeSolver(
        LBFGSBOptions(max_iterations=10),
        exact_check_interval=1,
        exact_increase_patience=2,
    ).optimize(
        problem,
        pulse,
        chain.control_groups,
        robustness,
        exact_progress_callback=lambda iteration, loss, elapsed: exact_progress.append(
            (iteration, loss, elapsed)
        ),
    )

    np.testing.assert_allclose(
        result.exact_objective_history,
        [0.5, 0.4, 0.45, 0.3, 0.35, 0.36],
    )
    np.testing.assert_array_equal(
        result.exact_evaluation_iterations,
        [0, 1, 2, 3, 4, 5],
    )
    assert result.exact_objective == pytest.approx(0.3)
    assert result.exact_objective_iteration == 3
    assert result.exact_deterioration_count == 2
    assert result.exact_safeguard_termination_iteration == 5
    np.testing.assert_allclose(
        result.coefficients,
        candidates[2].reshape(pulse.n_controls, pulse.n_parameters_per_control),
    )
    assert [iteration for iteration, _, _ in exact_progress] == [0, 1, 2, 3, 4, 5]
    assert "true loss deteriorated 2 times" in result.message


def test_st_grape_safeguard_can_restore_the_nominal_pulse(monkeypatch) -> None:
    """The unmodified input remains eligible as the best true-loss control."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(81),
    )
    initial_coefficients = pulse.coefficients.copy()

    def fake_minimize(_function, initial_parameters, *, callback, **_options):
        candidate = np.asarray(initial_parameters) + 1e-4
        intermediate = OptimizeResult(x=candidate, fun=0.1)
        with suppress(StopIteration):
            callback(intermediate)
        return OptimizeResult(
            x=candidate,
            fun=intermediate.fun,
            message="stopped by callback",
            success=False,
        )

    true_losses = iter((0.2, 0.3))

    monkeypatch.setattr(st_grape_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        st_grape_module.GrapeSolver,
        "_evaluate",
        staticmethod(lambda *_args, **_kwargs: (next(true_losses), None)),
    )
    result = STGrapeSolver(
        LBFGSBOptions(max_iterations=2),
        exact_check_interval=1,
        exact_increase_patience=1,
    ).optimize(problem, pulse, chain.control_groups, robustness)

    np.testing.assert_array_equal(result.exact_evaluation_iterations, [0, 1])
    assert result.exact_objective == pytest.approx(0.2)
    assert result.exact_objective_iteration == 0
    np.testing.assert_allclose(result.coefficients, initial_coefficients)


def test_st_grape_adaptive_splitting_restarts_once_at_the_configured_limit(
    monkeypatch,
) -> None:
    """A 30% true/approximate gap refines 1 -> 2 and restarts L-BFGS-B."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(28),
    )
    minimize_starts: list[np.ndarray] = []

    def fake_minimize(_function, initial_parameters, *, callback, **_options):
        minimize_starts.append(np.asarray(initial_parameters).copy())
        candidate = np.asarray(initial_parameters) + 1e-4
        approximate_loss = 0.5 if len(minimize_starts) == 1 else 0.2
        intermediate = OptimizeResult(x=candidate, fun=approximate_loss)
        with suppress(StopIteration):
            callback(intermediate)
        return OptimizeResult(
            x=candidate,
            fun=approximate_loss,
            message="synthetic stage complete",
            success=True,
        )

    true_losses = iter((0.55, 0.8, 0.21))

    def fake_true_evaluation(*_args, **_kwargs):
        return next(true_losses), None

    monkeypatch.setattr(st_grape_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        st_grape_module.GrapeSolver,
        "_evaluate",
        staticmethod(fake_true_evaluation),
    )
    refinements: list[tuple[int, int, int, float]] = []
    result = STGrapeSolver(
        LBFGSBOptions(max_iterations=4),
        exact_check_interval=1,
        adaptive_splitting=True,
        max_splitting_steps=2,
        splitting_relative_tolerance=0.30,
        splitting_absolute_tolerance=0.0,
    ).optimize(
        problem,
        pulse,
        chain.control_groups,
        robustness,
        splitting_progress_callback=lambda iteration, old, new, ratio: refinements.append(
            (iteration, old, new, ratio)
        ),
    )

    assert len(minimize_starts) == 2
    np.testing.assert_allclose(minimize_starts[1], minimize_starts[0] + 1e-4)
    np.testing.assert_array_equal(result.splitting_change_iterations, [0, 1])
    np.testing.assert_array_equal(result.splitting_change_steps, [1, 2])
    np.testing.assert_array_equal(result.trotter_splitting_steps_history, [1, 2])
    np.testing.assert_allclose(
        result.exact_approximate_objective_history[-2:],
        [0.5, 0.2],
    )
    np.testing.assert_allclose(
        result.exact_approximation_gap_ratio_history[-2:],
        [0.375, 1.0 / 21.0],
    )
    assert refinements[0][:3] == (1, 1, 2)
    assert refinements[0][3] == pytest.approx(0.375)
    assert result.propagation["final_splitting_steps"] == 2
    assert "splitting steps 1->2 at iteration 1" in result.message


def test_adaptive_splitting_refines_on_first_mixed_tolerance_exceedance(
    monkeypatch,
) -> None:
    """One mixed-error exceedance immediately requests a finer ST grid."""

    chain, problem = build_paper_problem(
        2,
        ControlTask.STATE_PREPARATION,
        time_step=0.0025,
    )
    robustness = build_static_coupling_robustness(chain, order=1)
    pulse = FilteredPulse.random(
        n_controls=problem.n_controls,
        n_time_steps=problem.n_time_steps,
        time_step=problem.time_step,
        rng=np.random.default_rng(29),
    )
    minimize_starts: list[np.ndarray] = []

    def fake_minimize(_function, initial_parameters, *, callback, **_options):
        minimize_starts.append(np.asarray(initial_parameters).copy())
        callback_count = 1
        candidate = np.asarray(initial_parameters).copy()
        for _ in range(callback_count):
            candidate = candidate + 1e-4
            try:
                callback(OptimizeResult(x=candidate, fun=0.2))
            except StopIteration:
                break
        return OptimizeResult(
            x=candidate,
            fun=0.2,
            message="synthetic stage complete",
            success=True,
        )

    true_losses = iter((0.5, 0.4, 0.21))
    monkeypatch.setattr(st_grape_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        st_grape_module.GrapeSolver,
        "_evaluate",
        staticmethod(lambda *_args, **_kwargs: (next(true_losses), None)),
    )

    result = STGrapeSolver(
        LBFGSBOptions(max_iterations=4),
        exact_check_interval=1,
        adaptive_splitting=True,
        max_splitting_steps=2,
        splitting_relative_tolerance=0.10,
        splitting_absolute_tolerance=0.01,
    ).optimize(problem, pulse, chain.control_groups, robustness)

    assert len(minimize_starts) == 2
    np.testing.assert_array_equal(result.splitting_change_iterations, [0, 1])
    np.testing.assert_array_equal(result.splitting_change_steps, [1, 2])
    np.testing.assert_array_equal(result.trotter_splitting_steps_history, [1, 2])
    np.testing.assert_allclose(
        result.exact_approximation_absolute_error_history[1:],
        [0.2, 0.01],
    )
    np.testing.assert_allclose(
        result.exact_splitting_tolerance_history[1:],
        [0.05, 0.031],
    )
    np.testing.assert_array_equal(
        result.exact_splitting_tolerance_exceeded_history[1:],
        [True, False],
    )
    assert result.propagation["splitting_error_criterion"] == ("absolute-plus-relative")
