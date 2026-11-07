import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from stgrape.plotting import (
    LossDecompositionSeries,
    plot_control_comparison,
    plot_convergence,
    plot_loss_decomposition,
)


def test_control_comparison_uses_all_ghz_channels() -> None:
    """Four-qubit X/Y controls occupy eight panels with shared PWC timing."""

    initial = np.arange(48.0).reshape(8, 6)
    figure, axes = plot_control_comparison(
        {
            "initial": initial,
            "grape": initial + 1.0,
            "st_grape": initial - 1.0,
        },
        time_step_us=0.00025,
        n_qubits=4,
    )

    assert axes.shape == (4, 2)
    assert len(figure.axes) == 8
    assert all(len(axis.lines) == 3 for axis in figure.axes)
    assert axes[0, 0].get_title() == "X control"
    assert axes[0, 1].get_title() == "Y control"
    np.testing.assert_allclose(axes[0, 0].lines[0].get_xdata(), np.arange(7) * 0.25)
    plt.close(figure)


def test_control_comparison_supports_two_grape_baselines() -> None:
    controls = np.arange(24.0).reshape(4, 6)
    figure, axes = plot_control_comparison(
        {
            "initial": controls,
            "dense_grape": controls + 1.0,
            "block_sparse_grape": controls + 2.0,
            "st_grape": controls + 3.0,
        },
        time_step_us=0.00025,
        n_qubits=2,
    )

    assert all(len(axis.lines) == 4 for axis in axes.flat)
    labels = {line.get_label() for line in axes[0, 0].lines}
    assert {"Dense GRAPE", "Block-sparse GRAPE", "ST-GRAPE"} <= labels
    plt.close(figure)


def test_convergence_plot_compares_both_baselines_with_true_st_loss() -> None:
    figure, axis = plot_convergence(
        baseline_series={
            "dense_grape": (np.array([0.0, 2.0]), np.array([0.5, 0.2])),
            "block_sparse_grape": (
                np.array([0.0, 1.0]),
                np.array([0.5, 0.2]),
            ),
        },
        st_times=np.array([0.0, 0.5]),
        st_losses=np.array([0.5, 0.1]),
        exact_times=np.array([0.0, 0.6]),
        exact_losses=np.array([0.5, 0.12]),
        exact_iterations=np.array([0, 1]),
        inset_iterations={
            "dense_grape": np.array([0.5, 0.2]),
            "block_sparse_grape": np.array([0.5, 0.2]),
            "st_grape": np.array([0.5, 0.1]),
        },
    )

    labels = {line.get_label() for line in axis.lines}
    assert {"Dense GRAPE", "Block-sparse GRAPE", r"$1-J$ in ST-GRAPE"} <= labels
    assert r"$1-\hat{J}$ in ST-GRAPE" not in labels
    plt.close(figure)


def test_loss_decomposition_matches_figure_seven_structure() -> None:
    """Figure 7 contains one fidelity and one penalty trajectory per size."""

    figure, axes = plot_loss_decomposition(
        (
            LossDecompositionSeries(
                n_qubits=7,
                times_s=np.array([0.0, 3600.0]),
                fidelity_losses=np.array([0.03, 0.001]),
                robustness_penalties=np.array([0.001, 0.002]),
            ),
            LossDecompositionSeries(
                n_qubits=9,
                times_s=np.array([0.0, 36000.0]),
                fidelity_losses=np.array([0.09, 0.04]),
                robustness_penalties=np.array([0.001, 0.03]),
                time_limit_hours=10.0,
                time_limit_reached=True,
            ),
        )
    )

    assert axes.shape == (2,)
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 3
    assert "converged" in axes[0].get_title(loc="left")
    assert "time limit reached" in axes[1].get_title(loc="left")
    plt.close(figure)
