"""Publication plotting helpers preserving the paper's visual conventions."""

from .controls import plot_control_comparison
from .convergence import plot_convergence
from .decomposition import LossDecompositionSeries, plot_loss_decomposition
from .distributions import (
    plot_error_cdf,
    plot_fidelity_cdf,
    plot_method_error_cdfs_by_noise,
    plot_method_fidelity_cdfs_by_noise,
)
from .scaling import plot_scaling_summary
from .style import PAPER_COLORS, apply_paper_style

__all__ = [
    "PAPER_COLORS",
    "LossDecompositionSeries",
    "apply_paper_style",
    "plot_control_comparison",
    "plot_convergence",
    "plot_error_cdf",
    "plot_fidelity_cdf",
    "plot_method_error_cdfs_by_noise",
    "plot_method_fidelity_cdfs_by_noise",
    "plot_loss_decomposition",
    "plot_scaling_summary",
]
