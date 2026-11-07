"""Shared Matplotlib style for figures accompanying the paper."""

from __future__ import annotations

import matplotlib as mpl

PAPER_COLORS = {
    "initial": "#e68613",
    "grape": "#2a9d55",
    "st_grape": "#d1495b",
    "trotter": "#2166ac",
    "dense_ode": "#7b3294",
    "block_sparse_ode": "#008837",
    "penalty": "#d6604d",
}


def apply_paper_style() -> None:
    """Apply the font sizes and editable PDF fonts used by existing figures."""

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )
