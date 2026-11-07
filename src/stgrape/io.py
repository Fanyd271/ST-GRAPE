"""Input helpers for pulse seeds saved with the current result schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pulse import (
    DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT,
    DEFAULT_CHEBYSHEV_BASIS_SIZE,
    ChebyshevNormalization,
    ChebyshevPulse,
    ControlBasis,
    FilteredPulse,
    Pulse,
)


def pulse_from_result(
    path: str | Path,
    *,
    n_controls: int,
    n_time_steps: int,
    time_step: float,
    rise_time: float = 0.01,
    cutoff_frequency: float = 60.0,
    filter_taps: int = 10,
    control_basis: ControlBasis = "pwc",
    chebyshev_k: int = DEFAULT_CHEBYSHEV_BASIS_SIZE,
    chebyshev_normalization: ChebyshevNormalization = "rms",
    chebyshev_amplitude_limit: float = DEFAULT_CHEBYSHEV_AMPLITUDE_LIMIT,
) -> Pulse:
    """Load a configured pulse from the current result schema only.

    Files without the current ``coefficients`` array are deliberately rejected.
    """

    result_path = Path(path)
    if result_path.suffix != ".npz" or not result_path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {result_path}")
    with np.load(result_path, allow_pickle=False) as data:
        if "coefficients" not in data.files:
            raise ValueError("Result is not in the current schema: missing 'coefficients'.")
        coefficients = np.asarray(data["coefficients"], dtype=float)
    common = {
        "n_controls": n_controls,
        "n_time_steps": n_time_steps,
        "time_step": time_step,
        "coefficients": coefficients,
        "rise_time": rise_time,
        "cutoff_frequency": cutoff_frequency,
        "filter_taps": filter_taps,
    }
    if control_basis == "pwc":
        return FilteredPulse(**common)
    if control_basis == "chebyshev":
        return ChebyshevPulse(
            **common,
            n_basis=chebyshev_k,
            normalization=chebyshev_normalization,
            amplitude_limit=chebyshev_amplitude_limit,
        )
    raise ValueError("control_basis must be 'pwc' or 'chebyshev'.")
