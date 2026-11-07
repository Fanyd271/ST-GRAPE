# Noise sampling

Quasistatic channels are independent zero-mean Gaussian amplitudes. Their physical
standard deviations multiply the perturbation operators; this keeps aggregated
Taylor penalties dimensionally consistent.

During optimization, the colored-noise correlations are represented by finite
exponential fits and deterministic auxiliary systems. Validation defaults to the
original smooth, band-limited 1/f-type correlation, not the exponential fit.
The reference Gaussian sampler uses circulant embedding with a positive-spectrum
check and an explicit failure if a valid embedding cannot be found. This avoids
silently clipping a materially non-positive spectrum into a different noise model.

Within each comparison, initial and optimized controls share the same standardized
noise draws. The uncertainty-strength scan scales those draws across all strengths,
using 500 samples per strength by default. Process workers accelerate propagation
without changing the stored draw ordering.

The archived fidelity arrays are sufficient to regenerate the published CDFs.
The complete temporal trajectories were not saved for every historical run;
metadata retain sampling seeds, spectrum/correlation parameters and software
versions. Do not claim that a later sampler revision regenerates those trajectories
bit-for-bit unless that has been independently verified.
