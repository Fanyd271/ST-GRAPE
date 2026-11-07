# Numerical methods and maintained interfaces

The public Python interface consists of `ControlProblem`, the quasistatic/colored
noise models in `stgrape.problem`, the shaped pulse classes in `stgrape.pulse`,
`OptimizationConfig`, `configure_ode_propagation`, and `optimize`. System examples
are built by `stgrape.models`; experiment orchestration is in `stgrape.execution`.
The latter never changes the core physical model implicitly.

- `augmented/indices.py` enumerates total-order Taylor multi-indices.
- `augmented/builders.py` constructs quasistatic and colored-noise augmented systems.
- `augmented/block_sparse.py` applies/combines block operators without dense augmentation.
- `propagation/trotter.py` evaluates structured factors and their derivatives.
- `propagation/ode.py` uses the same SciPy DOP853 tolerances for dense/block-sparse baselines.
- `propagation/exponential_action.py` supplies a matrix-free LinearOperator for the reply baseline.
- `propagation/exact.py` is a small dense-exponential **test reference**, not an old production API.

For a dimensionless independent-Gaussian parameterization, the aggregate first-
and second-order measures are the expected Frobenius norms of the corresponding
Taylor contributions, divided by the physical state-column count. The second-order
formula includes the nonzero cross terms between repeated-index coefficients.
The hybrid objective squares the first-order aggregate and uses the smooth
Charbonnier form for second-order and colored corrections. See the article for
the precise normalization, smoothing constants and Gaussian-moment derivation.

ST differentiation is exact for the **Trotterized objective**, not the original
unsplit evolution. Periodic ODE-reference monitoring may increase internal
splitting without increasing the number of control parameters. The fixed-budget
stress preset intentionally disables those checks and records an exact endpoint.

The squared and smooth-RMS objective options remain explicit general-purpose
choices; they are not aliases for the article's default hybrid objective. The
Chebyshev pulse parameterization likewise remains a tested optional model rather
than being silently enabled by the paper presets.
