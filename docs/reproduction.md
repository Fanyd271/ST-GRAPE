# Reproduction guide

## Level 1: saved-data reproduction (recommended starting point)

```sh
stgrape verify-data
stgrape reproduce --output outputs/paper
stgrape reproduce --supplementary --output outputs/paper_and_reply
```

The command only reads the reference archive. It regenerates the existing plotting
strategy, colors, endpoint markers, seed ranges, empirical CDFs, and representative
controls. Separate subdirectories prevent GHZ/QFT exports from overwriting each other.
`table_ghz.csv`, `table_qft.csv`, and `summary.json` give independently recomputed
statistics. Table entries are averaged across seeds, including per-seed time/iteration
ratios. ST true-loss checkpoints are not replaced by Trotterized trajectories.

The optional PDF-crop utility is not used by default. Tight bounding boxes are
available without external tools; final whitespace may differ slightly from the
archived publication figures, which remain in `data/reference_figures/`.

## Level 2: short numerical checks

```sh
stgrape demo --output outputs/demo
stgrape experiment scaling-blocks --task ghz --smoke --output outputs/smoke_blocks
stgrape experiment scaling-dimension --task qft --smoke --output outputs/smoke_dimension
stgrape experiment robustness --task ghz --initial-source random --smoke --method all --seeds 0 --output outputs/smoke_robustness
```

Use fresh output directories, or explicitly select `--existing-result overwrite`
only for your own generated runs. An old result without a current manifest is
rejected instead of silently converted. A smoke model is not the paper model.

## Level 3: full optimization and sampling

Use the JSON presets in `experiments/`, not implicit script defaults. They record
the archived physical grid, noise selection, loss coefficients and solver options.
The default GHZ/QFT optimization seeds are the saved nominal controls rather than
new random fields. The GHZ/QFT presets test both robustness orders and all three methods.

```sh
stgrape experiment robustness --config experiments/ghz.json
stgrape experiment robustness --config experiments/qft.json
stgrape experiment stress --config experiments/stress.json
stgrape experiment stress-sampling --help
stgrape experiment strength-scan --help
```

Stress optimization disables periodic true-loss checks, fixes internal splitting
at one (`q=1`, as recorded in the data), and performs a final ODE-reference evaluation outside the timed optimization.
Monte Carlo sampling is a separate step: use the stress-sampling command's
`--result-root`, `--samples 500`, `--sampling-seed 42`, and `--workers` options as
shown by its help. Do not mix a new sample set into the published archive.

The independently saved 500-trajectory stress sample set is used by Fig. 4.
The inline sampling summaries inside some optimization directories describe a
different run and must not be substituted for that sample set.

## Parallelism and timing

Sampling supports process workers; increase `--sampling-workers`/`--workers`
explicitly and account for per-worker memory. Notebook plots need no process pool.
Limit BLAS threads for consistent comparisons, for example:

```sh
# POSIX shells
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# PowerShell equivalents: $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"
```

Do not run competing CPU workloads during timing. Software versions and available
runtime metadata are preserved with the measured data; unrecorded historical
thread settings cannot be reconstructed. Current local validation versions are
not asserted to be the original experimental environment.

Repeated optimization need not follow bit-identical trajectories across BLAS,
platform, or SciPy versions. Validate objectives/gradients within numerical
accuracy and compare statistical outcomes, not exact wall-clock seconds.

## Historical versus current software

Representative archived benchmark metadata record Python 3.12.5, NumPy 1.26.4,
SciPy 1.13.1 and QuTiP 5.0.4 on Windows 11. These are not overwritten by the
current release checks. `requirements-reproduction.txt` pins the newly tested
plotting/runtime environment, while `docs/verification.json` records the complete
release-check context. Neither file certifies every older or future combination.
