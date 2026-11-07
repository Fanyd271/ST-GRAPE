# ST-GRAPE

**Structure-aware Suzuki-Trotter gradient ascent pulse engineering for robust quantum control.**

This repository provides the software and numerical data supporting:

> Yidian Fan and Re-Bing Wu, *Efficient robust control design for high-dimensional
> quantum systems against multiple uncertainties using Suzuki-Trotter decomposition*,
> accepted in **Physical Review A**. DOI: [10.1103/959l-ms9v](https://doi.org/10.1103/959l-ms9v).

Public repository: [Fanyd271/ST-GRAPE](https://github.com/Fanyd271/ST-GRAPE).
Complete source-and-data packages are available from the
[versioned releases](https://github.com/Fanyd271/ST-GRAPE/releases).

The maintained implementations are ST-GRAPE, dense GRAPE, and block-sparse GRAPE
with SciPy DOP853. A matrix-exponential-action baseline is included for the
supplementary comparison. QuTiP is used for some operators and physical-system
objects; the GRAPE reference dynamics use SciPy, not QuTiP's evolution solvers.

## Installation

Use Python **3.12 or newer**. From a downloaded/cloned source checkout:

```sh
git clone https://github.com/Fanyd271/ST-GRAPE.git
cd ST-GRAPE
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell (instead): .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For notebooks, also run `python -m pip install -e ".[notebook]"`. Conda users
can instead use `conda env create -f environment.yml` and `conda activate stgrape`.
The final validated environment is recorded in `requirements-reproduction.txt`
and `docs/verification.json`; historical experiment versions remain in the saved metadata.

## Quick start

```sh
stgrape --help
stgrape demo --output outputs/demo
stgrape verify-data
stgrape reproduce --output outputs/paper
```

- `demo` runs a small two-qubit comparison for two L-BFGS-B iterations per method.
  Use a new output directory for each run.
- `verify-data` checks the 974 archived file hashes, recomputes Tables I-II at the
  article's displayed precision, and checks the stress-test mean fidelities.
- `reproduce` regenerates Figs. 2-6 and numeric table CSVs from saved data. It
  copies the Fig. 1 diagram. **No optimization or noise propagation is launched.**
- Add `--supplementary` to also regenerate the exponential-action scaling
  comparisons and uncertainty-strength scans from the response letter.

Run from the checkout. For a separate data location, set `STGRAPE_DATA_DIR`
to its absolute `data/` path. Code wheels do not bundle the reference data;
the complete source-release archive does.

## Run new experiments

The final paper settings are explicit, versioned JSON presets:

```sh
stgrape experiment robustness --config experiments/ghz.json
stgrape experiment robustness --config experiments/qft.json
stgrape experiment scaling-blocks --config experiments/ghz_scaling_blocks.json
stgrape experiment scaling-dimension --config experiments/qft_scaling_dimension.json
stgrape experiment stress --config experiments/stress.json
```

**Full runs are expensive.** The stress preset permits approximately ten hours
**per method per seed** (15 runs), followed by reference evaluations. It does
not sample noise during the optimization. See [reproduction instructions](docs/reproduction.md)
for sampling and smoke commands. CLI arguments after the preset may override
scalar options such as `--output`. All new results belong in `outputs/`.

The `data/` archive is immutable reference input; all supported runners reject
output paths inside it. Existing result directories require matching current
manifests. There are no historical-format upgrades or deprecated solver aliases.

## Contents

| Path | Purpose |
|---|---|
| `src/stgrape/` | Models, augmented operators, propagation, objectives, optimization and sampling |
| `src/stgrape/benchmarks/` | Supported experiment runners behind the single CLI |
| `src/stgrape/analysis/` | Shared implementations for paper and supplementary plots |
| `experiments/` | Explicit full-size GHZ/QFT/scaling/stress presets |
| `notebooks/` | Thin interactive frontends to the shared plotting code |
| `data/paper/` | Original controls, convergence histories, sampled fidelities and timings |
| `data/initial_controls/` | Nominal input controls for reproducing optimization starts |
| `data/supplementary/` | Reply-only strength scans and exponential-action comparisons |
| `data/manifest.json` | Dataset provenance and released/original SHA-256 checksums |
| `tests/` | Numerical, interface and archived-data regression tests |
| `docs/` | Methods, reproduction, migration, data details and validation record |

See [data documentation](data/README.md) for the figure/table mapping, units,
file schemas, and exactly which seeds and Monte Carlo samples are included.

## Reproducibility and performance

The published hardware was an Intel Core i9-14900KF with 32 GB RAM. Absolute
wall-clock times depend on hardware, BLAS threads, dependencies and solver
adaptivity; iteration-level objectives and stored-data plots are the appropriate
portable regression targets. Scaling samples were not repeatedly measured at
all points. Do not interpret single stored measurements as confidence intervals.

This public release removes compatibility plumbing and unnecessary allocations
while preserving the numerical definitions. Historical timing arrays are retained
unchanged; no claim is made that release microbenchmarks reproduce published
speedups. See [performance notes](docs/performance.md) and the verification report.

Vector-PDF whitespace cropping is **optional**. The numerical code and normal
paper reproduction do not require LaTeX or Poppler. The separate
`stgrape-crop-pdf` utility requires `pdfinfo`, `pdftoppm`, and `pdfcrop`.

## Citation and licenses

Please cite the article DOI above. `CITATION.cff` provides GitHub's citation metadata.
The public GitHub release does not yet have a separate software/data DOI. The
article DOI identifies the paper, not the software or data archive.

- Software and code documentation: **MIT**, see [LICENSE](LICENSE).
- Everything under `data/`, including reference figures: **CC BY 4.0**, see
  [data/LICENSE](data/LICENSE) and [attribution instructions](data/README.md).

No third-party article PDFs, private Git history, development-agent settings,
credentials, or historical tuning runs are distributed.

## Development

```sh
ruff check src tests scripts
ruff format --check src tests scripts
pyright
pytest
```

CI is configured for Linux and Windows. Check the
[Actions results](https://github.com/Fanyd271/ST-GRAPE/actions) for the exact commit
being used. `docs/verification.json` records the separate local verification.

## Acknowledgments

This work was supported by the Innovation Program for Quantum Science and
Technology (No. 2021ZD0300201), the Tsinghua-Toyota Joint Research Fund, and the
Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry
of Education of China.
