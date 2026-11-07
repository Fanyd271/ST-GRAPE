# Reference data (CC BY 4.0)

These files support the article by Yidian Fan and Re-Bing Wu:
https://doi.org/10.1103/959l-ms9v

## Attribution and license

Copyright (c) 2026 Yidian Fan and Re-Bing Wu. All content below `data/`, including
reference figure artwork, is provided under **Creative Commons Attribution 4.0
International**. The full license is in `data/LICENSE`:
https://creativecommons.org/licenses/by/4.0/

Attribute the authors and article, identify the release you used, link the license,
and indicate changes. The software license in the repository root is MIT and does
not replace this data license.

## Mapping to the article

| Article item | Directory | Coverage |
|---|---|---|
| Fig. 1 | `reference_figures/flow_chart.pdf` | Algorithm diagram, no measured data |
| Fig. 2; Table I | `paper/ghz/` | Four qubits, 80 ns, orders 1/2, seeds 0-9, 10000 samples/control |
| Fig. 3 | `paper/scaling/ghz/` | Separate block-count and Hilbert-dimension timings |
| Fig. 4 | `paper/stress/` | Nine qubits, 200 ns, order 1, seeds 0-4, 500 samples/control |
| Fig. 5; Table II | `paper/qft/` | Three qubits, 300 ns, orders 1/2, seeds 0-9, 10000 samples/control |
| Fig. 6 | `paper/scaling/qft/` | Separate block-count and Hilbert-dimension timings |
| Nominal controls | `initial_controls/` | Current-format input controls used in the comparisons |
| Reply-only experiments | `supplementary/` | Exponential action and uncertainty-strength scans |

Do not substitute old ZZ-model scans or the other tuning runs from the development
repository. `manifest.json` identifies the selected files and their source hashes.
The historical nine-qubit nominal input directory had a misleading QFT prefix;
its stored task is state preparation. It is now named `initial_controls/stress/`.

## Formats and units

- `run_manifest.json` gives fixed Hamiltonian, noise, grid, optimizer and monitoring settings.
- `result.npz` stores controls and optimization histories. Coefficients are the
  optimization variables; `control_amplitudes` are the physical filtered/enveloped PWC controls.
- `method_summary.json` stores terminal reference loss, timing, and saved diagnostics.
- `fidelities.npz` / `initial_fidelities.npz`: numeric `values` arrays of sampled fidelities.
- `raw_data.npz`: timing arrays with axes documented by their names; failed points
  must be filtered using the saved status, not interpreted as zero time.
- Frequencies in the code use angular-frequency coefficients per microsecond.
  Published MHz values are obtained by dividing these coefficients by `2*pi`.
  Times use microseconds unless a key explicitly ends in `_s` or `_ns`.
- OptimizationResult `objective`, `initial_objective`, and their histories store
  losses (one minus the robust fidelity objective), despite the historical field
  names. The `true_objective` summary field is separately one minus `true_loss`.
  ST approximate histories and ODE-reference histories
  are distinct, and the stress experiment records the reference only at the end.

All NPZ files are retained byte-for-byte. JSON metadata have only environment-specific
absolute paths replaced with release-relative or explicitly redacted paths. The
manifest keeps both original and released file hashes. No numeric measurements
were edited. Noise-generation parameters, RNG seeds and fitted/reference spectra
are recorded; individual underlying noise trajectories were not always archived.
Saved fidelities permit exact CDF reconstruction without regenerating trajectories.

Run `stgrape verify-data` to check checksums and recompute all 76 displayed numeric
table entries, plus the four stress-test fidelity summaries, at the article's precision.
The verifier does not certify the scientific interpretation or recreate the original
hardware environment.
