# Changelog

## 1.0.0 (prepared for public release)

- Provide a single package/CLI for the final GHZ, QFT, scaling, and stress experiments.
- Remove historical TOML entrypoints, calibration sweeps, Adams/ZVODE adapters,
  solver-brand presets, implicit result-format upgrades, and source-local run presets.
- Archive the data underlying Figs. 2-6 and Tables I-II, with file checksums,
  current-schema initial controls, and separate reply-only supplementary results.
- Share plotting implementations between the CLI and lightweight notebooks.
- Protect reference data from experiment/analysis output paths.
- Remove unnecessary zero-array allocations in sparse block construction/composition.
- Remove SciPy's obsolete `disp` optimizer option without changing tolerances.
- License software under MIT and numerical/figure data under CC BY 4.0.

Historical timings remain the original measurements. They are not replaced by
measurements from this refactored implementation or from the release-check machine.
