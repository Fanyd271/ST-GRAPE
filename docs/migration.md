# Final-release interface changes

This is a deliberately cleaned public snapshot, not a backwards-compatible
continuation of the private working repository.

Removed: the old TOML CLI/configuration module, calibration sweeps and automatic
weight-calibration APIs, historical nested-result discovery, implicit old-manifest
upgrades, Adams/ZVODE adapters and QuTiP/Qiskit-named solver presets, and editable
`local_arguments`/`local_parameters` execution blocks.

Use `stgrape experiment <name> --config experiments/<preset>.json`, or the Python
workflow API. Experiments are installed under `stgrape.benchmarks`, model builders
under `stgrape.models`, and shared execution helpers under `stgrape.execution`.
The `new_experiments` package, source-folder import fallbacks, and old script aliases
are intentionally absent.

Published data were normalized once during release preparation. All NPZ bytes are
unchanged. Only private absolute paths in JSON metadata were redacted or rewritten.
Runtime code does not upgrade old inputs; unsupported layouts fail explicitly.
Archived field names (including historical wording inside metadata) are provenance,
not active deprecated interfaces.
