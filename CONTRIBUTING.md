# Contributing

Use Python 3.12 or newer and install the checkout with `python -m pip install -e ".[dev]"`.
Before opening a pull request, run:

```sh
ruff check src tests scripts
ruff format --check src tests scripts
pyright
pytest
stgrape verify-data
```

Keep numerical models, optimizers, experiment execution, and plotting separate.
Numerical changes must include deterministic propagation/gradient regression tests.
Benchmark changes must keep compared controls and uncertainty realizations paired.
Write new runs under `outputs/`. Do not edit `data/` to make a failing test pass.

The tests intentionally do not run full-size optimizations. Ten-hour budgets are
per method and per seed, not estimates of the time needed by the complete suite.

Code contributions are provided under MIT. Reference-data changes require
source provenance, explicit justification, and CC BY 4.0 attribution. Do not add
credentials, private paths, third-party papers, or generated notebook outputs.
