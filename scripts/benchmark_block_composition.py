"""Compare release block composition against its allocation-heavy reference.

This microbenchmark does not reproduce or replace any published wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from stgrape.augmented.block_sparse import BlockOperator
from stgrape.paths import ensure_output_path


def reference_compose(left: BlockOperator, right: BlockOperator) -> BlockOperator:
    entries = {}
    rows = {}
    for position, row in enumerate(right.rows):
        rows.setdefault(int(row), []).append(position)
    for i, middle in enumerate(left.columns):
        for j in rows.get(int(middle), []):
            key = (int(left.rows[i]), int(right.columns[j]))
            product = left.blocks[i] @ right.blocks[j]
            entries[key] = entries.get(key, np.zeros_like(product)) + product
    return BlockOperator.from_entries(left.n_blocks, [(r, c, b) for (r, c), b in entries.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/composition_benchmark.json"))
    args = parser.parse_args()
    ensure_output_path(args.output)
    rng = np.random.default_rng(19)
    left = BlockOperator.from_entries(
        20,
        [
            (i, j, rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
            for i in range(20)
            for j in range(i + 1)
        ],
    )
    right = left.adjoint()
    np.testing.assert_allclose(
        left.compose(right).to_dense(), reference_compose(left, right).to_dense(), atol=1e-12
    )
    timings = {"reference": [], "release": []}
    for repeat in range(22):
        order = ("reference", "release") if repeat % 2 == 0 else ("release", "reference")
        for label in order:
            start = time.perf_counter()
            reference_compose(left, right) if label == "reference" else left.compose(right)
            elapsed = time.perf_counter() - start
            if repeat >= 2:
                timings[label].append(elapsed)
    report = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "blocks": 20,
        "physical_dimension": 8,
        "repetitions": 20,
        "median_time_s": {k: float(np.median(v)) for k, v in timings.items()},
        "raw_times_s": timings,
        "correctness": "allclose to reference at atol=1e-12",
    }
    report["speedup"] = report["median_time_s"]["reference"] / report["median_time_s"]["release"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_times_s"}, indent=2))


if __name__ == "__main__":
    main()
