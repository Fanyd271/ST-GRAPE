# Release performance changes

## What changed

Sparse block coalescing/composition previously evaluated `np.zeros_like` even when
a dictionary already contained the destination block, then allocated another sum.
The release branches on an existing block and accumulates into the owned result.
A first constructor block is copied to avoid aliasing the caller's arrays.
Neither the sparsity pattern nor the matrix-product order is changed.

The changes remove avoidable allocations during factor/generator construction.
They do not replace DOP853, tighten/relax tolerances, alter monitoring decisions,
or reduce Monte Carlo sample counts. The obsolete SciPy `disp` option was also
removed; progress callbacks remain available.

## Measured scope

Run:

```sh
python scripts/benchmark_block_composition.py --output outputs/composition_benchmark.json
```

The included microbenchmark compares a 20-block, physical-dimension-8 product
against the earlier allocation-heavy implementation, checks agreement at
`atol=1e-12`, discards two warm-up pairs, alternates method ordering, and records
20 repetitions. The checked run had median times of approximately 0.00664 s and
0.00434 s (about 1.53 times faster; roughly 35% less time). Machine information
and all raw timing values are in `performance_benchmark.json`.

This is **not** an end-to-end ST-GRAPE speedup or a repeated measurement of a
paper result. Improvement depends on physical dimension, block pattern, BLAS,
and workload. Full optimization timing was not rerun for release preparation.
The stored original benchmark times remain unchanged.
