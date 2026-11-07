"""Multi-index utilities for truncated Taylor expansions."""

from __future__ import annotations

import math


def generate_multi_indices(n_sources: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Return all Taylor multi-indices with total degree at most ``order``.

    Indices are ordered by descending base-``order + 1`` value.  Consequently,
    the nominal index ``(0, ..., 0)`` is always last, matching the stacked
    augmented state used in the paper.
    """

    if n_sources < 1:
        raise ValueError("n_sources must be positive.")
    if order < 0:
        raise ValueError("order must be non-negative.")

    # Descending traversal of each exponent reproduces the previous descending
    # base-(order + 1) ordering without visiting inadmissible tuples.  The old
    # Cartesian-product implementation inspected (order + 1) ** n_sources
    # candidates even though only binomial(order + n_sources, n_sources) are
    # retained.  This direct traversal is proportional to the output size.
    current = [0] * n_sources
    indices: list[tuple[int, ...]] = []

    def append_admissible(source: int, remaining_degree: int) -> None:
        if source == n_sources:
            indices.append(tuple(current))
            return
        for exponent in range(remaining_degree, -1, -1):
            current[source] = exponent
            append_admissible(source + 1, remaining_degree - exponent)

    append_admissible(source=0, remaining_degree=order)
    result = tuple(indices)
    expected_count = math.comb(order + n_sources, n_sources)
    if len(result) != expected_count:
        raise RuntimeError("Internal error while generating Taylor multi-indices.")
    return result


def decrement(index: tuple[int, ...], source: int) -> tuple[int, ...] | None:
    """Decrease one component of a multi-index, or return ``None`` at zero."""

    if index[source] == 0:
        return None
    reduced = list(index)
    reduced[source] -= 1
    return tuple(reduced)
