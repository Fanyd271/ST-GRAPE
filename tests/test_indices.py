import itertools
import math

from stgrape.augmented.indices import generate_multi_indices


def test_multi_index_count_and_nominal_position() -> None:
    for n_sources in range(1, 5):
        for order in range(1, 4):
            indices = generate_multi_indices(n_sources, order)
            assert len(indices) == math.comb(order + n_sources, n_sources)
            assert indices[-1] == (0,) * n_sources
            assert all(sum(index) <= order for index in indices)


def test_direct_generator_preserves_the_legacy_order_exactly() -> None:
    """The efficient traversal must not change any augmented block number."""

    for n_sources in range(1, 7):
        for order in range(5):
            expected = tuple(
                sorted(
                    (
                        index
                        for index in itertools.product(
                            range(order + 1),
                            repeat=n_sources,
                        )
                        if sum(index) <= order
                    ),
                    reverse=True,
                )
            )
            assert generate_multi_indices(n_sources, order) == expected


def test_direct_generator_handles_the_large_block_benchmark() -> None:
    """The planned m=64, r=2 benchmark creates only its 2145 valid blocks."""

    indices = generate_multi_indices(n_sources=64, order=2)

    assert len(indices) == math.comb(66, 2) == 2145
    assert indices[0] == (2,) + (0,) * 63
    assert indices[-1] == (0,) * 64
    assert all(sum(index) <= 2 for index in indices)
