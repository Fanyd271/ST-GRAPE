import numpy as np

from stgrape.augmented.block_sparse import BlockOperator, nilpotent_exponential


def test_block_operator_matches_dense_linear_algebra() -> None:
    rng = np.random.default_rng(3)
    entries = [
        (0, 0, rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))),
        (0, 1, rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))),
        (1, 1, rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))),
    ]
    operator = BlockOperator.from_entries(2, entries)
    state = rng.normal(size=(2, 2, 1)) + 1j * rng.normal(size=(2, 2, 1))

    expected = operator.to_dense() @ state.reshape(4, 1)
    np.testing.assert_allclose(operator.apply(state).reshape(4, 1), expected)
    np.testing.assert_allclose(operator.adjoint().to_dense(), operator.to_dense().conj().T)
    np.testing.assert_allclose(
        operator.compose(operator.adjoint()).to_dense(),
        operator.to_dense() @ operator.to_dense().conj().T,
    )


def test_nilpotent_exponential_is_finite_and_exact() -> None:
    block = np.array([[2.0 + 0.0j]])
    generator = BlockOperator.from_entries(3, [(0, 1, block), (1, 2, block)])
    exponential = nilpotent_exponential(generator, coefficient=-0.2j, order=2)
    dense = generator.to_dense()
    expected = np.eye(3) - 0.2j * dense + ((-0.2j) ** 2 / 2) * dense @ dense
    np.testing.assert_allclose(exponential.to_dense(), expected)
