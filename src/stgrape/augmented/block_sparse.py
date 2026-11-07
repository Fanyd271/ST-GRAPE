"""Small block-sparse linear algebra tailored to augmented dynamics.

The augmented dimension may be large while each nonzero block is an ordinary
physical-space operator.  Storing a NumPy array per nonzero block avoids the
large dense Kronecker matrices used by the reference GRAPE implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class BlockOperator:
    """Square sparse matrix whose entries are dense physical operators."""

    n_blocks: int
    block_dimension: int
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    blocks: ComplexArray
    unique_rows: NDArray[np.int64] = field(init=False, repr=False, compare=False)
    row_starts: NDArray[np.int64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        blocks = np.asarray(self.blocks, dtype=np.complex128)
        expected = (len(rows), self.block_dimension, self.block_dimension)
        if rows.ndim != 1 or columns.shape != rows.shape:
            raise ValueError("rows and columns must be one-dimensional arrays of equal length.")
        if blocks.shape != expected:
            raise ValueError(f"blocks has shape {blocks.shape}; expected {expected}.")
        if np.any(rows < 0) or np.any(rows >= self.n_blocks):
            raise ValueError("row index outside the augmented block range.")
        if np.any(columns < 0) or np.any(columns >= self.n_blocks):
            raise ValueError("column index outside the augmented block range.")

        merged: dict[tuple[int, int], ComplexArray] = {}
        for row, column, block in zip(rows, columns, blocks, strict=True):
            key = (int(row), int(column))
            if key in merged:
                merged[key] += block
            else:
                merged[key] = block.copy()
        entries = [
            (key, block) for key, block in sorted(merged.items()) if np.linalg.norm(block) > 1e-14
        ]
        object.__setattr__(self, "rows", np.array([key[0] for key, _ in entries], dtype=np.int64))
        object.__setattr__(
            self, "columns", np.array([key[1] for key, _ in entries], dtype=np.int64)
        )
        object.__setattr__(
            self,
            "blocks",
            np.stack([block for _, block in entries])
            if entries
            else np.empty((0, self.block_dimension, self.block_dimension), dtype=np.complex128),
        )
        sorted_rows = np.array([key[0] for key, _ in entries], dtype=np.int64)
        unique_rows, row_starts = np.unique(sorted_rows, return_index=True)
        object.__setattr__(self, "unique_rows", unique_rows)
        object.__setattr__(self, "row_starts", row_starts.astype(np.int64, copy=False))

    @classmethod
    def from_entries(
        cls,
        n_blocks: int,
        entries: list[tuple[int, int, ComplexArray]],
    ) -> BlockOperator:
        """Construct an operator from ``(row, column, block)`` entries."""

        if n_blocks < 1:
            raise ValueError("n_blocks must be positive.")
        if not entries:
            raise ValueError("Use BlockOperator.zeros when no entries are present.")
        block_dimension = np.asarray(entries[0][2]).shape[0]
        return cls(
            n_blocks=n_blocks,
            block_dimension=block_dimension,
            rows=np.array([entry[0] for entry in entries]),
            columns=np.array([entry[1] for entry in entries]),
            blocks=np.stack([np.asarray(entry[2], dtype=np.complex128) for entry in entries]),
        )

    @classmethod
    def zeros(cls, n_blocks: int, block_dimension: int) -> BlockOperator:
        """Construct the additive zero operator."""

        return cls(
            n_blocks,
            block_dimension,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, block_dimension, block_dimension), dtype=np.complex128),
        )

    @classmethod
    def identity(cls, n_blocks: int, block_dimension: int) -> BlockOperator:
        """Construct the augmented identity operator."""

        identity = np.eye(block_dimension, dtype=np.complex128)
        return cls.from_entries(n_blocks, [(index, index, identity) for index in range(n_blocks)])

    @classmethod
    def physical_diagonal(cls, n_blocks: int, matrix: ComplexArray) -> BlockOperator:
        """Repeat a physical operator on every augmented diagonal block."""

        matrix = np.asarray(matrix, dtype=np.complex128)
        return cls.from_entries(n_blocks, [(index, index, matrix) for index in range(n_blocks)])

    @classmethod
    def block_diagonal(cls, matrices: list[ComplexArray]) -> BlockOperator:
        """Construct an operator from potentially different diagonal blocks."""

        if not matrices:
            raise ValueError("matrices cannot be empty.")
        return cls.from_entries(
            len(matrices), [(index, index, matrix) for index, matrix in enumerate(matrices)]
        )

    @classmethod
    def linear_combination(
        cls,
        operators: tuple[BlockOperator, ...],
        coefficients: tuple[complex, ...],
    ) -> BlockOperator:
        """Combine compatible operators while merging their block patterns once.

        ODE propagation calls this once per piecewise-constant control interval.
        The resulting effective generator is then reused by every Runge--Kutta
        stage, avoiding one sparse traversal per control channel and RHS call.
        """

        if not operators:
            raise ValueError("operators cannot be empty.")
        if len(operators) != len(coefficients):
            raise ValueError("operators and coefficients must have equal length.")
        reference = operators[0]
        for operator in operators[1:]:
            reference._validate_compatible(operator)

        active = [
            (coefficient, operator)
            for coefficient, operator in zip(coefficients, operators, strict=True)
            if coefficient != 0 and len(operator.rows)
        ]
        if not active:
            return cls.zeros(reference.n_blocks, reference.block_dimension)
        return cls(
            n_blocks=reference.n_blocks,
            block_dimension=reference.block_dimension,
            rows=np.concatenate([operator.rows for _, operator in active]),
            columns=np.concatenate([operator.columns for _, operator in active]),
            blocks=np.concatenate(
                [coefficient * operator.blocks for coefficient, operator in active], axis=0
            ),
        )

    def apply(self, state: ComplexArray) -> ComplexArray:
        """Apply the operator to a stacked block state.

        ``state`` has shape ``(n_blocks, d, d_x)``, where ``d_x`` is one for
        state preparation and ``d`` for gate synthesis.
        """

        state = np.asarray(state, dtype=np.complex128)
        if state.ndim != 3 or state.shape[:2] != (self.n_blocks, self.block_dimension):
            raise ValueError("state must have shape (n_blocks, block_dimension, state_columns).")
        result = np.zeros_like(state)
        if len(self.rows):
            products = self.blocks @ state[self.columns]
            result[self.unique_rows] = np.add.reduceat(products, self.row_starts, axis=0)
        return result

    def adjoint(self) -> BlockOperator:
        """Return the Hermitian adjoint."""

        return BlockOperator(
            self.n_blocks,
            self.block_dimension,
            self.columns,
            self.rows,
            self.blocks.conj().transpose(0, 2, 1),
        )

    def trace(self) -> complex:
        """Return the augmented trace without materializing a dense matrix."""

        diagonal = self.rows == self.columns
        if not np.any(diagonal):
            return 0.0j
        return complex(np.trace(self.blocks[diagonal], axis1=1, axis2=2).sum())

    def scaled(self, scalar: complex) -> BlockOperator:
        """Multiply every physical block by ``scalar``."""

        return BlockOperator(
            self.n_blocks,
            self.block_dimension,
            self.rows,
            self.columns,
            scalar * self.blocks,
        )

    def add(self, other: BlockOperator) -> BlockOperator:
        """Add operators with identical augmented and physical dimensions."""

        self._validate_compatible(other)
        return BlockOperator(
            self.n_blocks,
            self.block_dimension,
            np.concatenate((self.rows, other.rows)),
            np.concatenate((self.columns, other.columns)),
            np.concatenate((self.blocks, other.blocks), axis=0),
        )

    def compose(self, other: BlockOperator) -> BlockOperator:
        """Return ``self @ other`` using block-sparse multiplication."""

        self._validate_compatible(other)
        entries: dict[tuple[int, int], ComplexArray] = {}
        other_by_row: dict[int, list[int]] = {}
        for position, row in enumerate(other.rows):
            other_by_row.setdefault(int(row), []).append(position)

        for left_position, middle in enumerate(self.columns):
            for right_position in other_by_row.get(int(middle), []):
                key = (int(self.rows[left_position]), int(other.columns[right_position]))
                product = self.blocks[left_position] @ other.blocks[right_position]
                if key in entries:
                    entries[key] += product
                else:
                    entries[key] = product

        if not entries:
            return BlockOperator.zeros(self.n_blocks, self.block_dimension)
        return BlockOperator.from_entries(
            self.n_blocks, [(row, column, block) for (row, column), block in entries.items()]
        )

    def to_dense(self) -> ComplexArray:
        """Materialize the full augmented matrix for validation and small systems."""

        dimension = self.n_blocks * self.block_dimension
        dense = np.zeros((dimension, dimension), dtype=np.complex128)
        d = self.block_dimension
        for row, column, block in zip(self.rows, self.columns, self.blocks, strict=True):
            dense[row * d : (row + 1) * d, column * d : (column + 1) * d] += block
        return dense

    def _validate_compatible(self, other: BlockOperator) -> None:
        if self.n_blocks != other.n_blocks or self.block_dimension != other.block_dimension:
            raise ValueError("BlockOperator dimensions do not match.")


def nilpotent_exponential(
    generator: BlockOperator,
    coefficient: complex,
    order: int,
) -> BlockOperator:
    """Evaluate a finite exponential of a nilpotent augmented generator.

    For a Taylor expansion truncated at order ``r``, the recursive coupling
    generator satisfies ``generator ** (r + 1) == 0``.  Its exponential is
    therefore represented exactly by ``r + 1`` terms.
    """

    if order < 0:
        raise ValueError("order must be non-negative.")
    result = BlockOperator.identity(generator.n_blocks, generator.block_dimension)
    power = BlockOperator.identity(generator.n_blocks, generator.block_dimension)
    for exponent in range(1, order + 1):
        power = generator.compose(power)
        result = result.add(power.scaled(coefficient**exponent / factorial(exponent)))
    return result
