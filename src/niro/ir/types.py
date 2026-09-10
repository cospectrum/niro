"""Types in the Niro intermediate representation.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ScalarType(enum.Enum):
    """The supported boolean, signed integer, and floating-point scalar types."""

    BOOL = "bool"
    I32 = "i32"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"

    @property
    def byte_width(self) -> int:
        """Return the packed storage width of one scalar in bytes."""
        return {
            ScalarType.BOOL: 1,
            ScalarType.I32: 4,
            ScalarType.I64: 8,
            ScalarType.F32: 4,
            ScalarType.F64: 8,
        }[self]


Dimension = int | None
"""A nonnegative axis size, or None when the size is unknown."""

Shape = tuple[Dimension, ...]
"""Ordered tensor axis sizes; an empty tuple represents rank zero."""


@dataclass(frozen=True, slots=True)
class TensorType:
    """A scalar element type and optional shape.

    A missing shape denotes unknown rank; an empty shape denotes rank zero.
    Known dimensions must be nonnegative; None dimensions have unknown size.
    """

    element_type: ScalarType
    # None represents an unranked tensor; () represents a rank-zero tensor.
    shape: Shape | None

    def __post_init__(self) -> None:
        """Reject negative known dimensions while allowing unknown sizes and rank."""
        if self.shape is None:
            return
        for dimension in self.shape:
            if dimension is None:
                continue
            if dimension < 0:
                raise ValueError("tensor dimensions must be nonnegative")

    @property
    def rank(self) -> int | None:
        """Return the number of axes, or None for an unranked tensor."""
        if self.shape is None:
            return None
        return len(self.shape)


Type = ScalarType | TensorType
"""A scalar or tensor type for an SSA value or global."""
