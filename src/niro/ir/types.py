"""Types in the Niro intermediate representation.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ScalarType(enum.Enum):
    BOOL = "bool"
    I32 = "i32"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"

    @property
    def byte_width(self) -> int:
        return {
            ScalarType.BOOL: 1,
            ScalarType.I32: 4,
            ScalarType.I64: 8,
            ScalarType.F32: 4,
            ScalarType.F64: 8,
        }[self]


Dimension = int | None
Shape = tuple[Dimension, ...]


@dataclass(frozen=True, slots=True)
class TensorType:
    element_type: ScalarType
    # None represents an unranked tensor; () represents a rank-zero tensor.
    shape: Shape | None

    def __post_init__(self) -> None:
        if self.shape is not None:
            for dimension in self.shape:
                if dimension is None:
                    continue
                if dimension < 0:
                    raise ValueError("tensor dimensions must be nonnegative")

    @property
    def rank(self) -> int | None:
        if self.shape is None:
            return None
        return len(self.shape)


Type = ScalarType | TensorType
