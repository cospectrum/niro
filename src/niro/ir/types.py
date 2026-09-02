"""Types in the Niro intermediate representation."""

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


type Dimension = int | None
type Shape = tuple[Dimension, ...]


@dataclass(frozen=True, slots=True)
class TensorType:
    element_type: ScalarType
    # None represents an unranked tensor; () represents a rank-zero tensor.
    shape: Shape | None

    def __post_init__(self) -> None:
        _validate_tensor_type(self)


type Type = ScalarType | TensorType


def _validate_tensor_type(tensor_type: TensorType) -> None:
    if tensor_type.shape is None:
        return
    if any(dimension is not None and dimension < 0 for dimension in tensor_type.shape):
        raise ValueError("tensor dimensions cannot be negative")
