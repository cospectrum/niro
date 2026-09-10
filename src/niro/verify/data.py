"""Shared literal checks for constants and global initializers."""

from __future__ import annotations

import math

from niro import ir


def _verify_literal(type_: ir.Type, literal: ir.Literal, *, context: str) -> None:
    match type_:
        case ir.ScalarType.BOOL if isinstance(literal, bool):
            pass
        case ir.ScalarType.I32 | ir.ScalarType.I64 if isinstance(
            literal, int
        ) and not isinstance(literal, bool):
            pass
        case ir.ScalarType.F32 | ir.ScalarType.F64 if isinstance(literal, float):
            pass
        case ir.TensorType(element_type, shape) if (
            isinstance(literal, bytes)
            and shape is not None
            and all(dimension is not None for dimension in shape)
        ):
            size = math.prod(dimension for dimension in shape if dimension is not None)
            expected = size * element_type.byte_width
            if len(literal) != expected:
                raise ValueError(
                    f"tensor {context} has {len(literal)} bytes, expected {expected}"
                )
        case ir.TensorType():
            raise TypeError(
                f"tensor {context} requires packed bytes and a static shape"
            )
        case _:
            raise TypeError(f"{context} value does not match its declared type")
