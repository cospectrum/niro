"""Operation-local checks, independent of surrounding blocks and symbols."""

from __future__ import annotations

import math
from typing import assert_never

from niro.ir.infer import matmul_result_type, transpose_result_type
from niro.ir.ops import (
    Add,
    Call,
    Const,
    GetGlobal,
    If,
    MatMul,
    Mul,
    Op,
    Return,
    Transpose,
    UnknownOp,
    Yield,
)
from niro.ir.types import ScalarType, TensorType


def _verify_op(op: Op) -> None:
    match op:
        case Const():
            _verify_const(op)
        case Add() | Mul():
            _verify_numeric_binary(op)
        case MatMul():
            expected = matmul_result_type(op.lhs.type, op.rhs.type)
            if op.result.type != expected:
                raise TypeError("matmul result type does not match its operands")
        case Transpose():
            expected = transpose_result_type(op.operand.type, op.permutation)
            if op.result.type != expected:
                raise TypeError("transpose result type does not match its operands")
        case If():
            if op.condition.type is not ScalarType.BOOL:
                raise TypeError("if condition must be boolean")
        case UnknownOp():
            if not op.name:
                raise ValueError("UnknownOp name cannot be empty")
        case Call() | GetGlobal() | Return() | Yield():
            pass
        case _ as unreachable:
            assert_never(unreachable)


def _verify_const(op: Const) -> None:
    match op.result.type:
        case ScalarType.BOOL if isinstance(op.literal, bool):
            pass
        case ScalarType.I32 | ScalarType.I64 if isinstance(
            op.literal, int
        ) and not isinstance(op.literal, bool):
            pass
        case ScalarType.F32 | ScalarType.F64 if isinstance(op.literal, float):
            pass
        case TensorType(element_type, shape) if (
            isinstance(op.literal, bytes)
            and shape is not None
            and all(dimension is not None for dimension in shape)
        ):
            size = math.prod(dimension for dimension in shape if dimension is not None)
            expected = size * element_type.byte_width
            if len(op.literal) != expected:
                raise ValueError(
                    f"tensor constant has {len(op.literal)} bytes, expected {expected}"
                )
        case TensorType():
            raise TypeError("tensor constant requires packed bytes and a static shape")
        case _:
            raise TypeError("constant value does not match its result type")


def _verify_numeric_binary(op: Add | Mul) -> None:
    name = "add" if isinstance(op, Add) else "mul"
    if op.lhs.type != op.rhs.type or op.result.type != op.lhs.type:
        raise TypeError(f"{name} operands and result must have the same type")
    element_type = (
        op.lhs.type.element_type if isinstance(op.lhs.type, TensorType) else op.lhs.type
    )
    if element_type is ScalarType.BOOL:
        raise TypeError(f"{name} does not support boolean values")
