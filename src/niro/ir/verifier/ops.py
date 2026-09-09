"""Operation-local checks, independent of surrounding blocks and symbols."""

from __future__ import annotations

import math
from typing import assert_never

from niro import ir
from niro.ir.ops import (
    Add,
    Const,
    Mul,
    Op,
)


def _verify_op(op: Op) -> None:
    match op:
        case ir.Const():
            _verify_const(op)
        case ir.Add() | ir.Mul():
            _verify_numeric_binary(op)
        case ir.MatMul():
            expected = ir.infer.matmul_result_type(op.lhs.type, op.rhs.type)
            if op.result.type != expected:
                raise TypeError("matmul result type does not match its operands")
        case ir.Transpose():
            expected = ir.infer.transpose_result_type(op.operand.type, op.permutation)
            if op.result.type != expected:
                raise TypeError("transpose result type does not match its operands")
        case ir.If():
            if op.condition.type is not ir.ScalarType.BOOL:
                raise TypeError("if condition must be boolean")
        case ir.UnknownOp():
            if not op.name:
                raise ValueError("UnknownOp name cannot be empty")
        case ir.Call() | ir.GetGlobal() | ir.Return() | ir.Yield():
            pass
        case _ as unreachable:
            assert_never(unreachable)


def _verify_const(op: Const) -> None:
    match op.result.type:
        case ir.ScalarType.BOOL if isinstance(op.literal, bool):
            pass
        case ir.ScalarType.I32 | ir.ScalarType.I64 if isinstance(
            op.literal, int
        ) and not isinstance(op.literal, bool):
            pass
        case ir.ScalarType.F32 | ir.ScalarType.F64 if isinstance(op.literal, float):
            pass
        case ir.TensorType(element_type, shape) if (
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
        case ir.TensorType():
            raise TypeError("tensor constant requires packed bytes and a static shape")
        case _:
            raise TypeError("constant value does not match its result type")


def _verify_numeric_binary(op: Add | Mul) -> None:
    name = "add" if isinstance(op, ir.Add) else "mul"
    if op.lhs.type != op.rhs.type or op.result.type != op.lhs.type:
        raise TypeError(f"{name} operands and result must have the same type")
    element_type = (
        op.lhs.type.element_type
        if isinstance(op.lhs.type, ir.TensorType)
        else op.lhs.type
    )
    if element_type is ir.ScalarType.BOOL:
        raise TypeError(f"{name} does not support boolean values")
