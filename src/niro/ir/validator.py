"""Explicit validation of Niro IR modules."""

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
from niro.ir.program import Function, Global, Module, Region
from niro.ir.types import ScalarType, TensorType
from niro.ir.values import Value

__all__ = ["validate"]


def validate(module: Module) -> Module:
    """Validate existing IR invariants recursively and return the same module."""
    for global_ in module.globals:
        _validate_global(global_)
    for function in module.functions:
        if function.body is not None:
            _validate_region(function.body)
        _validate_function(function)
    _validate_module(module)
    return module


def _validate_region(region: Region) -> None:
    for block in region.blocks:
        for op in block.operations:
            if isinstance(op, If):
                _validate_region(op.then_region)
                _validate_region(op.else_region)
            if isinstance(op, Op):
                _validate_op(op)


def _validate_global(global_: Global) -> None:
    if not global_.name:
        raise ValueError("global name cannot be empty")


def _validate_module(module: Module) -> None:
    names = [global_.name for global_ in module.globals]
    names.extend(function.name for function in module.functions)
    if len(names) != len(set(names)):
        raise ValueError("module symbol names must be unique")


def _validate_function(function: Function) -> None:
    if not function.name:
        raise ValueError("function name cannot be empty")
    _validate_interface_names("input", function.input_names, len(function.type.inputs))
    _validate_interface_names(
        "output", function.output_names, len(function.type.outputs)
    )


def _validate_interface_names(
    kind: str,
    names: tuple[str | None, ...] | None,
    arity: int,
) -> None:
    if names is None:
        return
    if len(names) != arity:
        raise ValueError(f"{kind} names must match {kind} arity")
    if any(name == "" for name in names):
        raise ValueError(f"{kind} names cannot be empty")


def _validate_op(op: Op) -> None:
    result_ids = [val.id for val in op.get_results()]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("operation should produce unique values")

    match op:
        case Const():
            _validate_const(op)
        case GetGlobal(name=name):
            if not name:
                raise ValueError("global name cannot be empty")
        case Transpose():
            _validate_transpose(op)
        case Add():
            _validate_add(op)
        case Mul():
            _validate_mul(op)
        case MatMul():
            _validate_matmul(op)
        case Call() | Return() | Yield():
            pass
        case If(condition=condition):
            if condition.type is not ScalarType.BOOL:
                raise TypeError("if condition must be boolean")
        case UnknownOp(name=name):
            if not name:
                raise ValueError("UnknownOp name cannot be empty")
        case _ as unreachable:
            assert_never(unreachable)


def _validate_const(op: Const) -> None:
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


def _validate_transpose(op: Transpose) -> None:
    expected = transpose_result_type(op.operand.type, op.permutation)
    if op.result.type != expected:
        raise TypeError("transpose result type does not match its operands")


def _validate_add(op: Add) -> None:
    _require_matching_numeric_types("add", op.result, op.lhs, op.rhs)


def _validate_mul(op: Mul) -> None:
    _require_matching_numeric_types("mul", op.result, op.lhs, op.rhs)


def _validate_matmul(op: MatMul) -> None:
    if op.result.type != matmul_result_type(op.lhs.type, op.rhs.type):
        raise TypeError("matmul result type does not match its operands")


def _require_matching_numeric_types(
    operation: str, result: Value, lhs: Value, rhs: Value
) -> None:
    if lhs.type != rhs.type or result.type != lhs.type:
        raise TypeError(f"{operation} operands and result must have the same type")
    element_type = (
        lhs.type.element_type if isinstance(lhs.type, TensorType) else lhs.type
    )
    if element_type is ScalarType.BOOL:
        raise TypeError(f"{operation} does not support boolean values")
