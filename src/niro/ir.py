"""Core intermediate representation used by niro.

This module contains only the data model. Construction conveniences belong in
the builder module, while validation and transformations can operate directly
on these classes.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import NewType

ValueId = NewType("ValueId", int)


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


# None represents an anonymous dynamic dimension.
type Dimension = int | None
type Shape = tuple[Dimension, ...]


@dataclass(frozen=True, slots=True)
class TensorType:
    element_type: ScalarType
    # None represents an unranked tensor; () represents a rank-zero tensor.
    shape: Shape | None

    def __post_init__(self) -> None:
        validate_tensor_type(self)


type Type = ScalarType | TensorType


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type

    def __post_init__(self) -> None:
        validate_value(self)


type Attribute = None | bool | int | float | str | bytes | tuple[Attribute, ...]
type Literal = bool | int | float | bytes


@dataclass(frozen=True, slots=True)
class Const:
    result: Value
    value: Literal

    def __post_init__(self) -> None:
        validate_const(self)


@dataclass(frozen=True, slots=True)
class Transpose:
    result: Value
    operand: Value
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_transpose(self)


@dataclass(frozen=True, slots=True)
class Add:
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        validate_add(self)


@dataclass(frozen=True, slots=True)
class Mul:
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        validate_mul(self)


@dataclass(frozen=True, slots=True)
class MatMul:
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        validate_matmul(self)


@dataclass(frozen=True, slots=True)
class Call:
    callee: str
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Return:
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Yield:
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class If:
    results: tuple[Value, ...]
    condition: Value
    then_region: Region
    else_region: Region

    def __post_init__(self) -> None:
        validate_if(self)


@dataclass(frozen=True, slots=True)
class UnknownOp:
    """A structurally valid operation whose semantics are unknown to niro."""

    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: dict[str, Attribute] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()

    def __post_init__(self) -> None:
        validate_unknown_op(self)


type Op = (
    Const | Transpose | Add | Mul | MatMul | Call | Return | Yield | If | UnknownOp
)


@dataclass(slots=True)
class Block:
    arguments: tuple[Value, ...] = ()
    operations: list[Op] = field(default_factory=list)


@dataclass(slots=True)
class Region:
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FunctionType:
    inputs: tuple[Type, ...]
    outputs: tuple[Type, ...]


@dataclass(slots=True)
class Function:
    name: str
    type: FunctionType
    # None denotes an external declaration.
    body: Region | None = None
    attributes: dict[str, Attribute] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_function(self)

    @property
    def arguments(self) -> tuple[Value, ...]:
        """Return the entry block arguments of a defined function."""
        if self.body is None or not self.body.blocks:
            return ()
        return self.body.blocks[0].arguments


@dataclass(slots=True)
class Module:
    functions: list[Function] = field(default_factory=list)
    attributes: dict[str, Attribute] = field(default_factory=dict)


def transpose_result_type(
    operand_type: Type, permutation: tuple[int, ...]
) -> TensorType:
    if not isinstance(operand_type, TensorType):
        raise TypeError("transpose operand must be a tensor")
    if operand_type.shape is None:
        return operand_type
    if sorted(permutation) != list(range(len(operand_type.shape))):
        raise ValueError("transpose permutation must contain every dimension once")
    return TensorType(
        operand_type.element_type,
        tuple(operand_type.shape[index] for index in permutation),
    )


def matmul_result_type(lhs: Type, rhs: Type) -> TensorType:
    if not isinstance(lhs, TensorType) or not isinstance(rhs, TensorType):
        raise TypeError("matmul operands must be tensors")
    if lhs.shape is None or rhs.shape is None:
        raise TypeError("matmul operands must be ranked tensors")
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise TypeError("matmul operands must be rank-two tensors")
    if lhs.element_type is not rhs.element_type:
        raise TypeError("matmul operand element types must match")
    lhs_inner, rhs_inner = lhs.shape[1], rhs.shape[0]
    if lhs_inner is not None and rhs_inner is not None and lhs_inner != rhs_inner:
        raise ValueError("matmul contracting dimensions must match")
    return TensorType(lhs.element_type, (lhs.shape[0], rhs.shape[1]))


def validate_tensor_type(tensor_type: TensorType) -> None:
    if tensor_type.shape is None:
        return
    if any(dimension is not None and dimension < 0 for dimension in tensor_type.shape):
        raise ValueError("tensor dimensions cannot be negative")


def validate_value(value: Value) -> None:
    if value.id < 0:
        raise ValueError("value ID cannot be negative")


def validate_const(op: Const) -> None:
    match op.result.type:
        case ScalarType.BOOL if isinstance(op.value, bool):
            pass
        case ScalarType.I32 | ScalarType.I64 if isinstance(
            op.value, int
        ) and not isinstance(op.value, bool):
            pass
        case ScalarType.F32 | ScalarType.F64 if isinstance(op.value, float):
            pass
        case TensorType(element_type, shape) if (
            isinstance(op.value, bytes)
            and shape is not None
            and all(dimension is not None for dimension in shape)
        ):
            size = math.prod(dimension for dimension in shape if dimension is not None)
            expected = size * element_type.byte_width
            if len(op.value) != expected:
                raise ValueError(
                    f"tensor constant has {len(op.value)} bytes, expected {expected}"
                )
        case TensorType():
            raise TypeError("tensor constant requires packed bytes and a static shape")
        case _:
            raise TypeError("constant value does not match its result type")


def validate_transpose(op: Transpose) -> None:
    expected = transpose_result_type(op.operand.type, op.permutation)
    if op.result.type != expected:
        raise TypeError("transpose result type does not match its operands")


def validate_add(op: Add) -> None:
    _require_matching_numeric_types("add", op.result, op.lhs, op.rhs)


def validate_mul(op: Mul) -> None:
    _require_matching_numeric_types("mul", op.result, op.lhs, op.rhs)


def validate_matmul(op: MatMul) -> None:
    if op.result.type != matmul_result_type(op.lhs.type, op.rhs.type):
        raise TypeError("matmul result type does not match its operands")


def validate_if(op: If) -> None:
    if op.condition.type is not ScalarType.BOOL:
        raise TypeError("if condition must be boolean")


def validate_unknown_op(op: UnknownOp) -> None:
    if not op.name:
        raise ValueError("unknown operation name cannot be empty")


def validate_function(function: Function) -> None:
    if not function.name:
        raise ValueError("function name cannot be empty")


def _require_matching_numeric_types(
    operation: str,
    result: Value,
    lhs: Value,
    rhs: Value,
) -> None:
    if lhs.type != rhs.type or result.type != lhs.type:
        raise TypeError(f"{operation} operands and result must have the same type")
    element_type = (
        lhs.type.element_type if isinstance(lhs.type, TensorType) else lhs.type
    )
    if element_type is ScalarType.BOOL:
        raise TypeError(f"{operation} does not support boolean values")
