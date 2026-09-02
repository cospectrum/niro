"""Operations and their invariants in Niro IR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never, cast

from niro.ir.types import ScalarType, TensorType, Type
from niro.ir.values import Attribute, Literal, OpId, Value

if TYPE_CHECKING:
    from niro.ir.program import Region


class _OpMixin:
    """Common, immutable queries supported by every operation."""

    def get_operands(self) -> tuple[Value, ...]:
        """Return the SSA values consumed by this operation."""
        return _get_operands(cast("Op", self))

    def get_results(self) -> tuple[Value, ...]:
        """Return the SSA values produced by this operation."""
        return _get_results(cast("Op", self))


@dataclass(frozen=True, slots=True)
class Const(_OpMixin):
    id: OpId
    result: Value
    literal: Literal

    def __post_init__(self) -> None:
        _validate_const(self)


@dataclass(frozen=True, slots=True)
class Transpose(_OpMixin):
    id: OpId
    result: Value
    operand: Value
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_transpose(self)


@dataclass(frozen=True, slots=True)
class Add(_OpMixin):
    id: OpId
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        _validate_add(self)


@dataclass(frozen=True, slots=True)
class Mul(_OpMixin):
    id: OpId
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        _validate_mul(self)


@dataclass(frozen=True, slots=True)
class MatMul(_OpMixin):
    id: OpId
    result: Value
    lhs: Value
    rhs: Value

    def __post_init__(self) -> None:
        _validate_matmul(self)


@dataclass(frozen=True, slots=True)
class Call(_OpMixin):
    id: OpId
    callee: str
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]

    def __post_init__(self) -> None:
        _validate_op_id(self.id)


@dataclass(frozen=True, slots=True)
class Return(_OpMixin):
    id: OpId
    operands: tuple[Value, ...] = ()

    def __post_init__(self) -> None:
        _validate_op_id(self.id)


@dataclass(frozen=True, slots=True)
class Yield(_OpMixin):
    id: OpId
    operands: tuple[Value, ...] = ()

    def __post_init__(self) -> None:
        _validate_op_id(self.id)


@dataclass(frozen=True, slots=True)
class If(_OpMixin):
    id: OpId
    results: tuple[Value, ...]
    condition: Value
    then_region: Region
    else_region: Region

    def __post_init__(self) -> None:
        _validate_op_id(self.id)
        if self.condition.type is not ScalarType.BOOL:
            raise TypeError("if condition must be boolean")


@dataclass(frozen=True, slots=True)
class UnknownOp(_OpMixin):
    """A structurally valid operation whose semantics are unknown to niro."""

    id: OpId
    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: dict[str, Attribute] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()

    def __post_init__(self) -> None:
        _validate_op_id(self.id)
        if not self.name:
            raise ValueError("unknown operation name cannot be empty")


type Op = (
    Const | Transpose | Add | Mul | MatMul | Call | Return | Yield | If | UnknownOp
)


def _get_operands(op: Op) -> tuple[Value, ...]:
    match op:
        case Const():
            return ()
        case Transpose(operand=operand):
            return (operand,)
        case Add(lhs=lhs, rhs=rhs) | Mul(lhs=lhs, rhs=rhs) | MatMul(lhs=lhs, rhs=rhs):
            return lhs, rhs
        case Call(arguments=arguments):
            return arguments
        case Return(operands=operands) | Yield(operands=operands):
            return operands
        case If(condition=condition):
            return (condition,)
        case UnknownOp(operands=operands):
            return operands
        case _ as unreachable:
            assert_never(unreachable)


def _get_results(op: Op) -> tuple[Value, ...]:
    match op:
        case (
            Const(result=result)
            | Transpose(result=result)
            | Add(result=result)
            | Mul(result=result)
            | MatMul(result=result)
        ):
            return (result,)
        case Call(results=results) | If(results=results) | UnknownOp(results=results):
            return results
        case Return() | Yield():
            return ()
        case _ as unreachable:
            assert_never(unreachable)


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


def _validate_const(op: Const) -> None:
    _validate_op_id(op.id)
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
    _validate_op_id(op.id)
    expected = transpose_result_type(op.operand.type, op.permutation)
    if op.result.type != expected:
        raise TypeError("transpose result type does not match its operands")


def _validate_add(op: Add) -> None:
    _validate_op_id(op.id)
    _require_matching_numeric_types("add", op.result, op.lhs, op.rhs)


def _validate_mul(op: Mul) -> None:
    _validate_op_id(op.id)
    _require_matching_numeric_types("mul", op.result, op.lhs, op.rhs)


def _validate_matmul(op: MatMul) -> None:
    _validate_op_id(op.id)
    if op.result.type != matmul_result_type(op.lhs.type, op.rhs.type):
        raise TypeError("matmul result type does not match its operands")


def _validate_op_id(op_id: OpId) -> None:
    if op_id < 0:
        raise ValueError("operation ID cannot be negative")


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
