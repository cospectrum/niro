"""Operations and their invariants in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import field
from typing import cast

from pydantic.dataclasses import dataclass

from niro.ir.data import Attributes, Literal
from niro.ir.operation import Operation
from niro.ir.program import Region, SymbolName
from niro.ir.values import Value


@dataclass(frozen=True, slots=True)
class _BuiltinOperation(Operation):
    def get_operands(self) -> tuple[Value, ...]:
        return _get_operands(cast("Op", self))

    def get_results(self) -> tuple[Value, ...]:
        return _get_results(cast("Op", self))


@dataclass(frozen=True, slots=True)
class Const(_BuiltinOperation):
    result: Value
    literal: Literal


@dataclass(frozen=True, slots=True)
class GetGlobal(_BuiltinOperation):
    name: SymbolName
    result: Value


@dataclass(frozen=True, slots=True)
class Transpose(_BuiltinOperation):
    result: Value
    operand: Value
    permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Add(_BuiltinOperation):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Mul(_BuiltinOperation):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class MatMul(_BuiltinOperation):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Call(_BuiltinOperation):
    callee: SymbolName
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Return(_BuiltinOperation):
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Yield(_BuiltinOperation):
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class If(_BuiltinOperation):
    results: tuple[Value, ...]
    condition: Value
    then_region: Region
    else_region: Region


@dataclass(frozen=True, slots=True)
class UnknownOp(_BuiltinOperation):
    """A structurally valid operation whose semantics are unknown to niro."""

    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: Attributes = field(default_factory=dict)


Op = (
    Const
    | GetGlobal
    | Transpose
    | Add
    | Mul
    | MatMul
    | Call
    | Return
    | Yield
    | If
    | UnknownOp
)
"""Closed union of the operation variants understood by Niro."""


def as_op(operation: Operation) -> Op:
    """Return a built-in operation, rejecting extension operations."""
    if not isinstance(operation, Op):
        raise TypeError(f"{type(operation).__name__} is not a built-in Niro operation")
    return operation


def _get_operands(op: Op) -> tuple[Value, ...]:
    match op:
        case Const() | GetGlobal():
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


def _get_results(op: Op) -> tuple[Value, ...]:
    match op:
        case (
            Const(result=result)
            | GetGlobal(result=result)
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
