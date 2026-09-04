"""Operations in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from niro.ir.data import Attributes, Literal
from niro.ir.values import Value

if TYPE_CHECKING:
    from niro.ir.program import Region, SymbolName


type Op = (
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


@dataclass(frozen=True, slots=True)
class _BaseOp:
    def get_operands(self) -> tuple[Value, ...]:
        """Return the SSA values consumed by this operation."""
        return _get_operands(cast(Op, self))

    def get_results(self) -> tuple[Value, ...]:
        """Return the SSA values produced by this operation."""
        return _get_results(cast(Op, self))

    def is_terminator(self) -> bool:
        return _is_terminator(cast(Op, self))


@dataclass(frozen=True, slots=True)
class Const(_BaseOp):
    result: Value
    literal: Literal


@dataclass(frozen=True, slots=True)
class GetGlobal(_BaseOp):
    name: SymbolName
    result: Value


@dataclass(frozen=True, slots=True)
class Transpose(_BaseOp):
    result: Value
    operand: Value
    permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Add(_BaseOp):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Mul(_BaseOp):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class MatMul(_BaseOp):
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Call(_BaseOp):
    callee: SymbolName
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Return(_BaseOp):
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Yield(_BaseOp):
    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class If(_BaseOp):
    results: tuple[Value, ...]
    condition: Value
    then_region: Region
    else_region: Region


@dataclass(frozen=True, slots=True)
class UnknownOp(_BaseOp):
    """A structurally valid operation whose semantics are unknown to niro."""

    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: Attributes = field(default_factory=dict)


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


def _is_terminator(op: Op) -> bool:
    return isinstance(op, (Return, Yield))
