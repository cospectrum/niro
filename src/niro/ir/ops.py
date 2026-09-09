"""Operations and their invariants in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass, field

from niro.ir.data import Attributes, Literal
from niro.ir.program import Region, SymbolName
from niro.ir.values import Value


@dataclass(frozen=True, slots=True)
class Const:
    result: Value
    literal: Literal


@dataclass(frozen=True, slots=True)
class GetGlobal:
    name: SymbolName
    result: Value


@dataclass(frozen=True, slots=True)
class Transpose:
    result: Value
    operand: Value
    permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Add:
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Mul:
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class MatMul:
    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Call:
    callee: SymbolName
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


@dataclass(frozen=True, slots=True)
class UnknownOp:
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
