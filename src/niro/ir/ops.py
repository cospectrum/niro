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
    """An SSA constant whose literal matches its result type.

    Tensor literals use packed bytes and require a static shape with the
    corresponding byte count. Scalar literals use bool, int, or float.
    """

    result: Value
    literal: Literal


@dataclass(frozen=True, slots=True)
class GetGlobal:
    """Read a [`Global`][niro.ir.Global] into a result of its declared type."""

    name: SymbolName
    result: Value


@dataclass(frozen=True, slots=True)
class Transpose:
    """Permute a tensor's axes without changing its element type.

    For a ranked operand, the permutation contains each axis exactly once
    and determines the result shape; unranked operands remain unranked.
    """

    result: Value
    operand: Value
    permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Add:
    """Add numeric scalars or tensors elementwise, with identical operand/result types."""

    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Mul:
    """Multiply numeric scalars or tensors elementwise, with identical operand/result types."""

    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class MatMul:
    """Multiply rank-two tensors with matching element types.

    Known contracting dimensions must match. The result shape consists of
    the left row dimension and right column dimension.
    """

    result: Value
    lhs: Value
    rhs: Value


@dataclass(frozen=True, slots=True)
class Call:
    """Invoke a module function with arguments and results matching its signature."""

    callee: SymbolName
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class Return:
    """End a function body with values matching the function's output types."""

    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Yield:
    """End an [`If`][niro.ir.If] branch with values matching its result types."""

    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class If:
    """Select one branch using a scalar boolean condition.

    Each branch has one argument-free block ending in [`Yield`][niro.ir.Yield]
    with the result types. Branches may capture values available before this
    operation.
    """

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
