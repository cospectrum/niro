"""Operations and their invariants in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass, field

from niro.ir.data import Attributes, Literal
from niro.ir.program import Block, Region, SymbolName
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
class TensorExtract:
    """Read an element from an immutable ranked tensor into a scalar SSA value.

    Supply one integer scalar index per axis; indices must be in bounds at
    execution. A rank-zero tensor needs no indices. The tensor is unchanged.
    """

    result: Value
    operand: Value
    indices: tuple[Value, ...] = ()


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
class Branch:
    """Jump to a block in the current region, passing its arguments.

    The target is identified by object identity. Arguments match its block
    arguments in number, order, and type. This terminator produces no results.
    """

    target: Block
    arguments: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class CondBranch:
    """Select a successor using a scalar boolean condition.

    Targets belong to the current region and are identified by object identity.
    Each argument tuple matches its target's block arguments in number, order,
    and type. Operand order is condition, true arguments, then false arguments.
    This terminator produces no results.
    """

    condition: Value
    true_target: Block
    false_target: Block
    true_arguments: tuple[Value, ...] = ()
    false_arguments: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Return:
    """End a function body with values matching the function's output types."""

    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class Yield:
    """End a nested region; If branches yield values matching the If result types."""

    operands: tuple[Value, ...] = ()


@dataclass(frozen=True, slots=True)
class If:
    """Select one branch using a scalar boolean condition.

    Produce zero or more results. Both regions contain exactly one argument-free
    block ending in [`Yield`][niro.ir.Yield] with the result types. Branches may
    capture values available before this operation. Return, Branch, and
    CondBranch cannot appear directly in either region.
    """

    results: tuple[Value, ...]
    condition: Value
    then_region: Region
    else_region: Region


@dataclass(frozen=True, slots=True)
class UnknownOp:
    """An opaque operation with attributes, owned regions, and CFG successors.

    Regions may capture enclosing values and contain SSA control flow ending in
    Yield or branches. Yield types and successor operand conventions are opaque.
    An operation with successors terminates its block; targets belong to the
    enclosing region. Empty regions are allowed.
    """

    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: Attributes = field(default_factory=dict)
    regions: tuple[Region, ...] = ()
    successors: tuple[Block, ...] = ()


Op = (
    Const
    | GetGlobal
    | Transpose
    | TensorExtract
    | Add
    | Mul
    | MatMul
    | Call
    | Branch
    | CondBranch
    | Return
    | Yield
    | If
    | UnknownOp
)
"""Closed union of the operation variants understood by Niro."""
