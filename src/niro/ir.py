"""Core intermediate representation used by niro.

This module contains only the data model. Construction conveniences belong in
the builder module, while validation and transformations can operate directly
on these classes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import NewType

ValueId = NewType("ValueId", int)


class ScalarType(enum.Enum):
    BOOL = "bool"
    I32 = "i32"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"


# None represents an anonymous dynamic dimension.
type Dimension = int | None
type Shape = tuple[Dimension, ...]


@dataclass(frozen=True, slots=True)
class TensorType:
    element_type: ScalarType
    # None represents an unranked tensor; () represents a rank-zero tensor.
    shape: Shape | None


type Type = ScalarType | TensorType


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type


type Attribute = None | bool | int | float | str | bytes | tuple[Attribute, ...]
type Literal = bool | int | float | bytes


@dataclass(frozen=True, slots=True)
class Const:
    result: Value
    value: Literal


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


@dataclass(frozen=True, slots=True)
class UnknownOp:
    """A structurally valid operation whose semantics are unknown to niro."""

    name: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attributes: dict[str, Attribute] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()


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
