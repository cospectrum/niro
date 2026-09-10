"""Program structure in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

from niro.ir.data import Attributes, Literal
from niro.ir.types import Type
from niro.ir.values import Value

if TYPE_CHECKING:
    from niro.ir.ops import Op

SymbolName = str


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
    """A function declaration or definition.

    A definition's entry block arguments represent its inputs and must have the
    types in ``type.inputs``. A declaration has no body or entry block.
    """

    name: SymbolName
    type: FunctionType
    body: Region | None = None

    input_names: tuple[str | None, ...] | None = None
    output_names: tuple[str | None, ...] | None = None
    attributes: Attributes = field(default_factory=dict)

    @property
    def first_block(self) -> Block | None:
        if not self.body:
            return None
        body = self.body
        if not body.blocks:
            return None
        return body.blocks[0]


@dataclass(slots=True)
class Global:
    """An immutable, initialized value in the module symbol table.

    The initializer must match the declared type, using the same literal rules
    as [`Const`][niro.ir.ops.Const]. Tensor initializers require a static shape
    and exactly its element count times the element byte width in packed bytes.
    [`niro.verify.module`][] checks every global, including unused globals.
    """

    name: SymbolName
    type: Type
    initializer: Literal
    attributes: Attributes = field(default_factory=dict)


@dataclass(slots=True)
class Module:
    functions: list[Function] = field(default_factory=list)
    globals: list[Global] = field(default_factory=list)
    attributes: Attributes = field(default_factory=dict)


VerifiedModule = NewType("VerifiedModule", Module)
"""A module that has passed [`niro.verify.module`][].

This is a static type marker, not an immutable snapshot. Verify again after
changing the module.
"""
