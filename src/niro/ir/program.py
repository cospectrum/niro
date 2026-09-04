"""Program structure in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass, field

from niro.ir.data import Attributes, Literal
from niro.ir.ops import Op
from niro.ir.types import Type
from niro.ir.values import Value

type SymbolName = str


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
    """An immutable, initialized value in the module symbol table."""

    name: SymbolName
    type: Type
    initializer: Literal
    attributes: Attributes = field(default_factory=dict)


@dataclass(slots=True)
class Module:
    functions: list[Function] = field(default_factory=list)
    globals: list[Global] = field(default_factory=list)
    attributes: Attributes = field(default_factory=dict)
