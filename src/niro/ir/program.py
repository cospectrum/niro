"""Program structure in Niro IR."""

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

    def __post_init__(self) -> None:
        _validate_function(self)

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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("global name cannot be empty")


@dataclass(slots=True)
class Module:
    functions: list[Function] = field(default_factory=list)
    globals: list[Global] = field(default_factory=list)
    attributes: Attributes = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [global_.name for global_ in self.globals]
        names.extend(function.name for function in self.functions)
        if len(names) != len(set(names)):
            raise ValueError("module symbol names must be unique")


def _validate_function(function: Function) -> None:
    if not function.name:
        raise ValueError("function name cannot be empty")
    _validate_interface_names("input", function.input_names, len(function.type.inputs))
    _validate_interface_names(
        "output", function.output_names, len(function.type.outputs)
    )


def _validate_interface_names(
    kind: str,
    names: tuple[str | None, ...] | None,
    arity: int,
) -> None:
    if names is None:
        return
    if len(names) != arity:
        raise ValueError(f"{kind} names must match {kind} arity")
    if any(name == "" for name in names):
        raise ValueError(f"{kind} names cannot be empty")
