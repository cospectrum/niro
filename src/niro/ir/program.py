"""Program structure in Niro IR."""

from __future__ import annotations

from dataclasses import dataclass, field

from niro.ir.ops import Op
from niro.ir.types import Type
from niro.ir.values import Attribute, FuncId, Value


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
    id: FuncId
    name: str
    type: FunctionType
    # None denotes an external declaration.
    body: Region | None = None
    attributes: dict[str, Attribute] = field(default_factory=dict)
    input_names: tuple[str | None, ...] | None = None
    output_names: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        _validate_function(self)

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


def _validate_function(function: Function) -> None:
    if function.id < 0:
        raise ValueError("function ID cannot be negative")
    if not function.name:
        raise ValueError("function name cannot be empty")
    _validate_interface_names("input", function.input_names, len(function.type.inputs))
    _validate_interface_names(
        "output", function.output_names, len(function.type.outputs)
    )


def _validate_interface_names(
    kind: str, names: tuple[str | None, ...] | None, arity: int
) -> None:
    if names is None:
        return
    if len(names) != arity:
        raise ValueError(f"{kind} names must match {kind} arity")
    if any(name == "" for name in names):
        raise ValueError(f"{kind} names cannot be empty")
