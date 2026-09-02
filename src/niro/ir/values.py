"""Identifiers, values, and stored data in Niro IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from niro.ir.types import Type

FuncId = NewType("FuncId", int)
OpId = NewType("OpId", int)
ValueId = NewType("ValueId", int)


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("value ID cannot be negative")


type Attribute = None | bool | int | float | str | bytes | tuple[Attribute, ...]
type Literal = bool | int | float | bytes
