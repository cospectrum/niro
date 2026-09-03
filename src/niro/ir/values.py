"""Identifiers, values, and stored data in Niro IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from niro.ir.types import Type

ValueId = NewType("ValueId", int)
type SymbolName = str


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("value ID cannot be negative")


type AttributeName = str
type AttributeValue = (
    None | bool | int | float | str | bytes | tuple[AttributeValue, ...]
)
type Attributes = dict[AttributeName, AttributeValue]
type Literal = bool | int | float | bytes
