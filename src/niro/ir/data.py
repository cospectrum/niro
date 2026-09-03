"""Literals and attributes stored in Niro IR."""

from __future__ import annotations

type Literal = bool | int | float | bytes

type AttributeName = str
type AttributeValue = (
    None | bool | int | float | str | bytes | tuple[AttributeValue, ...]
)
type Attributes = dict[AttributeName, AttributeValue]
