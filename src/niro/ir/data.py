"""Literals and attributes stored in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

Literal = bool | int | float | bytes
"""A scalar constant value or packed tensor bytes, interpreted by its IR type."""

AttributeName = str
"""A string key identifying operation, function, global, or module metadata."""

type AttributeValue = (
    None | bool | int | float | str | bytes | tuple[AttributeValue, ...]
)
"""A metadata value, optionally containing recursively nested tuples."""

Attributes = dict[AttributeName, AttributeValue]
"""A mutable mapping from metadata names to attribute values."""
