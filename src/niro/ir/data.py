"""Literals and attributes stored in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

Literal = bool | int | float | bytes

AttributeName = str
type AttributeValue = (
    None | bool | int | float | str | bytes | tuple[AttributeValue, ...]
)
Attributes = dict[AttributeName, AttributeValue]
