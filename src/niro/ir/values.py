"""SSA values in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from typing import NewType

from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from niro.ir.types import Type

ValueId = NewType("ValueId", NonNegativeInt)


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type
