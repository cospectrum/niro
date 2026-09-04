"""SSA values in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from niro.ir.types import Type

ValueId = NewType("ValueId", int)


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: ValueId
    type: Type
