"""SSA values in Niro IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

from typing import Annotated, NewType

from pydantic import Field
from pydantic.dataclasses import dataclass

from niro.ir.types import Type

ValueId = NewType("ValueId", int)


@dataclass(frozen=True, slots=True)
class Value:
    """A typed SSA value whose ID is unique within its function."""

    id: Annotated[ValueId, Field(ge=0)]
    type: Type
