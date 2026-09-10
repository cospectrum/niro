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

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("value ID must be nonnegative")


@dataclass(slots=True)
class ValueSupply:
    """A mutable allocator shared by builders and rewrite helpers.

    Use one supply for all definitions in a function, including nested regions.
    For an existing function, initialize it with
    [`niro.rewrite.value_supply`][]. Allocating a value does not define it: place
    it in an operation's results or a block's arguments.

    Attributes:
        next_id: Next unused function-local value ID.
    """

    next_id: int = 0

    def __post_init__(self) -> None:
        if self.next_id < 0:
            raise ValueError("next value ID must be nonnegative")

    def fresh(self, type: Type) -> Value:
        """Allocate a typed value and advance this supply.

        Examples:
            ```python
            from niro import ir

            supply = ir.ValueSupply(10)
            first = supply.fresh(ir.ScalarType.I32)
            second = supply.fresh(ir.ScalarType.F32)
            assert (first.id, second.id, supply.next_id) == (10, 11, 12)
            ```
        """
        value = Value(ValueId(self.next_id), type)
        self.next_id += 1
        return value

    def clone(self) -> ValueSupply:
        """Copy the current allocation position into an independent supply.

        Clones produce overlapping IDs. Use them for alternative rewrites, not
        to allocate fragments that will coexist in the same function.

        Examples:
            ```python
            from niro import ir

            supply = ir.ValueSupply(10)
            trial = supply.clone()
            value = trial.fresh(ir.ScalarType.I32)
            assert value.id == 10
            assert supply.next_id == 10
            assert trial.next_id == 11
            ```
        """
        return ValueSupply(self.next_id)
