"""Base operation abstraction for Niro IR."""

from __future__ import annotations

from abc import ABC, abstractmethod

from niro.ir.values import Value


class Operation(ABC):
    """Runtime base class for every operation that can be stored in a block."""

    @abstractmethod
    def get_operands(self) -> tuple[Value, ...]:
        """Return the SSA values consumed by this operation."""
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> tuple[Value, ...]:
        """Return the SSA values produced by this operation."""
        raise NotImplementedError

    @abstractmethod
    def is_terminator(self) -> bool:
        """Return whether this operation terminates its containing block."""
        raise NotImplementedError
