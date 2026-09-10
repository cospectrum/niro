from collections.abc import Iterable

from niro import ir

OnnxValueName = str
"""An ONNX graph value name; registered names must be nonempty and unique."""


class OnnxValueTable:
    """Map unique, nonempty ONNX graph value names to their Niro SSA values."""

    def __init__(self) -> None:
        """Create an empty value table."""
        self._values: dict[OnnxValueName, ir.Value] = {}

    def __contains__(self, name: object) -> bool:
        """Return whether the name has a registered SSA value."""
        return name in self._values

    def define(self, name: OnnxValueName, value: ir.Value) -> None:
        """Register a value, rejecting empty names and duplicate definitions."""
        if not name:
            raise ValueError("ONNX value name cannot be empty")
        if name in self._values:
            raise ValueError(f"ONNX value is already defined: {name!r}")
        self._values[name] = value

    def define_many(
        self,
        names_: Iterable[OnnxValueName],
        values_: Iterable[ir.Value],
    ) -> None:
        """Register equal-length name and value sequences in order.

        Earlier definitions remain registered if a later name is invalid.
        """
        names = tuple(names_)
        values = tuple(values_)
        assert len(names) == len(values)
        for name, value in zip(names, values, strict=True):
            self.define(name, value)

    def lookup(self, name: OnnxValueName) -> ir.Value:
        """Return the registered SSA value, raising `ValueError` for unknown names."""
        try:
            return self._values[name]
        except KeyError:
            raise ValueError(f"unknown ONNX value: {name!r}") from None
