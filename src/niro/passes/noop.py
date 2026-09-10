"""A pass that leaves the indexed module unchanged."""

from niro.index import ModuleIndex

__all__ = ["noop"]


def noop(indexed: ModuleIndex) -> ModuleIndex:
    """Return the input index unchanged, preserving the module and all lookup tables."""
    return indexed
