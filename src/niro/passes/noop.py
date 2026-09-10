"""A pass that leaves the module unchanged."""

from niro.ir import VerifiedModule

__all__ = ["noop"]


def noop(module: VerifiedModule) -> VerifiedModule:
    """Return the input module unchanged."""
    return module
