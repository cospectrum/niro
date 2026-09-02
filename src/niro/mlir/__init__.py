"""MLIR integration for Niro."""

from niro.mlir.lowering import lower_to_mlir
from niro.mlir.output import format_mlir, write_mlir

__all__ = ["format_mlir", "lower_to_mlir", "write_mlir"]
