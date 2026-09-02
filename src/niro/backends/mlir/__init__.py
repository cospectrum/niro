"""MLIR backend for Niro."""

from niro.backends.mlir.lowering import lower_to_mlir
from niro.backends.mlir.output import format_mlir, write_mlir

__all__ = ["format_mlir", "lower_to_mlir", "write_mlir"]
