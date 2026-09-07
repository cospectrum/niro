"""MLIR integration for Niro."""

from niro.mlir.printer import format_mlir, write_mlir
from niro.mlir.to_mlir import to_mlir

__all__ = ["format_mlir", "to_mlir", "write_mlir"]
