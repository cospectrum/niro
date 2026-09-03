"""MLIR integration for Niro."""

from niro.mlir.exporter import export_mlir
from niro.mlir.printer import format_mlir, write_mlir

__all__ = ["export_mlir", "format_mlir", "write_mlir"]
