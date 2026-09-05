"""Format and write textual MLIR modules."""

from io import StringIO
from os import PathLike
from pathlib import Path
from typing import TextIO

from xdsl.dialects import builtin
from xdsl.printer import Printer


def format_mlir(mlir_module: builtin.ModuleOp) -> str:
    """Format an MLIR module as text."""
    stream = StringIO()
    _print_mlir(mlir_module, stream)
    return stream.getvalue()


def write_mlir(
    mlir_module: builtin.ModuleOp,
    destination: str | PathLike[str] | TextIO,
) -> None:
    """Write an MLIR module to a path or text stream."""
    if isinstance(destination, str | PathLike):
        with Path(destination).open("w", encoding="utf-8") as stream:
            _print_mlir(mlir_module, stream)
        return
    _print_mlir(mlir_module, destination)


def _print_mlir(module: builtin.ModuleOp, stream: TextIO) -> None:
    printer = Printer(stream=stream)
    printer.print_op(module)
    printer.print_metadata([builtin.Builtin])
