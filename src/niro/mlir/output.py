"""Text output for MLIR modules."""

from io import StringIO
from os import PathLike
from pathlib import Path
from typing import TextIO

from xdsl.dialects import builtin
from xdsl.printer import Printer


def format_mlir(module: builtin.ModuleOp) -> str:
    """Format an MLIR module as text."""
    stream = StringIO()
    _print_mlir(module, stream)
    return stream.getvalue()


def write_mlir(
    module: builtin.ModuleOp,
    destination: str | PathLike[str] | TextIO,
) -> None:
    """Write an MLIR module to a path or text stream."""
    if isinstance(destination, str | PathLike):
        with Path(destination).open("w", encoding="utf-8") as stream:
            _print_mlir(module, stream)
        return
    _print_mlir(module, destination)


def _print_mlir(module: builtin.ModuleOp, stream: TextIO) -> None:
    Printer(stream=stream).print_op(module)
