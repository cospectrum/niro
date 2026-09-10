"""Format and write textual MLIR modules."""

from io import StringIO
from os import PathLike
from pathlib import Path
from typing import TextIO

from xdsl.dialects import builtin
from xdsl.ir import Region
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


class _Printer(Printer):
    """Preserve CFG terminators omitted by xDSL's zero-result execute-region printer."""

    def print_region(
        self,
        region: Region,
        print_entry_block_args: bool = True,
        print_empty_block: bool = True,
        print_block_terminators: bool = True,
    ) -> None:
        """Print regions with explicit terminators whenever multiple blocks exist."""
        super().print_region(
            region,
            print_entry_block_args,
            print_empty_block,
            print_block_terminators or len(region.blocks) > 1,
        )


def _print_mlir(module: builtin.ModuleOp, stream: TextIO) -> None:
    """Write the operation and its builtin resource metadata to the text stream."""
    printer = _Printer(stream=stream)
    printer.print_op(module)
    printer.print_metadata([builtin.Builtin])
