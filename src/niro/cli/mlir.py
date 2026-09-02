"""MLIR command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from google.protobuf.message import DecodeError

from niro.backends.mlir import lower_to_mlir, write_mlir
from niro.cli.input import InputFormat, load_model, resolve_input_format
from niro.frontends.onnx import import_onnx


def emit_mlir(
    input_path: Annotated[
        Path | None,
        typer.Argument(
            metavar="INPUT",
            help="Input model path, or stdin when omitted or '-'.",
        ),
    ] = None,
    input_format: Annotated[
        InputFormat | None,
        typer.Option(
            "--input-format",
            metavar="FORMAT",
            help="Format of the input model.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            metavar="OUTPUT",
            help="Output path, or stdout when omitted or '-'.",
        ),
    ] = None,
) -> None:
    """Emit textual MLIR for a model."""
    resolved_format = resolve_input_format(input_path, input_format)
    try:
        model = load_model(input_path, resolved_format)
        lowered = lower_to_mlir(import_onnx(model))
        destination = (
            sys.stdout
            if output_path is None or output_path == Path("-")
            else output_path
        )
        write_mlir(lowered, destination)
    except (DecodeError, OSError, TypeError, ValueError, NotImplementedError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
