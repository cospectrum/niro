"""MLIR command-line interface."""

from __future__ import annotations

import enum
from io import StringIO
from pathlib import Path
from typing import Annotated

import onnx
import typer
from google.protobuf.message import DecodeError
from xdsl.printer import Printer

from niro.backends.mlir import lower_to_mlir
from niro.frontends.onnx import import_onnx


class InputFormat(enum.StrEnum):
    ONNX = "onnx"


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
    resolved_format = _resolve_input_format(input_path, input_format)
    try:
        model = _load_model(input_path, resolved_format)
        lowered = lower_to_mlir(import_onnx(model))
        stream = StringIO()
        Printer(stream=stream).print_op(lowered)
        _write_output(output_path, stream.getvalue())
    except (DecodeError, OSError, TypeError, ValueError, NotImplementedError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _resolve_input_format(
    input_path: Path | None,
    input_format: InputFormat | None,
) -> InputFormat:
    if input_format is not None:
        return input_format
    if input_path is None or input_path == Path("-"):
        raise typer.BadParameter(
            "required when reading from stdin",
            param_hint="--input-format",
        )
    if input_path.suffix.lower() == ".onnx":
        return InputFormat.ONNX
    raise typer.BadParameter(
        f"cannot infer from {input_path.name!r}",
        param_hint="--input-format",
    )


def _load_model(
    input_path: Path | None,
    input_format: InputFormat,
) -> onnx.ModelProto:
    match input_format:
        case InputFormat.ONNX:
            if input_path is None or input_path == Path("-"):
                data = typer.get_binary_stream("stdin").read()
                return onnx.load_model_from_string(data)
            return onnx.load(input_path)


def _write_output(output_path: Path | None, output: str) -> None:
    if output_path is None or output_path == Path("-"):
        typer.echo(output, nl=False)
        return
    output_path.write_text(output, encoding="utf-8")
