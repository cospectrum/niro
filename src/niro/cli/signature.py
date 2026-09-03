"""Entry-point signature inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from google.protobuf.message import DecodeError

from niro import ir
from niro.cli.input import InputFormat, load_model, resolve_input_format
from niro.onnx import inspect_signature as inspect_onnx_signature

LINE_WIDTH = 100


def inspect_signature(
    input_path: Annotated[
        Path | None,
        typer.Argument(
            metavar="INPUT", help="Input model path, or stdin when omitted or '-'."
        ),
    ] = None,
    input_format: Annotated[
        InputFormat | None,
        typer.Option(
            "--input-format", metavar="FORMAT", help="Format of the input model."
        ),
    ] = None,
) -> None:
    """Print the model entry-point signature."""
    resolved_format = resolve_input_format(input_path, input_format)
    try:
        model = load_model(input_path, resolved_format)
        typer.echo(format_signature(inspect_onnx_signature(model)))
    except (DecodeError, OSError, TypeError, ValueError, NotImplementedError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def format_signature(function: ir.Function) -> str:
    """Format a public function interface with a 100-column target."""
    inputs = _items(function.type.inputs, function.input_names)
    outputs = _items(function.type.outputs, function.output_names)
    input_inline = f"{function.name}({', '.join(inputs)})"
    output_inline = f" -> ({', '.join(outputs)})"
    complete = input_inline + output_inline
    if len(complete) <= LINE_WIDTH:
        return complete

    expand_inputs = len(input_inline) > LINE_WIDTH or not outputs
    if not expand_inputs:
        return _expanded_outputs(input_inline, outputs)

    input_lines = [f"{function.name}(", *[f"  {item}," for item in inputs]]
    if len(")" + output_inline) <= LINE_WIDTH:
        return "\n".join([*input_lines, ")" + output_inline])
    return "\n".join([*input_lines, ") -> (", *[f"  {item}," for item in outputs], ")"])


def _expanded_outputs(prefix: str, outputs: list[str]) -> str:
    return "\n".join([f"{prefix} -> (", *[f"  {item}," for item in outputs], ")"])


def _items(
    types: tuple[ir.Type, ...], names: tuple[str | None, ...] | None
) -> list[str]:
    resolved_names = names if names is not None else (None,) * len(types)
    return [
        f"{name}: {_format_type(value_type)}"
        if name is not None
        else _format_type(value_type)
        for name, value_type in zip(resolved_names, types, strict=True)
    ]


def _format_type(value_type: ir.Type) -> str:
    if isinstance(value_type, ir.ScalarType):
        return value_type.value
    if value_type.shape is None:
        dimensions = "*x"
    elif not value_type.shape:
        dimensions = ""
    else:
        dimensions = "".join(
            f"{'?' if dimension is None else dimension}x"
            for dimension in value_type.shape
        )
    return f"tensor<{dimensions}{value_type.element_type.value}>"
