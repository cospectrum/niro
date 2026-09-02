"""Shared model input handling for CLI commands."""

from __future__ import annotations

import enum
from pathlib import Path

import onnx
import typer


class InputFormat(enum.StrEnum):
    ONNX = "onnx"


def resolve_input_format(
    input_path: Path | None, input_format: InputFormat | None
) -> InputFormat:
    if input_format is not None:
        return input_format
    if input_path is None or input_path == Path("-"):
        raise typer.BadParameter(
            "required when reading from stdin", param_hint="--input-format"
        )
    if input_path.suffix.lower() == ".onnx":
        return InputFormat.ONNX
    raise typer.BadParameter(
        f"cannot infer from {input_path.name!r}", param_hint="--input-format"
    )


def load_model(input_path: Path | None, input_format: InputFormat) -> onnx.ModelProto:
    match input_format:
        case InputFormat.ONNX:
            if input_path is None or input_path == Path("-"):
                return onnx.load_model_from_string(
                    typer.get_binary_stream("stdin").read()
                )
            return onnx.load(input_path)
