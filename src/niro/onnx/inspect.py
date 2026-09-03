"""Inspect ONNX models without importing their graph bodies."""

import onnx

from niro import ir
from niro.builder import ModuleBuilder

from .importer import _declare_entry_point


def inspect_signature(model: onnx.ModelProto) -> ir.Function:
    """Return the model's entry-point signature as a function declaration."""
    module = ModuleBuilder()
    return _declare_entry_point(model.graph, module).ir
