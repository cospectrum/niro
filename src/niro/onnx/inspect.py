"""Inspect ONNX models without importing their graph bodies."""

import onnx

from niro import builder
from niro.ir import Function
from niro.onnx import _from_onnx


def inspect_signature(model: onnx.ModelProto) -> Function:
    """Return the model's entry-point signature as a function declaration."""
    module = builder.ModuleBuilder()
    return _from_onnx._declare_entry_point(model.graph, module).raw
