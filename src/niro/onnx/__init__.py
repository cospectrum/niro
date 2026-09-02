"""ONNX integration for Niro."""

from niro.onnx.importer import import_onnx, operation_name
from niro.onnx.op_type import OnnxOpType

__all__ = ["OnnxOpType", "import_onnx", "operation_name"]
