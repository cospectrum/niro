"""Core intermediate representation used by Niro."""

from niro.ir.ops import (
    Add,
    Call,
    Const,
    If,
    MatMul,
    Mul,
    Op,
    OpMixin,
    Return,
    Transpose,
    UnknownOp,
    Yield,
    matmul_result_type,
    transpose_result_type,
)
from niro.ir.program import Block, Function, FunctionType, Module, Region
from niro.ir.types import Dimension, ScalarType, Shape, TensorType, Type
from niro.ir.values import Attribute, FuncId, Literal, OpId, Value, ValueId

__all__ = [
    "Add",
    "Attribute",
    "Block",
    "Call",
    "Const",
    "Dimension",
    "FuncId",
    "Function",
    "FunctionType",
    "If",
    "Literal",
    "MatMul",
    "Module",
    "Mul",
    "Op",
    "OpId",
    "OpMixin",
    "Region",
    "Return",
    "ScalarType",
    "Shape",
    "TensorType",
    "Transpose",
    "Type",
    "UnknownOp",
    "Value",
    "ValueId",
    "Yield",
    "matmul_result_type",
    "transpose_result_type",
]
