"""Niro IR is a strongly typed, SSA-based representation composed of types,
values, program structure, operations, literals, and attributes.
"""

from niro.ir.data import AttributeName, Attributes, AttributeValue, Literal
from niro.ir.operation import Operation
from niro.ir.ops import (
    Add,
    Call,
    Const,
    GetGlobal,
    If,
    MatMul,
    Mul,
    Op,
    Return,
    Transpose,
    UnknownOp,
    Yield,
    as_op,
    matmul_result_type,
    transpose_result_type,
)
from niro.ir.program import (
    Block,
    Function,
    FunctionType,
    Global,
    Module,
    Region,
    SymbolName,
)
from niro.ir.types import Dimension, ScalarType, Shape, TensorType, Type
from niro.ir.values import Value, ValueId

__all__ = [
    "Add",
    "AttributeName",
    "AttributeValue",
    "Attributes",
    "Block",
    "Call",
    "Const",
    "Dimension",
    "Function",
    "FunctionType",
    "GetGlobal",
    "Global",
    "If",
    "Literal",
    "MatMul",
    "Module",
    "Mul",
    "Op",
    "Operation",
    "Region",
    "Return",
    "ScalarType",
    "Shape",
    "SymbolName",
    "TensorType",
    "Transpose",
    "Type",
    "UnknownOp",
    "Value",
    "ValueId",
    "Yield",
    "as_op",
    "matmul_result_type",
    "transpose_result_type",
]
