"""Niro IR is a strongly typed, SSA-based representation composed of types,
values, program structure, operations, literals, and attributes.
"""

from niro.ir import infer
from niro.ir.builder import BlockBuilder, FunctionBuilder, ModuleBuilder
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
    "BlockBuilder",
    "Call",
    "Const",
    "Dimension",
    "Function",
    "FunctionBuilder",
    "FunctionType",
    "GetGlobal",
    "Global",
    "If",
    "Literal",
    "MatMul",
    "Module",
    "ModuleBuilder",
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
    "infer",
]
