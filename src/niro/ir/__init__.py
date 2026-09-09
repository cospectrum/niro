"""Niro IR is a strongly typed, SSA-based representation composed of types,
values, program structure, operations, literals, and attributes.

Construct programs with [`niro.builder`][] and verify them with
[`niro.verify.module`][].
"""

from niro.ir import infer
from niro.ir.accessors import get_operands, get_regions, get_results
from niro.ir.data import AttributeName, Attributes, AttributeValue, Literal
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
)
from niro.ir.program import (
    Block,
    Function,
    FunctionType,
    Global,
    Module,
    Region,
    SymbolName,
    VerifiedModule,
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
    "VerifiedModule",
    "Yield",
    "get_operands",
    "get_regions",
    "get_results",
    "infer",
]
