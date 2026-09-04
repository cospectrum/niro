"""Verification rules and checked result-type inference for Niro IR.

The ``check_*`` functions check local rules during construction.
[`verify`][niro.ir.verify.verify] checks a complete module, including symbols,
SSA scope, and nested regions. Both leave their inputs unchanged and raise
[`VerificationError`][niro.ir.VerificationError] on invalid IR.

Re-exported in [`niro.ir`][].
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from typing import assert_never

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
)
from niro.ir.types import ScalarType, TensorType, Type
from niro.ir.values import Value, ValueId


class VerificationError(ValueError):
    """An IR validity rule was violated."""


def verify(module: Module) -> None:
    """Check a complete module without modifying it.

    Verify symbols, global initializers, function signatures, operation types,
    SSA uniqueness and visibility, and region structure and terminators.
    Function declarations are allowed. Unknown operations receive structural
    and SSA checks; their semantics remain unknown.

    Args:
        module: The module to verify, including all referenced declarations.

    Raises:
        VerificationError: The first invalid IR construct, with its location.
        TypeError: The argument is not a [`Module`][niro.ir.Module].
    """
    if not isinstance(module, Module):
        raise TypeError("verify expects a Module")

    with _at("module"):
        _check_attributes(module.attributes)
    symbols: dict[SymbolName, Global | Function] = {}
    for symbol in (*module.globals, *module.functions):
        if symbol.name in symbols:
            raise VerificationError(
                f"module: symbol names must be unique; duplicate {symbol.name!r}"
            )
        symbols[symbol.name] = symbol

    for global_ in module.globals:
        with _at(f"global {global_.name!r}"):
            check_global(global_)

    verifier = _Verifier(symbols)
    for function in module.functions:
        with _at(f"function {function.name!r}"):
            verifier.function(function)


def check_type(value_type: Type) -> None:
    """Check a scalar or tensor value type."""
    match value_type:
        case ScalarType():
            pass
        case TensorType(element_type, shape):
            if not isinstance(element_type, ScalarType):
                raise VerificationError("tensor element type must be a scalar type")
            if shape is None:
                return
            if not isinstance(shape, tuple):
                raise VerificationError("tensor shape must be a tuple or None")
            for dimension in shape:
                if dimension is None:
                    continue
                if type(dimension) is not int or dimension < 0:
                    raise VerificationError(
                        "tensor dimension must be a non-negative integer or None"
                    )
        case _:
            raise VerificationError("value type must be a scalar or tensor type")


def check_function_type(function_type: FunctionType) -> None:
    """Check a function type's input and output value types."""
    if not isinstance(function_type, FunctionType):
        raise VerificationError("function signature must be a FunctionType")
    for value_type in (*function_type.inputs, *function_type.outputs):
        check_type(value_type)


def check_symbol_name(name: SymbolName) -> None:
    """Check the name of a module-level function or global symbol."""
    if not isinstance(name, str) or not name:
        raise VerificationError("symbol name must be a non-empty string")


def check_attribute_name(name: AttributeName) -> None:
    """Check an attribute key."""
    if not isinstance(name, str) or not name:
        raise VerificationError("attribute name must be a non-empty string")


def check_value(value: Value) -> None:
    """Check an SSA value's ID and type, without checking its definition or scope."""
    if type(value.id) is not int or value.id < 0:
        raise VerificationError("value ID must be a non-negative integer")
    check_type(value.type)


def check_function_signature(function: Function) -> None:
    """Check function metadata and signature, allowing an unfinished body."""
    with _at("function name"):
        check_symbol_name(function.name)
    check_function_type(function.type)
    for kind, names, types in (
        ("input", function.input_names, function.type.inputs),
        ("output", function.output_names, function.type.outputs),
    ):
        if names is None:
            continue
        if len(names) != len(types):
            raise VerificationError(f"{kind} names must match {kind} arity")
        for name in names:
            if name is not None and (not isinstance(name, str) or not name):
                raise VerificationError(f"{kind} name must be a non-empty string")
    _check_attributes(function.attributes)


def check_global(global_: Global) -> None:
    """Check a global's name, type, initializer, and attributes."""
    with _at("global name"):
        check_symbol_name(global_.name)
    check_type(global_.type)
    _check_literal(global_.initializer, global_.type)
    _check_attributes(global_.attributes)


def check_symbol_available(module: Module, name: SymbolName) -> None:
    """Check a new declaration's name before inserting it into a module."""
    check_symbol_name(name)
    if any(symbol.name == name for symbol in (*module.globals, *module.functions)):
        raise VerificationError("module symbol names must be unique")


def check_block_arguments(block: Block, owner: Function | If) -> None:
    """Check entry arguments for a function or an argument-free If branch."""
    ids: set[ValueId] = set()
    for argument in block.arguments:
        check_value(argument)
        if argument.id in ids:
            raise VerificationError(f"value %{argument.id} is defined more than once")
        ids.add(argument.id)
    expected = owner.type.inputs if isinstance(owner, Function) else ()
    if tuple(argument.type for argument in block.arguments) != expected:
        raise VerificationError("block argument types must match region input types")


def check_op(op: Op) -> None:
    """Check an operation's values, attributes, and type constraints.

    This local check allows unfinished nested regions. It does not resolve
    symbols or check SSA scope, region bodies, or terminator placement. Use
    [`verify`][niro.ir.verify.verify] to check a complete module.
    """
    for value in (*op.get_operands(), *op.get_results()):
        check_value(value)
    result_ids = [value.id for value in op.get_results()]
    if len(result_ids) != len(set(result_ids)):
        raise VerificationError("operation must produce unique value IDs")

    match op:
        case Const():
            _check_literal(op.literal, op.result.type)
        case GetGlobal(name=name):
            with _at("global name"):
                check_symbol_name(name)
        case Transpose():
            if op.result.type != transpose_result_type(op.operand.type, op.permutation):
                raise VerificationError(
                    "transpose result type does not match its operands"
                )
        case Add() | Mul():
            operation = type(op).__name__.lower()
            if op.lhs.type != op.rhs.type or op.result.type != op.lhs.type:
                raise VerificationError(
                    f"{operation} operands and result must have the same type"
                )
            element_type = (
                op.lhs.type.element_type
                if isinstance(op.lhs.type, TensorType)
                else op.lhs.type
            )
            if element_type is ScalarType.BOOL:
                raise VerificationError(f"{operation} does not support boolean values")
        case MatMul():
            if op.result.type != matmul_result_type(op.lhs.type, op.rhs.type):
                raise VerificationError(
                    "matmul result type does not match its operands"
                )
        case Call(callee=callee):
            with _at("callee name"):
                check_symbol_name(callee)
        case Return() | Yield():
            pass
        case If(condition=condition):
            if condition.type is not ScalarType.BOOL:
                raise VerificationError("if condition must be boolean")
        case UnknownOp(name=name, attributes=attributes):
            if not isinstance(name, str) or not name:
                raise VerificationError("UnknownOp name must be a non-empty string")
            _check_attributes(attributes)
        case _ as unreachable:
            assert_never(unreachable)


def check_call_signature(op: Call, callee: Function) -> None:
    """Check a call's local invariants and signature against its resolved callee."""
    check_op(op)
    check_function_signature(callee)
    if op.callee != callee.name:
        raise VerificationError("call target must match callee name")
    if tuple(argument.type for argument in op.arguments) != callee.type.inputs:
        raise VerificationError(
            f"call argument types must match {op.callee!r} input types"
        )
    if tuple(result.type for result in op.results) != callee.type.outputs:
        raise VerificationError(
            f"call result types must match {op.callee!r} output types"
        )


def check_terminator(op: Return | Yield, owner: Function | If) -> None:
    """Check a terminator and its operands against the enclosing function or If."""
    check_op(op)
    if isinstance(owner, Function):
        expected = Return
        outputs = owner.type.outputs
    else:
        expected = Yield
        outputs = tuple(result.type for result in owner.results)
    if not isinstance(op, expected):
        raise VerificationError(f"expected {expected.__name__} terminator")
    if tuple(value.type for value in op.operands) != outputs:
        raise VerificationError(
            f"{expected.__name__} operand types must match region output types {outputs!r}"
        )


def transpose_result_type(
    operand_type: Type, permutation: tuple[int, ...]
) -> TensorType:
    """Check a transpose's inputs and derive its result type."""
    check_type(operand_type)
    if not isinstance(operand_type, TensorType):
        raise VerificationError("transpose operand must be a tensor")
    rank = len(permutation) if operand_type.shape is None else len(operand_type.shape)
    if any(type(index) is not int for index in permutation) or sorted(
        permutation
    ) != list(range(rank)):
        raise VerificationError(
            "transpose permutation must contain every dimension once"
        )
    if operand_type.shape is None:
        return operand_type
    return TensorType(
        operand_type.element_type,
        tuple(operand_type.shape[index] for index in permutation),
    )


def matmul_result_type(lhs: Type, rhs: Type) -> TensorType:
    """Check numeric rank-two matrix inputs and derive the result type."""
    check_type(lhs)
    check_type(rhs)
    if not isinstance(lhs, TensorType) or not isinstance(rhs, TensorType):
        raise VerificationError("matmul operands must be tensors")
    if lhs.rank != 2 or rhs.rank != 2:
        raise VerificationError("matmul operands must be rank-two tensors")
    if lhs.element_type is not rhs.element_type:
        raise VerificationError("matmul operand element types must match")
    if lhs.element_type is ScalarType.BOOL:
        raise VerificationError("matmul does not support boolean values")
    assert lhs.shape is not None and rhs.shape is not None
    lhs_inner, rhs_inner = lhs.shape[1], rhs.shape[0]
    if lhs_inner is not None and rhs_inner is not None and lhs_inner != rhs_inner:
        raise VerificationError("matmul contracting dimensions must match")
    return TensorType(lhs.element_type, (lhs.shape[0], rhs.shape[1]))


def _check_literal(literal: Literal, value_type: Type) -> None:
    match value_type:
        case ScalarType.BOOL if type(literal) is bool:
            return
        case ScalarType.I32 | ScalarType.I64 if type(literal) is int:
            bits = value_type.byte_width * 8
            if not -(1 << (bits - 1)) <= literal < (1 << (bits - 1)):
                raise VerificationError(
                    f"integer literal is out of range for {value_type.value}"
                )
            return
        case ScalarType.F32 | ScalarType.F64 if type(literal) is float:
            if value_type is ScalarType.F32:
                try:
                    struct.pack("<f", literal)
                except OverflowError:
                    raise VerificationError(
                        "floating-point literal is out of range for f32"
                    ) from None
            return
        case TensorType(element_type, shape):
            if (
                not isinstance(literal, bytes)
                or shape is None
                or any(dim is None for dim in shape)
            ):
                raise VerificationError(
                    "tensor literal requires packed bytes and a static shape"
                )
            size = math.prod(dim for dim in shape if dim is not None)
            expected = size * element_type.byte_width
            if len(literal) != expected:
                raise VerificationError(
                    f"tensor literal has {len(literal)} bytes, expected {expected}"
                )
            return
    raise VerificationError("literal does not match its type")


def _check_attributes(attributes: Attributes) -> None:
    for name, value in attributes.items():
        check_attribute_name(name)
        with _at(f"attribute {name!r}"):
            _check_attribute_value(value)


def _check_attribute_value(value: AttributeValue) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if isinstance(value, tuple):
        for item in value:
            _check_attribute_value(item)
        return
    raise VerificationError(
        "attribute value must be a primitive or tuple of attributes"
    )


@contextmanager
def _at(location: str) -> Iterator[None]:
    try:
        yield
    except VerificationError as error:
        raise VerificationError(f"{location}: {error}") from error


class _Verifier:
    def __init__(self, symbols: dict[SymbolName, Global | Function]) -> None:
        self.symbols = symbols
        self.defined: set[ValueId] = set()
        self.regions: set[int] = set()
        self.blocks: set[int] = set()

    def function(self, function: Function) -> None:
        check_function_signature(function)
        self.defined.clear()
        if function.body is not None:
            self.region(function.body, {}, function)

    def region(
        self,
        region: Region,
        visible: dict[ValueId, Type],
        owner: Function | If,
    ) -> None:
        if id(region) in self.regions:
            raise VerificationError("region has multiple owners or contains a cycle")
        self.regions.add(id(region))
        if len(region.blocks) != 1:
            raise VerificationError("region must contain exactly one block")
        (block,) = region.blocks
        if id(block) in self.blocks:
            raise VerificationError("block has multiple owners")
        self.blocks.add(id(block))

        check_block_arguments(block, owner)
        visible = dict(visible)
        for argument in block.arguments:
            self.define(argument, visible)

        for index, op in enumerate(block.operations):
            with _at(f"operation {index} ({type(op).__name__})"):
                check_op(op)
                for operand in op.get_operands():
                    if operand.id not in visible:
                        raise VerificationError(f"value %{operand.id} is not visible")
                    if operand.type != visible[operand.id]:
                        raise VerificationError(
                            f"value %{operand.id} type does not match its definition"
                        )

                if isinstance(op, (Return, Yield)):
                    if index != len(block.operations) - 1:
                        raise VerificationError("terminator must be the last operation")
                    check_terminator(op, owner)

                self.operation(op, visible)
                for result in op.get_results():
                    self.define(result, visible)

        if not block.operations or not block.operations[-1].is_terminator():
            terminator = Return if isinstance(owner, Function) else Yield
            raise VerificationError(f"block must end with {terminator.__name__}")

    def operation(self, op: Op, visible: dict[ValueId, Type]) -> None:
        match op:
            case Call():
                callee = self.symbols.get(op.callee)
                if not isinstance(callee, Function):
                    raise VerificationError(
                        f"call target {op.callee!r} must name a function"
                    )
                check_call_signature(op, callee)
            case GetGlobal():
                global_ = self.symbols.get(op.name)
                if not isinstance(global_, Global):
                    raise VerificationError(
                        f"global reference {op.name!r} must name a global"
                    )
                if op.result.type != global_.type:
                    raise VerificationError(
                        f"get_global result type must match global {op.name!r} type"
                    )
            case If():
                for name, region in (
                    ("then region", op.then_region),
                    ("else region", op.else_region),
                ):
                    with _at(name):
                        # Branches capture values available before the If. Its
                        # results become visible only after both branches.
                        self.region(region, visible, op)

    def define(self, value: Value, visible: dict[ValueId, Type]) -> None:
        if value.id in self.defined:
            raise VerificationError(f"value %{value.id} is defined more than once")
        self.defined.add(value.id)
        visible[value.id] = value.type
