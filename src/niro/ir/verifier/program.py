"""Verify the structure and references of a completed Niro IR module."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from typing import assert_never, cast

from niro import ir
from niro.ir.ops import (
    Return,
    Yield,
)
from niro.ir.program import (
    Block,
    Function,
    Global,
    Module,
    Region,
    SymbolName,
    VerifiedModule,
)
from niro.ir.types import Type
from niro.ir.values import Value, ValueId
from niro.ir.verifier import ops as op_verifier

__all__ = ["verify"]


def verify(module: Module) -> VerifiedModule:
    """Verify module structure, references, and operations; return the same module."""
    _verify_symbol_names(module)
    functions = {function.name: function for function in module.functions}
    globals_ = {global_.name: global_ for global_ in module.globals}
    for function in module.functions:
        _verify_function(function, functions, globals_)
    return ir.VerifiedModule(module)


def _verify_symbol_names(module: Module) -> None:
    names = [symbol.name for symbol in [*module.functions, *module.globals]]
    for name, count in Counter(names).items():
        if not name:
            raise ValueError("module symbol names cannot be empty")
        if count > 1:
            raise ValueError(f"duplicate module symbol: {name!r}")


def _verify_function(
    function: Function,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
) -> None:
    _verify_interface_names("input", function.input_names, len(function.type.inputs))
    _verify_interface_names("output", function.output_names, len(function.type.outputs))
    if function.body is None:
        return
    _verify_value_ids(function.body)
    _verify_region(
        region=function.body,
        outer_scope={},
        input_types=function.type.inputs,
        output_types=function.type.outputs,
        terminator=ir.Return,
        functions=functions,
        globals_=globals_,
    )


def _verify_interface_names(
    kind: str, names: tuple[str | None, ...] | None, arity: int
) -> None:
    if names is None:
        return
    if len(names) != arity:
        raise ValueError(f"{kind} names must match {kind} arity")
    if any(name == "" for name in names):
        raise ValueError(f"{kind} names cannot be empty")


def _iter_defined_values(region: Region) -> Iterator[Value]:
    for block in region.blocks:
        yield from block.arguments
        for operation in block.operations:
            op = cast(ir.Op, operation)
            yield from op.get_results()
            if isinstance(op, ir.If):
                yield from _iter_defined_values(op.then_region)
                yield from _iter_defined_values(op.else_region)


def _verify_value_ids(region: Region) -> None:
    counts = Counter(value.id for value in _iter_defined_values(region))
    for value_id, count in counts.items():
        if count > 1:
            raise ValueError(f"duplicate value ID in function: {value_id}")


def _verify_region(
    region: Region,
    outer_scope: Mapping[ValueId, Type],
    input_types: tuple[Type, ...],
    output_types: tuple[Type, ...],
    terminator: type[Return | Yield],
    *,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
) -> None:
    if not region.blocks:
        raise ValueError("region must contain a block")
    if terminator is ir.Yield and len(region.blocks) != 1:
        raise ValueError("if region must contain exactly one block")
    if tuple(value.type for value in region.blocks[0].arguments) != input_types:
        raise TypeError("region argument types do not match expected input types")
    for block in region.blocks:
        _verify_block(
            block,
            outer_scope,
            output_types,
            terminator,
            functions=functions,
            globals_=globals_,
        )


def _verify_block(
    block: Block,
    outer_scope: Mapping[ValueId, Type],
    output_types: tuple[Type, ...],
    terminator: type[Return | Yield],
    *,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
) -> None:
    scope = {**outer_scope, **{value.id: value.type for value in block.arguments}}
    if not block.operations or not isinstance(block.operations[-1], terminator):
        raise ValueError(f"region must end with {terminator.__name__}")

    for index, operation in enumerate(block.operations):
        op = cast(ir.Op, operation)
        for operand in op.get_operands():
            if operand.id not in scope:
                raise ValueError(f"value {operand.id} is not defined in this scope")
            if operand.type != scope[operand.id]:
                raise TypeError(f"value {operand.id} type differs from its definition")

        op_verifier._verify_op(op)
        match op:
            case ir.Return() | ir.Yield():
                if index != len(block.operations) - 1 or not isinstance(op, terminator):
                    raise ValueError("unexpected block terminator")
                if tuple(value.type for value in op.operands) != output_types:
                    raise TypeError(
                        "terminator operand types do not match region results"
                    )
            case ir.Call():
                function = functions.get(op.callee)
                if function is None:
                    raise ValueError(f"unknown function: {op.callee!r}")
                if tuple(value.type for value in op.arguments) != function.type.inputs:
                    raise TypeError("call argument types do not match function inputs")
                if tuple(value.type for value in op.results) != function.type.outputs:
                    raise TypeError("call result types do not match function outputs")
            case ir.GetGlobal():
                global_ = globals_.get(op.name)
                if global_ is None:
                    raise ValueError(f"unknown global: {op.name!r}")
                if op.result.type != global_.type:
                    raise TypeError("global load type does not match global type")
            case ir.If():
                result_types = tuple(value.type for value in op.results)
                _verify_region(
                    region=op.then_region,
                    outer_scope=scope,
                    input_types=(),
                    output_types=result_types,
                    terminator=ir.Yield,
                    functions=functions,
                    globals_=globals_,
                )
                if op.else_region.blocks or result_types:
                    _verify_region(
                        region=op.else_region,
                        outer_scope=scope,
                        input_types=(),
                        output_types=result_types,
                        terminator=ir.Yield,
                        functions=functions,
                        globals_=globals_,
                    )

            case (
                ir.Const()
                | ir.Add()
                | ir.Mul()
                | ir.MatMul()
                | ir.Transpose()
                | ir.UnknownOp()
            ):
                pass
            case _ as unreachable:
                assert_never(unreachable)

        scope.update((value.id, value.type) for value in op.get_results())
