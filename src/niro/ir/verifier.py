"""Verify the structure and references of a completed Niro IR module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from niro.ir.ops import Call, GetGlobal, If, Op, Return, Yield
from niro.ir.program import Block, Function, Global, Module, Region, SymbolName
from niro.ir.types import ScalarType, Type
from niro.ir.values import Value, ValueId

__all__ = ["verify"]


def verify(module: Module) -> Module:
    """Check structural and reference invariants and return the same module.

    Operation-specific arithmetic and literal checks are not performed.
    """
    _verify_symbol_names(module)
    functions = {function.name: function for function in module.functions}
    globals_ = {global_.name: global_ for global_ in module.globals}
    for function in module.functions:
        _verify_function(function, functions, globals_)
    return module


def _verify_symbol_names(module: Module) -> None:
    seen_names: set[SymbolName] = set()
    for symbol in [*module.functions, *module.globals]:
        if not symbol.name:
            raise ValueError("module symbol names cannot be empty")
        if symbol.name in seen_names:
            raise ValueError(f"duplicate module symbol: {symbol.name!r}")
        seen_names.add(symbol.name)


def _verify_function(
    function: Function,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
) -> None:
    if function.body is None:
        return
    _verify_region(
        function.body,
        {},
        function.type.inputs,
        function.type.outputs,
        Return,
        functions=functions,
        globals_=globals_,
        defined_ids=set(),
    )


def _define_values(
    values: tuple[Value, ...],
    scope: dict[ValueId, Type],
    defined_ids: set[ValueId],
) -> None:
    for value in values:
        if value.id in defined_ids:
            raise ValueError(f"duplicate value ID in function: {value.id}")
        defined_ids.add(value.id)
        scope[value.id] = value.type


def _verify_region(
    region: Region,
    outer_scope: Mapping[ValueId, Type],
    input_types: tuple[Type, ...],
    output_types: tuple[Type, ...],
    terminator: type[Return | Yield],
    *,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
    defined_ids: set[ValueId],
) -> None:
    if not region.blocks:
        raise ValueError("region must contain a block")
    if terminator is Yield and len(region.blocks) != 1:
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
            defined_ids=defined_ids,
        )


def _verify_block(
    block: Block,
    outer_scope: Mapping[ValueId, Type],
    output_types: tuple[Type, ...],
    terminator: type[Return | Yield],
    *,
    functions: Mapping[SymbolName, Function],
    globals_: Mapping[SymbolName, Global],
    defined_ids: set[ValueId],
) -> None:
    scope = dict(outer_scope)
    _define_values(block.arguments, scope, defined_ids)
    if not block.operations or not isinstance(block.operations[-1], terminator):
        raise ValueError(f"region must end with {terminator.__name__}")

    for index, operation in enumerate(block.operations):
        op = cast(Op, operation)
        for operand in op.get_operands():
            if operand.id not in scope:
                raise ValueError(f"value {operand.id} is not defined in this scope")
            if operand.type != scope[operand.id]:
                raise TypeError(f"value {operand.id} type differs from its definition")

        match op:
            case Return() | Yield():
                if index != len(block.operations) - 1 or not isinstance(op, terminator):
                    raise ValueError("unexpected block terminator")
                if tuple(value.type for value in op.operands) != output_types:
                    raise TypeError(
                        "terminator operand types do not match region results"
                    )
            case Call():
                function = functions.get(op.callee)
                if function is None:
                    raise ValueError(f"unknown function: {op.callee!r}")
                if tuple(value.type for value in op.arguments) != function.type.inputs:
                    raise TypeError("call argument types do not match function inputs")
                if tuple(value.type for value in op.results) != function.type.outputs:
                    raise TypeError("call result types do not match function outputs")
            case GetGlobal():
                global_ = globals_.get(op.name)
                if global_ is None:
                    raise ValueError(f"unknown global: {op.name!r}")
                if op.result.type != global_.type:
                    raise TypeError("global load type does not match global type")
            case If():
                if op.condition.type is not ScalarType.BOOL:
                    raise TypeError("if condition must be boolean")
                result_types = tuple(value.type for value in op.results)
                _verify_region(
                    op.then_region,
                    scope,
                    (),
                    result_types,
                    Yield,
                    functions=functions,
                    globals_=globals_,
                    defined_ids=defined_ids,
                )
                if op.else_region.blocks or result_types:
                    _verify_region(
                        op.else_region,
                        scope,
                        (),
                        result_types,
                        Yield,
                        functions=functions,
                        globals_=globals_,
                        defined_ids=defined_ids,
                    )

        _define_values(op.get_results(), scope, defined_ids)
