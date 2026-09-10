"""Verify the structure and references of a completed Niro IR module."""

from __future__ import annotations

import collections
from collections.abc import Mapping
from typing import assert_never

from niro import ir
from niro.ir.program import Module, VerifiedModule
from niro.verify.data import _verify_literal
from niro.verify.ops import _verify_op

__all__ = ["module"]

type Terminator = ir.Return | ir.Yield
"""A region-ending operation: a function return or a nested-region yield."""


def module(module_: Module) -> VerifiedModule:
    """Verify module structure, references, and operations; return the same module."""
    _verify_module(module_)
    return ir.VerifiedModule(module_)


def _verify_module(module_: ir.Module) -> None:
    """Validate module symbols, global literals, and every function definition."""
    _verify_symbol_names(module_)
    for global_ in module_.globals:
        _verify_literal(
            global_.type,
            global_.initializer,
            context=f"global {global_.name!r} initializer",
        )
    functions = {function.name: function for function in module_.functions}
    globals_ = {global_.name: global_ for global_ in module_.globals}
    for function in module_.functions:
        _verify_function(function, functions, globals_)


def _verify_symbol_names(module: ir.Module) -> None:
    """Reject empty or duplicate names across functions and globals."""
    names = [symbol.name for symbol in [*module.functions, *module.globals]]
    for name, count in collections.Counter(names).items():
        if not name:
            raise ValueError("module symbol names cannot be empty")
        if count > 1:
            raise ValueError(f"duplicate module symbol: {name!r}")


def _verify_function(
    function: ir.Function,
    functions: Mapping[ir.SymbolName, ir.Function],
    globals_: Mapping[ir.SymbolName, ir.Global],
) -> None:
    """Validate interface names and, for definitions, body values and structure."""
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
    """Require supplied names to match arity and forbid empty-string names."""
    if names is None:
        return
    if len(names) != arity:
        raise ValueError(f"{kind} names must match {kind} arity")
    if any(name == "" for name in names):
        raise ValueError(f"{kind} names cannot be empty")


def _verify_value_ids(region: ir.Region) -> None:
    """Reject repeated value definitions across a function and its nested regions."""
    counts = collections.Counter(value.id for value in ir.iter_defined_values(region))
    for value_id, count in counts.items():
        if count > 1:
            raise ValueError(f"duplicate value ID in function: {value_id}")


def _verify_region(
    region: ir.Region,
    outer_scope: Mapping[ir.ValueId, ir.Type],
    input_types: tuple[ir.Type, ...],
    output_types: tuple[ir.Type, ...],
    terminator: type[Terminator],
    *,
    functions: Mapping[ir.SymbolName, ir.Function],
    globals_: Mapping[ir.SymbolName, ir.Global],
) -> None:
    """Require a single block with the expected inputs and validate its contents."""
    if not region.blocks:
        raise ValueError("region must contain a block")
    if len(region.blocks) != 1:
        raise ValueError("multiple blocks per region are not supported yet")
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
    block: ir.Block,
    outer_scope: Mapping[ir.ValueId, ir.Type],
    output_types: tuple[ir.Type, ...],
    terminator: type[Terminator],
    *,
    functions: Mapping[ir.SymbolName, ir.Function],
    globals_: Mapping[ir.SymbolName, ir.Global],
) -> None:
    """Validate scoped operands, symbols, nested regions, and the final terminator.

    Outer bindings remain unchanged; each result becomes visible only after its
    operation has been checked.
    """
    scope = {**outer_scope, **{value.id: value.type for value in block.arguments}}
    if not block.operations or not isinstance(block.operations[-1], terminator):
        raise ValueError(f"region must end with {terminator.__name__}")

    for index, op in enumerate(block.operations):
        for operand in ir.get_operands(op):
            if operand.id not in scope:
                raise ValueError(f"value {operand.id} is not defined in this scope")
            if operand.type != scope[operand.id]:
                raise TypeError(f"value {operand.id} type differs from its definition")

        _verify_op(op)
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

        scope.update((value.id, value.type) for value in ir.get_results(op))
