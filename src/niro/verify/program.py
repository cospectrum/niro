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
    _verify_ownership(module_)
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


def _verify_ownership(module: ir.Module) -> None:
    """Reject shared or cyclic region/block ownership before recursive traversal."""
    regions: set[int] = set()
    blocks: set[ir.Block] = set()
    pending = [f.body for f in module.functions if f.body is not None]
    while pending:
        region = pending.pop()
        if id(region) in regions:
            raise ValueError("region must have a unique owner")
        regions.add(id(region))
        for block in region.blocks:
            if block in blocks:
                raise ValueError("block must have a unique owner")
            blocks.add(block)
            for op in block.operations:
                pending.extend(ir.get_regions(op))


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
    """Validate entry inputs, CFG edges, dominance, and each block's contents."""
    if not region.blocks:
        raise ValueError("region must contain a block")
    if tuple(value.type for value in region.blocks[0].arguments) != input_types:
        raise TypeError("region argument types do not match expected input types")
    dominators = _verify_cfg(region, terminator)
    definitions = {
        block: {
            value.id: value.type
            for value in (
                *block.arguments,
                *(v for op in block.operations for v in ir.get_results(op)),
            )
        }
        for block in region.blocks
    }
    for block in region.blocks:
        scope = dict(outer_scope)
        for dominator in dominators[block] - {block}:
            scope.update(definitions[dominator])
        _verify_block(
            block,
            scope,
            output_types,
            terminator,
            functions=functions,
            globals_=globals_,
        )


def _verify_cfg(
    region: ir.Region, terminator: type[Terminator]
) -> dict[ir.Block, set[ir.Block]]:
    """Validate a nonempty region's edges and return its block dominator sets.

    Require reachable blocks and no edges to entry. Loops need not have an exit.
    Compute dominance by fixed-point predecessor intersection, independent of
    the order of non-entry blocks.
    """
    entry = region.blocks[0]
    predecessors: dict[ir.Block, set[ir.Block]] = {b: set() for b in region.blocks}
    for block in region.blocks:
        if not block.operations or not isinstance(
            block.operations[-1], (terminator, ir.Branch, ir.CondBranch)
        ):
            raise ValueError(f"block must end with {terminator.__name__} or a branch")
        if any(
            isinstance(op, (ir.Return, ir.Yield, ir.Branch, ir.CondBranch))
            for op in block.operations[:-1]
        ):
            raise ValueError("unexpected block terminator")
        op = block.operations[-1]
        edges: tuple[tuple[ir.Block, tuple[ir.Value, ...]], ...] = ()
        if isinstance(op, ir.Branch):
            edges = ((op.target, op.arguments),)
        elif isinstance(op, ir.CondBranch):
            edges = (
                (op.true_target, op.true_arguments),
                (op.false_target, op.false_arguments),
            )
        for target, arguments in edges:
            if target not in predecessors:
                raise ValueError("branch target is outside the current region")
            if target is entry:
                raise ValueError("branches cannot target the region entry block")
            if tuple(v.type for v in arguments) != tuple(
                v.type for v in target.arguments
            ):
                raise TypeError("branch argument types do not match target block")
            predecessors[target].add(block)

    reachable: set[ir.Block] = set()
    pending = [entry]
    while pending:
        block = pending.pop()
        if block in reachable:
            continue
        reachable.add(block)
        pending.extend(ir.get_successors(block.operations[-1]))
    if len(reachable) != len(region.blocks):
        raise ValueError("region contains unreachable blocks")

    dominators = {block: set(reachable) for block in region.blocks}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in region.blocks[1:]:
            common = set.intersection(*(dominators[p] for p in predecessors[block]))
            updated = {block} | common
            if updated != dominators[block]:
                dominators[block] = updated
                changed = True
    return dominators


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
    for index, op in enumerate(block.operations):
        for operand in ir.get_operands(op):
            if operand.id not in scope:
                raise ValueError(f"value {operand.id} is not defined in this scope")
            if operand.type != scope[operand.id]:
                raise TypeError(f"value {operand.id} type differs from its definition")

        _verify_op(op)
        match op:
            case ir.Branch() | ir.CondBranch():
                pass  # Placement and target signatures were checked with the CFG.
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
