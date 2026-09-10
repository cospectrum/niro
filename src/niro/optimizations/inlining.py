"""Inlining of nonrecursive function definitions."""

import dataclasses
import graphlib
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from niro import ir, rewrite, verify
from niro.ir import SymbolName, VerifiedModule

__all__ = ["inline_functions"]


def inline_functions(
    module: VerifiedModule,
    *,
    callees: AbstractSet[SymbolName] | None = None,
    max_callee_ops: int | None = None,
) -> VerifiedModule:
    """Inline calls to nonrecursive definitions, including inside nested regions.

    Clone the callee's body at each call site with fresh value IDs, substitute
    its parameters with the call arguments, and redirect uses of call results
    to the returned values. Preserve operation order and function definitions.
    External declarations and all functions in direct or indirect call cycles
    are excluded as callees. Calls to eligible helpers inside recursive
    functions can still be inlined.

    Selection applies to call targets throughout the module, including inside
    unselected callers. Calls to unselected functions remain calls when a
    selected callee's body is copied. Selection does not override recursion,
    external-declaration, or size restrictions.

    Process callees before their callers, independently of module function
    order. The size limit applies to each callee's body after its own eligible
    calls have been inlined. Count every operation recursively, including `If`,
    `Return`, and `Yield`; exclude the function container. This follows MLIR's
    operation counting, using an absolute limit rather than a callee/caller ratio.
    The limit bounds each copied body, not total code growth.

    For a call chain `A -> B -> C`, first inline `C` into `B` if eligible,
    then inline the updated `B` into `A` if eligible. If expanding `C` makes
    `B` exceed the size limit, `A` keeps its call to `B`, while `B` keeps
    the inlined body of `C`.

    Args:
        module: Verified input module. Its functions and metadata are not mutated.
        callees: Function symbols to consider for inlining. None selects all
            functions; an empty set inlines nothing. Accepts sets and frozensets.
        max_callee_ops: Maximum number of operations in an eligible callee's body.
            None imposes no size limit; zero disables inlining. Must be nonnegative.

    Returns:
        A verified module with eligible calls inlined, or the input object if
        nothing changes. Unchanged IR and metadata may be shared.

    Raises:
        ValueError: max_callee_ops is negative, or callees contains names that
            do not identify functions in the module.

    Examples:
        Inline a two-operation helper while retaining its definition:

        Before:

        ```text
        module {
          func @double(%0: i32) -> i32 {
            %1 = add %0, %0 : i32
            return %1
          }
          func @caller(%0: i32) -> i32 {
            %1 = call @double(%0) : i32
            return %1
          }
        }
        ```

        After:

        ```text
        module {
          func @double(%0: i32) -> i32 {
            %1 = add %0, %0 : i32
            return %1
          }
          func @caller(%0: i32) -> i32 {
            %2 = add %0, %0 : i32
            return %2
          }
        }
        ```

        ```python
        from niro import ir, optimizations, verify

        scalar = ir.ScalarType.I32
        signature = ir.FunctionType((scalar,), (scalar,))
        arg = ir.Value(ir.ValueId(0), scalar)
        doubled = ir.Value(ir.ValueId(1), scalar)
        helper = ir.Function("double", signature, ir.Region([ir.Block((arg,), [
            ir.Add(doubled, arg, arg), ir.Return((doubled,)),
        ])]))
        x = ir.Value(ir.ValueId(0), scalar)
        result = ir.Value(ir.ValueId(1), scalar)
        call = ir.Call("double", (x,), (result,))
        caller = ir.Function("caller", signature, ir.Region([ir.Block((x,), [
            call, ir.Return((result,)),
        ])]))
        module = verify.module(ir.Module(functions=[helper, caller]))
        optimized = optimizations.inline_functions(module, max_callee_ops=2)
        body = optimized.functions[1].first_block
        assert body is not None
        fresh = ir.Value(ir.ValueId(2), scalar)
        assert body.operations == [ir.Add(fresh, x, x), ir.Return((fresh,))]
        assert optimized.functions[0] is helper
        assert optimizations.inline_functions(module, max_callee_ops=1) is module
        assert optimizations.inline_functions(module, callees={"double"}) == optimized
        assert optimizations.inline_functions(module, callees={"caller"}) is module
        assert optimizations.inline_functions(module, callees=set()) is module
        ```
    """
    if max_callee_ops is not None and max_callee_ops < 0:
        raise ValueError("max_callee_ops must be nonnegative or None")

    functions = {function.name: function for function in module.functions}
    if callees is not None:
        unknown = callees - functions.keys()
        if unknown:
            raise ValueError(f"Unknown callee functions: {', '.join(sorted(unknown))}")
        if not callees:
            return module
    graph = {
        name: {
            op.callee
            for op in (() if function.body is None else ir.iter_ops(function.body))
            if isinstance(op, ir.Call)
        }
        for name, function in functions.items()
    }
    recursive = _recursive_functions(graph)
    dependencies = {name: targets - recursive for name, targets in graph.items()}
    eligible: dict[ir.SymbolName, ir.Function] = {}
    for name in graphlib.TopologicalSorter(dependencies).static_order():
        function = _inline_calls(functions[name], eligible)
        functions[name] = function
        if function.body is None or name in recursive:
            continue
        if callees is not None and name not in callees:
            continue
        if max_callee_ops is not None:
            size = sum(1 for _ in ir.iter_ops(function.body))
            if size > max_callee_ops:
                continue
        eligible[name] = function

    updated = [functions[function.name] for function in module.functions]
    if all(a is b for a, b in zip(updated, module.functions, strict=True)):
        return module
    return verify.module(dataclasses.replace(module, functions=updated))


def _recursive_functions(
    graph: Mapping[ir.SymbolName, set[ir.SymbolName]],
) -> set[ir.SymbolName]:
    """Return the names of functions that can call themselves, directly or indirectly.

    Follow each function's calls to see whether any chain leads back to that
    function. For example, if `a` calls `b` and `b` calls `a`, return both names.
    A function that only calls into this cycle is not itself recursive.

    Args:
        graph: Map each function name to the names of functions it calls directly.
            Every called function must also be a key, with an empty set if it
            calls no functions. The mapping and its sets are not modified.

    Returns:
        Names belonging to at least one call cycle, including direct self-calls.
        The inliner excludes these functions as callees to avoid repeated expansion.
    """
    recursive: set[ir.SymbolName] = set()
    for name, callees in graph.items():
        pending = list(callees)
        visited: set[ir.SymbolName] = set()
        while pending:
            callee = pending.pop()
            if callee == name:
                recursive.add(name)
                break
            if callee in visited:
                continue
            visited.add(callee)
            pending.extend(graph[callee])
    return recursive


def _inline_calls(
    function: ir.Function, callees: Mapping[ir.SymbolName, ir.Function]
) -> ir.Function:
    """Return a function with all calls to eligible callees inlined recursively.

    The eligible callee mapping must exclude recursive definitions. Preserve the
    input function when no call changes.
    """
    if function.body is None:
        return function
    supply = rewrite.value_supply(function)
    while True:
        assert function.body is not None
        for op in ir.iter_ops(function.body):
            if not isinstance(op, ir.Call) or op.callee not in callees:
                continue
            function = _inline_call(function, op, callees[op.callee], supply)
            # Edits invalidate references; resume traversal on the new function.
            break
        else:
            return function


def _inline_call(
    function: ir.Function,
    call: ir.Call,
    callee: ir.Function,
    supply: ir.ValueSupply,
) -> ir.Function:
    """Return a function with one call replaced by a cloned callee body.

    The callee must have a single-block definition ending in `Return`. Consume
    fresh IDs from the supply and redirect call results to cloned return values.
    """
    assert callee.body is not None
    (block,) = callee.body.blocks
    # Omitting the parameters makes them captures that cloning can substitute.
    fragment = ir.Region([ir.Block(operations=block.operations)])
    captures = {
        parameter.id: argument
        for parameter, argument in zip(block.arguments, call.arguments, strict=True)
    }
    cloned, _ = rewrite.clone_region(fragment, supply, captures=captures)
    operations = cloned.blocks[0].operations
    ret = operations[-1]
    assert isinstance(ret, ir.Return)
    replacements = {
        result.id: returned
        for result, returned in zip(call.results, ret.operands, strict=True)
    }
    return rewrite.replace_op(
        function, call, operations[:-1], replacements=replacements
    )
