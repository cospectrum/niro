"""Derived lookup tables for repeated queries over unchanged IR.

Build with [`index_function`][niro.index.index_function] or
[`index_module`][niro.index.index_module]. Construction takes linear time in the
indexed IR, including operands and CFG edges. Table lookups take expected
constant time; enumerating uses or predecessors takes time proportional to the
number returned.

Indexes retain the original IR objects. Their tables are read-only, but the IR
is still mutable. For functional edits, use [`reindex_function`][niro.index.reindex_function]
or [`reindex_module`][niro.index.reindex_module] to reuse unchanged function indexes.
Changed functions are indexed from scratch. Previously indexed IR, including
shared objects, must remain untouched: replace changed objects and their
enclosing function/module. This contract does not depend on a particular rewriter.

There is no automatic invalidation or verification. After in-place mutation,
use `index_function` or `index_module` to build fresh tables; identity-based
reuse cannot detect such edits. Module indexing requires verified input.

Examples:
    Query a definition, repeated uses, and an operation's containing block:

    ```python
    from niro import index, ir

    x = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    y = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    add = ir.Add(y, x, x)
    block = ir.Block((x,), [add, ir.Return((y,))])
    function = ir.Function(
        "double", ir.FunctionType((x.type,), (y.type,)), ir.Region([block])
    )
    indexed = index.index_function(function)

    assert indexed.definitions[y.id] == ir.OpResult(add, 0)
    assert indexed.uses[x.id] == (ir.Use(add, 0), ir.Use(add, 1))
    assert indexed.parent_blocks[id(add)] is block
    assert indexed.predecessors[block] == ()
    ```
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass

from niro import ir
from niro.ir.accessors import Definition, Use
from niro.ir.program import Block, Function, Global, SymbolName, VerifiedModule
from niro.ir.values import ValueId

__all__ = [
    "FunctionIndex",
    "ModuleIndex",
    "index_function",
    "index_module",
    "reindex_function",
    "reindex_module",
]


@dataclass(frozen=True, slots=True)
class FunctionIndex:
    """Lookup tables for one unchanged function, including its nested regions.

    Construct with [`index_function`][niro.index.index_function] for read-only
    mappings. Missing definitions, uses, or operations have no table entry;
    declarations have empty tables.

    Attributes:
        function: Original function; retained, not copied or frozen.
        definitions: Function-local value IDs mapped to their definitions.
        uses: Operand slots in [`iter_uses`][niro.ir.iter_uses] order. Repeated
            operands produce separate entries; unused values have no entry.
        parent_blocks: Operation object IDs (`id(op)`) mapped to their first
            containing block, matching [`get_parent_block`][niro.ir.get_parent_block].
            Structural operation equality is deliberately ignored.
        predecessors: Every owned block mapped to its incoming source blocks,
            ordered by [`iter_blocks`][niro.ir.iter_blocks], then successor index.
            Repeated edges repeat their source. Nested If blocks have no CFG
            predecessors; region ownership does not create a CFG edge.
    """

    function: Function
    definitions: Mapping[ValueId, Definition]
    uses: Mapping[ValueId, tuple[Use, ...]]
    parent_blocks: Mapping[int, Block]
    predecessors: Mapping[Block, tuple[Block, ...]]


@dataclass(frozen=True, slots=True)
class ModuleIndex:
    """Symbol tables and function indexes for one unchanged verified module.

    Construct with [`index_module`][niro.index.index_module] for read-only mappings.
    Symbols retain module order and refer to the original IR. Value IDs remain
    local to each FunctionIndex, including when functions reuse the same IDs.

    Attributes:
        module: Verified module; retained, not copied or frozen.
        functions: Function names mapped to indexes, including declarations.
        globals: Global names mapped to the original globals.
    """

    module: VerifiedModule
    functions: Mapping[SymbolName, FunctionIndex]
    globals: Mapping[SymbolName, Global]


def index_function(function: Function) -> FunctionIndex:
    """Build read-only lookup tables without copying or verifying the function.

    Assume unique definition IDs, tree-shaped region/block ownership, and branch
    targets within their owning region, as required by the IR. Empty bodies and
    blocks are supported for queries during construction. Only final operations
    contribute CFG edges. Rebuild after changing the function or its contents.
    """
    definitions: dict[ir.ValueId, ir.Definition] = {}
    uses: dict[ir.ValueId, list[ir.Use]] = {}
    parent_blocks: dict[int, ir.Block] = {}
    blocks = tuple(ir.iter_blocks(function.body)) if function.body is not None else ()
    predecessors: dict[ir.Block, list[ir.Block]] = {block: [] for block in blocks}
    for block in blocks:
        for position, argument in enumerate(block.arguments):
            definitions[argument.id] = ir.BlockArgument(block, position)
        for op in block.operations:
            parent_blocks.setdefault(id(op), block)
            for position, result in enumerate(ir.get_results(op)):
                definitions[result.id] = ir.OpResult(op, position)
        if block.operations:
            for target in ir.get_successors(block.operations[-1]):
                predecessors[target].append(block)

    # Operation traversal interleaves nested regions with their parent's ops.
    if function.body is not None:
        for op in ir.iter_ops(function.body):
            for position, operand in enumerate(ir.get_operands(op)):
                uses.setdefault(operand.id, []).append(ir.Use(op, position))

    return FunctionIndex(
        function,
        types.MappingProxyType(definitions),
        types.MappingProxyType(
            {value_id: tuple(slots) for value_id, slots in uses.items()}
        ),
        types.MappingProxyType(parent_blocks),
        types.MappingProxyType(
            {block: tuple(sources) for block, sources in predecessors.items()}
        ),
    )


def index_module(module: VerifiedModule) -> ModuleIndex:
    """Build read-only symbol tables and function indexes for a verified module.

    Obtain the input through [`niro.verify.module`][]; this function does not
    repeat verification. After changing module symbols or function contents,
    verify the resulting module and build a new index.
    """
    return ModuleIndex(
        module,
        types.MappingProxyType(
            {function.name: index_function(function) for function in module.functions}
        ),
        types.MappingProxyType({global_.name: global_ for global_ in module.globals}),
    )


def reindex_function(old_index: FunctionIndex, new_function: Function) -> FunctionIndex:
    """Return an index for a function produced without mutating previously indexed IR.

    Return `old_index` in constant time when its function is `new_function` by
    identity. Otherwise build fresh tables with [`index_function`][niro.index.index_function];
    no structural comparison or reuse within a changed function is performed.
    The old index and IR remain untouched.

    After an in-place edit, use `index_function` instead: this function assumes
    unchanged identity means unchanged contents, including nested regions.
    """
    if old_index.function is new_function:
        return old_index
    return index_function(new_function)


def reindex_module(old_index: ModuleIndex, new_module: VerifiedModule) -> ModuleIndex:
    """Return an index for a verified module, reusing unchanged function indexes.

    Previously indexed IR must remain untouched. Return `old_index` in constant
    time when its module is `new_module` by identity. Otherwise rebuild symbol
    tables in the new module's order, matching functions by name and reusing
    their indexes only when the function objects are identical. Added or changed
    functions are indexed from scratch; removed symbols are omitted.

    Work is linear in the new module's symbol count plus the size of functions
    requiring fresh indexes. Verify changed IR with [`niro.verify.module`][]
    before calling; no verification is performed here. After in-place mutation,
    use [`index_module`][niro.index.index_module] instead.
    """
    if old_index.module is new_module:
        return old_index
    functions: dict[ir.SymbolName, FunctionIndex] = {}
    for function in new_module.functions:
        previous = old_index.functions.get(function.name)
        functions[function.name] = (
            index_function(function)
            if previous is None
            else reindex_function(previous, function)
        )
    return ModuleIndex(
        new_module,
        types.MappingProxyType(functions),
        types.MappingProxyType(
            {global_.name: global_ for global_ in new_module.globals}
        ),
    )
