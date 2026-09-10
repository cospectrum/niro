import dataclasses
import types

import pytest

from niro import index, ir, rewrite, verify


def cfg_function() -> ir.Function:
    supply = ir.ValueSupply()
    flag = supply.fresh(ir.ScalarType.BOOL)
    x, state, doubled, inner_result, selected, final, output = (
        supply.fresh(ir.ScalarType.F32) for _ in range(7)
    )
    inner = ir.If(
        (inner_result,),
        flag,
        ir.Region([ir.Block(operations=[ir.Yield((doubled,))])]),
        ir.Region([ir.Block(operations=[ir.Yield((state,))])]),
    )
    select = ir.If(
        (selected,),
        flag,
        ir.Region(
            [
                ir.Block(
                    operations=[
                        ir.Add(doubled, state, state),
                        inner,
                        ir.Yield((inner_result,)),
                    ]
                )
            ]
        ),
        ir.Region([ir.Block(operations=[ir.Yield((state,))])]),
    )
    loop = ir.Block((state,))
    exit_ = ir.Block((output,), [ir.Return((output,))])
    loop.operations = [
        select,
        ir.Add(final, selected, state),
        ir.CondBranch(flag, loop, exit_, (final,), (final,)),
    ]
    entry = ir.Block((flag, x), [ir.CondBranch(flag, loop, loop, (x,), (x,))])
    function = ir.Function(
        "loop",
        ir.FunctionType((flag.type, x.type), (output.type,)),
        ir.Region([entry, exit_, loop]),
    )
    verify.module(ir.Module([function]))
    return function


def test_index_matches_scanning_queries_through_nested_regions_and_cfg() -> None:
    function = cfg_function()
    assert function.body is not None
    indexed = index.index_function(function)

    assert indexed.function is function
    for value in ir.iter_defined_values(function.body):
        expected = ir.get_definition(function, value.id)
        actual = indexed.definitions[value.id]
        assert expected is not None
        assert actual == expected
        assert actual.owner is expected.owner
        expected_uses = list(ir.iter_uses(function, value.id))
        actual_uses = indexed.uses.get(value.id, ())
        assert list(actual_uses) == expected_uses
        assert all(
            actual.owner is expected.owner
            for actual, expected in zip(actual_uses, expected_uses, strict=True)
        )
    for op in ir.iter_ops(function.body):
        assert indexed.parent_blocks[id(op)] is ir.get_parent_block(function, op)
    assert indexed.definitions.get(ir.ValueId(99)) is None
    assert indexed.uses.get(ir.ValueId(99), ()) == ()
    assert indexed.parent_blocks.get(id(ir.Return())) is None


def test_predecessors_preserve_parallel_edges_and_loops_without_region_edges() -> None:
    function = cfg_function()
    assert function.body is not None
    entry, exit_, loop = function.body.blocks
    indexed = index.index_function(function)

    assert indexed.predecessors[entry] == ()
    assert indexed.predecessors[loop] == (entry, entry, loop)
    assert indexed.predecessors[exit_] == (loop,)
    nested_blocks = set(ir.iter_blocks(function.body)) - {entry, exit_, loop}
    assert nested_blocks
    for block in nested_blocks:
        assert indexed.predecessors[block] == ()


def test_parent_blocks_distinguish_structurally_equal_operations() -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    first, second, absent = ir.Yield(), ir.Yield(), ir.Yield()
    assert first == second == absent
    then = ir.Block(operations=[first])
    else_ = ir.Block(operations=[second])
    select = ir.If((), flag, ir.Region([then]), ir.Region([else_]))
    entry = ir.Block((flag,), [select, ir.Return()])
    function = ir.Function("f", ir.FunctionType((flag.type,), ()), ir.Region([entry]))
    verify.module(ir.Module([function]))

    indexed = index.index_function(function)
    assert indexed.parent_blocks[id(first)] is then
    assert indexed.parent_blocks[id(second)] is else_
    assert indexed.parent_blocks.get(id(absent)) is None
    assert indexed.parent_blocks[id(select)] is entry


@pytest.mark.parametrize("body", [None, ir.Region(), ir.Region([ir.Block()])])
def test_index_supports_declarations_and_empty_fragments(
    body: ir.Region | None,
) -> None:
    function = ir.Function("f", ir.FunctionType((), ()), body)
    indexed = index.index_function(function)
    assert indexed.function is function
    assert indexed.definitions == {}
    assert indexed.uses == {}
    assert indexed.parent_blocks == {}
    assert indexed.predecessors == (
        {body.blocks[0]: ()} if body and body.blocks else {}
    )


def test_module_index_keeps_symbols_and_value_ids_local_to_functions() -> None:
    value, first, second = (
        ir.Value(ir.ValueId(i), ir.ScalarType.F32) for i in range(3)
    )
    callee = ir.Function("external", ir.FunctionType((value.type,), (value.type,) * 2))
    call = ir.Call(callee.name, (value,), (first, second))
    caller = ir.Function(
        "caller",
        ir.FunctionType((value.type,), (value.type,)),
        ir.Region([ir.Block((value,), [call, ir.Return((second,))])]),
    )
    constant = ir.Const(value, 1.0)
    other = ir.Function(
        "other",
        ir.FunctionType((), (value.type,)),
        ir.Region([ir.Block(operations=[constant, ir.Return((value,))])]),
    )
    global_ = ir.Global("weight", value.type, 2.0)
    module = verify.module(ir.Module([callee, caller, other], [global_]))
    indexed = index.index_module(module)

    assert indexed.module is module
    assert list(indexed.functions) == ["external", "caller", "other"]
    assert indexed.functions["external"].function is callee
    assert indexed.functions["external"].definitions == {}
    assert caller.body is not None
    assert indexed.functions["caller"].definitions[value.id] == ir.BlockArgument(
        caller.body.blocks[0], 0
    )
    assert indexed.functions["caller"].definitions[first.id] == ir.OpResult(call, 0)
    assert indexed.functions["caller"].definitions[second.id] == ir.OpResult(call, 1)
    assert indexed.functions["other"].definitions[value.id] == ir.OpResult(constant, 0)
    assert indexed.globals["weight"] is global_
    assert indexed.functions.get("missing") is None
    assert indexed.globals.get("missing") is None
    empty = index.index_module(verify.module(ir.Module()))
    assert empty.functions == empty.globals == {}


def test_factories_return_read_only_tables() -> None:
    module = verify.module(ir.Module([cfg_function()]))
    indexed = index.index_module(module)
    function = indexed.functions["loop"]
    for table in (
        indexed.functions,
        indexed.globals,
        function.definitions,
        function.uses,
        function.parent_blocks,
        function.predecessors,
    ):
        assert isinstance(table, types.MappingProxyType)


def test_reindex_after_functional_rewrite_preserves_original_index() -> None:
    function = cfg_function()
    assert function.body is not None
    original = index.index_function(function)
    entry, _, loop = function.body.blocks
    flag, x = entry.arguments
    (state,) = loop.arguments
    rewritten = rewrite.replace_uses(function, {state.id: x})
    verify.module(ir.Module([rewritten]))
    updated = index.reindex_function(original, rewritten)

    assert original.function is function
    assert updated.function is rewritten
    assert updated.uses.get(state.id, ()) == ()
    assert original.uses[state.id]
    assert list(original.uses[state.id]) == list(ir.iter_uses(function, state.id))
    assert len(updated.uses[x.id]) == len(original.uses[x.id]) + len(
        original.uses[state.id]
    )
    assert len(updated.uses[flag.id]) == len(original.uses[flag.id])
    assert rewritten.body is not None
    new_entry, _, new_loop = rewritten.body.blocks
    assert new_loop is not loop
    assert updated.predecessors[new_loop] == (new_entry, new_entry, new_loop)
    assert original.predecessors[loop] == (entry, entry, loop)
    assert updated == index.index_function(rewritten)
    assert original == index.index_function(function)


@pytest.mark.parametrize("declaration", [False, True])
def test_reindex_function_reuses_only_identical_functions(declaration: bool) -> None:
    function = (
        ir.Function("f", ir.FunctionType((), ())) if declaration else cfg_function()
    )
    original = index.index_function(function)
    assert index.reindex_function(original, function) is original

    # Equal function containers are still distinct versions of the IR.
    copied = dataclasses.replace(function)
    verify.module(ir.Module([copied]))
    assert copied == function
    updated = index.reindex_function(original, copied)
    assert updated is not original
    assert updated.function is copied
    assert updated == index.index_function(copied)


def test_reindex_module_reuses_functions_and_tracks_changed_symbols() -> None:
    function = cfg_function()
    kept = ir.Function("kept", ir.FunctionType((), ()))
    removed = ir.Function("removed", ir.FunctionType((), ()))
    renamed = ir.Function("before", ir.FunctionType((), ()))
    weight = ir.Global("weight", ir.ScalarType.F32, 1.0)
    removed_global = ir.Global("removed_global", ir.ScalarType.I32, 1)
    module = verify.module(
        ir.Module([function, kept, removed, renamed], [weight, removed_global])
    )
    original = index.index_module(module)
    assert index.reindex_module(original, module) is original

    assert function.body is not None
    entry, _, loop = function.body.blocks
    rewritten = rewrite.replace_uses(
        function, {loop.arguments[0].id: entry.arguments[1]}
    )
    added = ir.Function("added", ir.FunctionType((), ()))
    new_name = dataclasses.replace(renamed, name="after")
    new_weight = dataclasses.replace(weight, initializer=2.0)
    added_global = ir.Global("added_global", ir.ScalarType.BOOL, True)
    new_module = verify.module(
        ir.Module(
            [new_name, kept, added, rewritten],
            [added_global, new_weight],
        )
    )
    updated = index.reindex_module(original, new_module)

    assert updated.module is new_module
    assert list(updated.functions) == ["after", "kept", "added", "loop"]
    assert updated.functions["kept"] is original.functions["kept"]
    assert updated.functions["loop"] is not original.functions["loop"]
    assert updated.functions["loop"].function is rewritten
    assert updated.functions["after"].function is new_name
    assert updated.functions["added"].function is added
    assert list(updated.globals) == ["added_global", "weight"]
    assert updated.globals["weight"] is new_weight
    assert original.globals["weight"] is weight
    assert updated == index.index_module(new_module)
    assert original == index.index_module(module)


def test_reindex_module_with_only_metadata_changes_reuses_all_function_indexes() -> (
    None
):
    module = verify.module(ir.Module([cfg_function()], attributes={"version": 1}))
    original = index.index_module(module)
    new_module = verify.module(dataclasses.replace(module, attributes={"version": 2}))
    updated = index.reindex_module(original, new_module)
    assert updated is not original
    assert updated.module is new_module
    assert updated.functions["loop"] is original.functions["loop"]
    assert original.module.attributes == {"version": 1}
    assert updated == index.index_module(new_module)


def test_reindex_module_can_remove_all_symbols_and_add_them_back() -> None:
    module = verify.module(
        ir.Module([cfg_function()], [ir.Global("x", ir.ScalarType.I32, 1)])
    )
    original = index.index_module(module)
    empty_module = verify.module(ir.Module())
    empty = index.reindex_module(original, empty_module)
    assert empty.module is empty_module
    assert empty.functions == empty.globals == {}
    restored = index.reindex_module(empty, module)
    assert restored == original
    assert restored.functions["loop"] is not original.functions["loop"]
