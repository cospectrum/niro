import copy
import dataclasses

import pytest

from niro import ir, rewrite, verify


def value(index: int, type: ir.Type = ir.ScalarType.I32) -> ir.Value:
    return ir.Value(ir.ValueId(index), type)


def function(block: ir.Block) -> ir.Function:
    ret = block.operations[-1]
    assert isinstance(ret, ir.Return)
    return ir.Function(
        "f",
        ir.FunctionType(
            tuple(v.type for v in block.arguments), tuple(v.type for v in ret.operands)
        ),
        ir.Region([block]),
        attributes={"tag": "preserved"},
    )


def operations(fn: ir.Function) -> list[ir.Op]:
    assert fn.first_block is not None
    return fn.first_block.operations


def check(fn: ir.Function) -> None:
    verify.module(ir.Module(functions=[fn]))


def test_replace_uses_is_simultaneous_and_preserves_definitions_and_input() -> None:
    a, b, c, result = (value(i) for i in range(4))
    add = ir.Add(result, a, b)
    original = function(ir.Block((a, b, c), [add, ir.Return((result,))]))
    snapshot = copy.deepcopy(original)

    updated = rewrite.replace_uses(original, {a.id: b, b.id: c})

    assert operations(updated)[0] == ir.Add(result, b, c)
    assert updated.first_block is not None
    assert updated.first_block.arguments == (a, b, c)
    assert original == snapshot
    assert updated.attributes == original.attributes
    check(updated)


def test_selective_substitution_uses_original_owners_and_operand_slots() -> None:
    a, b, result = (value(i) for i in range(3))
    condition = value(3, ir.ScalarType.BOOL)
    add = ir.Add(result, a, a)
    branch = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[add, ir.Yield()])]),
        ir.Region([ir.Block(operations=[ir.Yield()])]),
    )
    original = function(ir.Block((a, b, condition), [branch, ir.Return((a,))]))
    seen = []

    def select(use: ir.Use) -> bool:
        seen.append(use)
        return use.owner is add and use.operand_index == 1

    updated = rewrite.replace_uses(original, {a.id: b}, where=select)
    updated_branch = operations(updated)[0]
    assert isinstance(updated_branch, ir.If)
    assert updated_branch.then_region.blocks[0].operations[0] == ir.Add(result, a, b)
    assert operations(updated)[-1] == ir.Return((a,))
    assert [(use.owner is add, use.operand_index) for use in seen] == [
        (True, 0),
        (True, 1),
        (False, 0),
    ]
    assert branch.then_region.blocks[0].operations[0] is add
    check(updated)


def test_replace_multiple_nonadjacent_ops_and_preserve_result_ids() -> None:
    x, a, unrelated, b = (value(i) for i in range(4))
    first, middle, last = ir.Const(a, 2), ir.Const(unrelated, 3), ir.Add(b, x, a)
    block = ir.Block((x,), [first, middle, last, ir.Return((b,))])
    original = function(block)
    replacement = ir.Add(b, x, unrelated)
    updated = rewrite.replace_ops(
        original, (first, last), (replacement,), at=rewrite.InsertPoint(block, 2)
    )
    assert operations(updated) == [middle, replacement, ir.Return((b,))]
    assert block.operations == [first, middle, last, ir.Return((b,))]
    check(updated)


def test_decomposition_remaps_inserted_and_surviving_uses() -> None:
    x, y, old, intermediate, result = (value(i) for i in range(5))
    add = ir.Add(old, x, x)
    block = ir.Block((x, y), [add, ir.Return((old,))])
    updated = rewrite.replace_ops(
        function(block),
        (add,),
        (ir.Add(intermediate, x, x), ir.Add(result, intermediate, y)),
        at=rewrite.InsertPoint(block, 0),
        replacements={x.id: y, old.id: result},
    )
    assert operations(updated) == [
        ir.Add(intermediate, y, y),
        ir.Add(result, intermediate, y),
        ir.Return((result,)),
    ]
    check(updated)


def test_identity_elimination_redirects_nested_captures() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    x, y, out = (value(i, tensor) for i in range(3))
    condition = value(3, ir.ScalarType.BOOL)
    transpose = ir.Transpose(y, x, (0, 1))
    branch = ir.If(
        (out,),
        condition,
        ir.Region([ir.Block(operations=[ir.Yield((y,))])]),
        ir.Region([ir.Block(operations=[ir.Yield((y,))])]),
    )
    original = function(
        ir.Block((x, condition), [transpose, branch, ir.Return((out,))])
    )
    updated = rewrite.replace_ops(original, (transpose,), (), replacements={y.id: x})
    assert updated.body is not None
    assert list(ir.iter_uses(updated, y.id)) == []
    assert len(list(ir.iter_uses(updated, x.id))) == 2
    assert len(list(ir.iter_uses(original, y.id))) == 2
    check(updated)


def test_fold_if_can_extract_branch_and_discard_other_region() -> None:
    x, out, local, discarded = (value(i) for i in range(4))
    condition = value(4, ir.ScalarType.BOOL)
    add = ir.Add(local, x, x)
    branch = ir.If(
        (out,),
        condition,
        ir.Region([ir.Block(operations=[add, ir.Yield((local,))])]),
        ir.Region(
            [ir.Block(operations=[ir.Const(discarded, 0), ir.Yield((discarded,))])]
        ),
    )
    block = ir.Block((x,), [ir.Const(condition, True), branch, ir.Return((out,))])
    original = function(block)
    updated = rewrite.replace_ops(
        original,
        (branch,),
        (add,),
        at=rewrite.InsertPoint(block, 1),
        replacements={out.id: local},
    )
    assert operations(updated)[1:] == [add, ir.Return((local,))]
    assert ir.get_definition(updated, discarded.id) is None
    assert ir.get_definition(original, discarded.id) is not None
    check(updated)


def test_erasure_allows_uses_inside_removed_subgraph() -> None:
    a, b = value(0), value(1)
    producer, consumer = ir.Const(a, 1), ir.Add(b, a, a)
    original = function(ir.Block(operations=[producer, consumer, ir.Return()]))
    with pytest.raises(ValueError, match="dangling use"):
        rewrite.erase_ops(original, (producer,))
    updated = rewrite.erase_ops(original, (producer, consumer))
    assert operations(updated) == [ir.Return()]
    check(updated)


def test_erase_matches_identity_for_equal_zero_result_ops() -> None:
    first, second = ir.UnknownOp("effect", (), ()), ir.UnknownOp("effect", (), ())
    original = function(ir.Block(operations=[first, second, ir.Return()]))
    updated = rewrite.erase_ops(original, (first,))
    assert operations(updated)[0] is second
    assert len(operations(updated)) == 2
    with pytest.raises(ValueError, match="not in the input"):
        rewrite.erase_ops(original, (ir.UnknownOp("effect", (), ()),))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_insertion_uses_original_gaps(index: int) -> None:
    block = ir.Block(operations=[ir.UnknownOp("a", (), ()), ir.Return()])
    original = function(block)
    inserted = ir.UnknownOp("b", (), ())
    updated = rewrite.insert_ops(
        original, (inserted,), at=rewrite.InsertPoint(block, index)
    )
    assert (
        operations(updated)
        == block.operations[:index] + [inserted] + block.operations[index:]
    )
    assert len(block.operations) == 2


def test_insert_into_empty_block_and_reject_old_block_reference() -> None:
    block = ir.Block()
    original = ir.Function("f", ir.FunctionType((), ()), ir.Region([block]))
    updated = rewrite.insert_ops(
        original, (ir.Return(),), at=rewrite.InsertPoint(block, 0)
    )
    check(updated)
    with pytest.raises(ValueError, match="not in the input"):
        rewrite.insert_ops(updated, (), at=rewrite.InsertPoint(block, 0))


@pytest.mark.parametrize(
    "index, expected", [(0, [1, 0, 2]), (1, [0, 1, 2]), (3, [0, 2, 1])]
)
def test_move_uses_original_index_without_off_by_one(
    index: int, expected: list[int]
) -> None:
    constants = [ir.Const(value(i), i) for i in range(3)]
    block = ir.Block(operations=[*constants, ir.Return()])
    original = function(block)
    updated = rewrite.move_ops(
        original, (constants[1],), to=rewrite.InsertPoint(block, index)
    )
    assert operations(updated) == [*(constants[i] for i in expected), ir.Return()]
    assert all(operations(updated)[j] is constants[i] for j, i in enumerate(expected))
    check(updated)


def test_move_from_nested_region_preserves_values_and_regions() -> None:
    condition, local = value(0, ir.ScalarType.BOOL), value(1)
    constant = ir.Const(local, 1)
    branch = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[constant, ir.Yield()])]),
        ir.Region([ir.Block(operations=[ir.Yield()])]),
    )
    block = ir.Block((condition,), [branch, ir.Return()])
    original = function(block)
    updated = rewrite.move_ops(original, (constant,), to=rewrite.InsertPoint(block, 0))
    assert operations(updated)[0] is constant
    moved_branch = operations(updated)[1]
    assert isinstance(moved_branch, ir.If)
    assert moved_branch.then_region.blocks[0].operations == [ir.Yield()]
    assert len(branch.then_region.blocks[0].operations) == 2
    check(updated)


def test_reject_overlapping_targets_and_destination_inside_removed_op() -> None:
    condition = value(0, ir.ScalarType.BOOL)
    end = ir.Yield()
    inner = ir.Block(operations=[end])
    branch = ir.If(
        (),
        condition,
        ir.Region([inner]),
        ir.Region([ir.Block(operations=[ir.Yield()])]),
    )
    original = function(ir.Block((condition,), [branch, ir.Return()]))
    for targets, message in [
        ((branch, end), "overlap"),
        ((branch, branch), "distinct"),
    ]:
        with pytest.raises(ValueError, match=message):
            rewrite.erase_ops(original, targets)
    with pytest.raises(ValueError, match="inside a removed"):
        rewrite.move_ops(original, (branch,), to=rewrite.InsertPoint(inner, 0))


def test_invalid_edits_leave_input_untouched() -> None:
    x, y = value(0), value(1)
    constant = ir.Const(y, 1)
    block = ir.Block((x,), [constant, ir.Return((y,))])
    original = function(block)
    snapshot = copy.deepcopy(original)
    for mapping, message in [
        ({x.id: value(3, ir.ScalarType.F32)}, "types do not match"),
        ({value(99).id: y}, "not defined"),
        ({x.id: value(99)}, "dangling use"),
    ]:
        # Make x used so an undefined replacement necessarily survives.
        used = dataclasses.replace(
            original, body=ir.Region([ir.Block((x,), [constant, ir.Return((x,))])])
        )
        with pytest.raises(ValueError, match=message):
            rewrite.replace_uses(used, mapping)
    with pytest.raises(ValueError, match="duplicate definition"):
        rewrite.insert_ops(original, (constant,), at=rewrite.InsertPoint(block, 0))
    with pytest.raises(ValueError, match="requires an insertion point"):
        rewrite.replace_ops(original, (), (ir.Return(),))
    with pytest.raises(ValueError, match="operand type disagrees"):
        rewrite.insert_ops(
            original,
            (ir.UnknownOp("use", (value(0, ir.ScalarType.F32),), ()),),
            at=rewrite.InsertPoint(block, 0),
        )
    assert original == snapshot


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_insert_point(index: int) -> None:
    with pytest.raises(ValueError, match="outside the block"):
        rewrite.InsertPoint(ir.Block(operations=[ir.Return()]), index)


def test_fresh_supply_includes_nested_definitions_and_is_functional() -> None:
    condition = value(0, ir.ScalarType.BOOL)
    branch = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[ir.Const(value(100), 1), ir.Yield()])]),
        ir.Region([ir.Block(operations=[ir.Yield()])]),
    )
    original = function(ir.Block((condition,), [branch, ir.Return()]))
    supply = rewrite.value_supply(original)
    first, advanced = rewrite.fresh_value(supply, ir.ScalarType.F32)
    second, final = rewrite.fresh_value(advanced, ir.ScalarType.I32)
    assert (first.id, second.id, final.next_id, supply.next_id) == (101, 102, 103, 101)
    with pytest.raises(ValueError, match="nonnegative"):
        rewrite.ValueSupply(-1)


def test_clone_freshens_arguments_results_and_nested_definitions() -> None:
    x, a, b, local, result = (value(i) for i in range(5))
    condition = value(5, ir.ScalarType.BOOL)
    branch = ir.If(
        (result,),
        condition,
        ir.Region([ir.Block(operations=[ir.Add(local, a, x), ir.Yield((local,))])]),
        ir.Region([ir.Block(operations=[ir.Yield((b,))])]),
    )
    region = ir.Region([ir.Block((a, b), [branch, ir.Return((result,))])])
    snapshot = copy.deepcopy(region)
    capture = value(50)
    cloned, mapping, supply = rewrite.clone_region(
        region, rewrite.ValueSupply(100), captures={x.id: capture}
    )
    assert set(mapping) == {a.id, b.id, result.id, local.id}
    assert len({v.id for v in mapping.values()}) == 4
    assert supply.next_id == 104
    assert cloned.blocks[0].arguments == (mapping[a.id], mapping[b.id])
    copied = cloned.blocks[0].operations[0]
    assert isinstance(copied, ir.If)
    assert copied.condition is condition
    assert copied.results == (mapping[result.id],)
    assert copied.then_region.blocks[0].operations == [
        ir.Add(mapping[local.id], mapping[a.id], capture),
        ir.Yield((mapping[local.id],)),
    ]
    assert copied.else_region.blocks[0].operations == [ir.Yield((mapping[b.id],))]
    assert cloned.blocks[0].operations[-1] == ir.Return((mapping[result.id],))
    assert region == snapshot
    assert copied.then_region is not branch.then_region


def test_clone_supports_all_op_interfaces_and_multiple_results() -> None:
    a, b, c, d, e, f, g, h, i, j, k = (value(n) for n in range(11))
    region = ir.Region(
        [
            ir.Block(
                (a,),
                [
                    ir.Const(b, 1),
                    ir.GetGlobal("g", c),
                    ir.Transpose(d, a, ()),
                    ir.Add(e, a, b),
                    ir.Mul(f, a, b),
                    ir.MatMul(g, a, b),
                    ir.Call("callee", (a, b), (h, i)),
                    ir.UnknownOp("custom", (h, i), (j, k), {"data": b"payload"}),
                    ir.Return((j, k)),
                ],
            )
        ]
    )
    cloned, mapping, _ = rewrite.clone_region(region, rewrite.ValueSupply(20))
    for original, copied in zip(ir.iter_ops(region), ir.iter_ops(cloned), strict=True):
        assert type(copied) is type(original)
        assert ir.get_operands(copied) == tuple(
            mapping[v.id] for v in ir.get_operands(original)
        )
        assert ir.get_results(copied) == tuple(
            mapping[v.id] for v in ir.get_results(original)
        )
    custom = cloned.blocks[0].operations[-2]
    assert isinstance(custom, ir.UnknownOp)
    assert custom.attributes == {"data": b"payload"}


def test_clone_rejects_collisions_invalid_captures_and_duplicate_definitions() -> None:
    x, result = value(10), value(0)
    region = ir.Region([ir.Block(operations=[ir.Add(result, x, x)])])
    for supply, captures, message in [
        (rewrite.ValueSupply(10), {}, "collides"),
        (rewrite.ValueSupply(20), {result.id: x}, "region definition"),
        (
            rewrite.ValueSupply(20),
            {x.id: value(30, ir.ScalarType.F32)},
            "types do not match",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            rewrite.clone_region(region, supply, captures=captures)
    duplicate = ir.Region([ir.Block((result,), [ir.Const(result, 1)])])
    with pytest.raises(ValueError, match="duplicate definition"):
        rewrite.clone_region(duplicate, rewrite.ValueSupply(20))


def test_inline_using_clone_and_replace() -> None:
    arg, local = value(0), value(1)
    callee = function(ir.Block((arg,), [ir.Add(local, arg, arg), ir.Return((local,))]))
    x, result = value(0), value(1)
    call = ir.Call("f", (x,), (result,))
    block = ir.Block((x,), [call, ir.Return((result,))])
    caller = function(block)
    # Clone the callee's operations as a fragment: its parameters become captures.
    fragment = ir.Region([ir.Block(operations=operations(callee))])
    cloned, mapping, _ = rewrite.clone_region(
        fragment, rewrite.value_supply(caller), captures={arg.id: x}
    )
    updated = rewrite.replace_ops(
        caller,
        (call,),
        cloned.blocks[0].operations[:-1],
        at=rewrite.InsertPoint(block, 0),
        replacements={result.id: mapping[local.id]},
    )
    assert operations(updated) == [
        ir.Add(mapping[local.id], x, x),
        ir.Return((mapping[local.id],)),
    ]
    check(updated)


def test_maps_visit_only_immediate_children_in_accessor_order() -> None:
    condition = value(0, ir.ScalarType.BOOL)
    leaf = ir.If((), condition, ir.Region(), ir.Region())
    then_block, else_block = ir.Block(operations=[leaf]), ir.Block()
    branch = ir.If((), condition, ir.Region([then_block]), ir.Region([else_block]))
    seen = []

    def transform(region: ir.Region) -> ir.Region:
        seen.append(region)
        return rewrite.map_blocks(
            region, lambda block: dataclasses.replace(block, arguments=(condition,))
        )

    updated = rewrite.map_regions(branch, transform)
    assert isinstance(updated, ir.If)
    assert seen[0] is branch.then_region and seen[1] is branch.else_region
    assert len(seen) == 2
    assert updated.then_region.blocks[0].operations[0] is leaf
    assert then_block.arguments == ()
    assert rewrite.map_regions(leaf, lambda region: region) is leaf
    constant = ir.Const(value(1), 0)
    assert rewrite.map_regions(constant, transform) is constant


def test_with_operands_does_not_rewrite_nested_regions_or_definitions() -> None:
    a, b = value(0, ir.ScalarType.BOOL), value(1, ir.ScalarType.BOOL)
    branch = ir.If(
        (), a, ir.Region([ir.Block(operations=[ir.Yield((a,))])]), ir.Region()
    )
    updated = rewrite.with_operands(branch, (b,))
    assert isinstance(updated, ir.If)
    assert updated.condition is b
    assert updated.then_region is branch.then_region
    with pytest.raises(ValueError, match="arity"):
        rewrite.with_operands(branch, ())


def test_empty_fragments_and_external_functions() -> None:
    external = ir.Function("external", ir.FunctionType((), ()))
    assert rewrite.replace_uses(external, {}) is external
    assert rewrite.erase_ops(external, ()) is external
    assert rewrite.value_supply(external).next_id == 0
    with pytest.raises(ValueError, match="not defined"):
        rewrite.replace_uses(external, {value(0).id: value(1)})
    region = ir.Region()
    cloned, mapping, supply = rewrite.clone_region(region, rewrite.ValueSupply(3))
    assert cloned == region
    assert mapping == {}
    assert supply.next_id == 3


def test_unused_replacement_still_requires_a_defined_target() -> None:
    unused = value(0)
    original = function(ir.Block((unused,), [ir.Return()]))
    with pytest.raises(ValueError, match="dangling"):
        rewrite.replace_uses(original, {unused.id: value(99)})


def test_clone_accepts_unused_capture_mappings() -> None:
    local, unused = value(0), value(1)
    region = ir.Region([ir.Block(operations=[ir.Const(local, 1)])])
    cloned, mapping, supply = rewrite.clone_region(
        region, rewrite.ValueSupply(10), captures={unused.id: value(20)}
    )
    assert cloned.blocks[0].operations == [ir.Const(mapping[local.id], 1)]
    assert supply.next_id == 11


def test_shared_target_identity_is_ambiguous() -> None:
    op = ir.UnknownOp("effect", (), ())
    original = function(ir.Block(operations=[op, op, ir.Return()]))
    with pytest.raises(ValueError, match="occurs more than once"):
        rewrite.erase_ops(original, (op,))
    block = ir.Block()
    repeated = ir.Function("f", ir.FunctionType((), ()), ir.Region([block, block]))
    with pytest.raises(ValueError, match="occurs more than once"):
        rewrite.insert_ops(repeated, (ir.Return(),), at=rewrite.InsertPoint(block, 0))
