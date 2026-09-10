import pytest

from niro import ir

Operands = tuple[ir.Value, ...]
Results = tuple[ir.Value, ...]


def test_get_definition_finds_arguments_and_results_in_nested_regions() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    argument, nested_argument, first, second, result, else_value = (
        ir.Value(ir.ValueId(index), ir.ScalarType.F32) for index in range(1, 7)
    )
    call = ir.Call("callee", (argument,), (first, second))
    inner = ir.Block(arguments=(nested_argument,), operations=[call])
    nested = ir.If((), condition, ir.Region([inner]), ir.Region())
    else_op = ir.Const(else_value, 1.0)
    branch = ir.If(
        (result,),
        condition,
        ir.Region([ir.Block(operations=[nested])]),
        ir.Region([ir.Block(operations=[else_op])]),
    )
    entry = ir.Block(arguments=(condition, argument), operations=[branch])
    function = ir.Function("f", ir.FunctionType((), ()), ir.Region([entry]))

    for value, owner, index in (
        (condition, entry, 0),
        (argument, entry, 1),
        (nested_argument, inner, 0),
    ):
        definition = ir.get_definition(function, value.id)
        assert isinstance(definition, ir.BlockArgument)
        assert definition.owner is owner
        assert definition.argument_index == index
        assert owner.arguments[index] is value
    for value, op, index in (
        (first, call, 0),
        (second, call, 1),
        (result, branch, 0),
        (else_value, else_op, 0),
    ):
        definition = ir.get_definition(function, value.id)
        assert isinstance(definition, ir.OpResult)
        assert definition.owner is op
        assert definition.result_index == index
        assert ir.get_results(op)[index] is value
    assert ir.get_definition(function, ir.ValueId(99)) is None


def test_iter_uses_reports_operand_slots_in_nested_traversal_order() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    value = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    reference = ir.Value(value.id, value.type)
    assert reference is not value
    result = ir.Value(ir.ValueId(2), value.type)
    add = ir.Add(result, reference, reference)
    then_yield = ir.Yield((value,))
    else_yield = ir.Yield((reference,))
    branch = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[add, then_yield])]),
        ir.Region([ir.Block(operations=[else_yield])]),
    )
    ret = ir.Return((value,))
    function = ir.Function(
        "f",
        ir.FunctionType((), ()),
        ir.Region(
            [
                ir.Block(arguments=(condition, value), operations=[branch]),
                ir.Block(operations=[ret]),
            ]
        ),
    )

    expected = [(add, 0), (add, 1), (then_yield, 0), (else_yield, 0), (ret, 0)]
    for use, (op, index) in zip(
        ir.iter_uses(function, value.id), expected, strict=True
    ):
        assert use.owner is op
        assert use.operand_index == index
    condition_uses = list(ir.iter_uses(function, condition.id))
    assert len(condition_uses) == 1
    assert condition_uses[0].owner is branch
    assert condition_uses[0].operand_index == 0
    assert list(ir.iter_uses(function, result.id)) == []
    assert list(ir.iter_uses(function, ir.ValueId(99))) == []


def test_get_parent_block_matches_identity_in_nested_regions() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    first, second, absent = ir.Yield(), ir.Yield(), ir.Yield()
    assert first == second == absent
    then_block = ir.Block(operations=[first])
    else_block = ir.Block(operations=[second])
    nested = ir.If((), condition, ir.Region([then_block]), ir.Region([else_block]))
    outer = ir.If(
        (), condition, ir.Region([ir.Block(operations=[nested])]), ir.Region()
    )
    entry = ir.Block(operations=[outer])
    ret = ir.Return()
    last = ir.Block(operations=[ret])
    function = ir.Function("f", ir.FunctionType((), ()), ir.Region([entry, last]))

    assert ir.get_parent_block(function, outer) is entry
    assert ir.get_parent_block(function, nested) is outer.then_region.blocks[0]
    assert ir.get_parent_block(function, first) is then_block
    assert ir.get_parent_block(function, second) is else_block
    assert ir.get_parent_block(function, ret) is last
    assert ir.get_parent_block(function, absent) is None


@pytest.mark.parametrize("body", [None, ir.Region(), ir.Region([ir.Block()])])
def test_function_accessors_handle_absent_or_empty_bodies(
    body: ir.Region | None,
) -> None:
    function = ir.Function("f", ir.FunctionType((), ()), body)
    assert ir.get_definition(function, ir.ValueId(0)) is None
    assert list(ir.iter_uses(function, ir.ValueId(0))) == []
    assert ir.get_parent_block(function, ir.Return()) is None


def test_value_lookups_are_function_local_and_do_not_require_a_definition() -> None:
    value = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    const = ir.Const(value, 1.0)
    ret = ir.Return((value,))
    defining = ir.Function(
        "defining", ir.FunctionType((), ()), ir.Region([ir.Block(operations=[const])])
    )
    using = ir.Function(
        "using", ir.FunctionType((), ()), ir.Region([ir.Block(operations=[ret])])
    )

    assert ir.get_definition(defining, value.id) == ir.OpResult(const, 0)
    assert ir.get_definition(using, value.id) is None
    assert list(ir.iter_uses(defining, value.id)) == []
    assert list(ir.iter_uses(using, value.id)) == [ir.Use(ret, 0)]
    assert ir.get_parent_block(using, const) is None


def test_operations_expose_generic_operands_and_results() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    scalar_result = ir.Value(ir.ValueId(2), ir.ScalarType.F32)
    condition = ir.Value(ir.ValueId(3), ir.ScalarType.BOOL)
    tensor_lhs = ir.Value(ir.ValueId(4), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    tensor_rhs = ir.Value(ir.ValueId(5), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    tensor_result = ir.Value(ir.ValueId(6), ir.TensorType(ir.ScalarType.F32, (2, 4)))
    transpose_result = ir.Value(ir.ValueId(7), ir.TensorType(ir.ScalarType.F32, (3, 2)))
    then_region = ir.Region([ir.Block()])
    else_region = ir.Region([ir.Block()])

    operations: list[tuple[ir.Op, Operands, Results]] = [
        (ir.Const(scalar_result, 1.0), (), (scalar_result,)),
        (ir.GetGlobal("value", scalar_result), (), (scalar_result,)),
        (
            ir.Transpose(transpose_result, tensor_lhs, (1, 0)),
            (tensor_lhs,),
            (transpose_result,),
        ),
        (ir.Add(scalar_result, lhs, rhs), (lhs, rhs), (scalar_result,)),
        (ir.Mul(scalar_result, lhs, rhs), (lhs, rhs), (scalar_result,)),
        (
            ir.MatMul(tensor_result, tensor_lhs, tensor_rhs),
            (tensor_lhs, tensor_rhs),
            (tensor_result,),
        ),
        (
            ir.Call("callee", (lhs, rhs), (scalar_result,)),
            (lhs, rhs),
            (scalar_result,),
        ),
        (ir.Return((lhs,)), (lhs,), ()),
        (ir.Yield((rhs,)), (rhs,), ()),
        (
            ir.If(
                (scalar_result,),
                condition,
                then_region,
                else_region,
            ),
            (condition,),
            (scalar_result,),
        ),
        (
            ir.UnknownOp("test.op", (lhs,), (scalar_result,)),
            (lhs,),
            (scalar_result,),
        ),
    ]

    operations.extend(
        [
            (ir.Call("callee", (), ()), (), ()),
            (ir.Call("callee", (lhs,), (lhs, rhs)), (lhs,), (lhs, rhs)),
            (ir.Return(), (), ()),
            (ir.Yield(), (), ()),
            (ir.If((), condition, then_region, else_region), (condition,), ()),
            (
                ir.If((lhs, rhs), condition, then_region, else_region),
                (condition,),
                (lhs, rhs),
            ),
            (ir.UnknownOp("test.empty", (), ()), (), ()),
            (
                ir.UnknownOp("test.multi", (rhs, lhs), (lhs, rhs)),
                (rhs, lhs),
                (lhs, rhs),
            ),
        ]
    )

    for operation, operands, results in operations:
        actual_operands = ir.get_operands(operation)
        actual_results = ir.get_results(operation)
        assert actual_operands == operands
        assert actual_results == results
        assert all(
            actual is expected
            for actual, expected in zip(actual_operands, operands, strict=True)
        )
        assert all(
            actual is expected
            for actual, expected in zip(actual_results, results, strict=True)
        )
        regions = ir.get_regions(operation)
        if isinstance(operation, ir.If):
            assert len(regions) == 2
            assert regions[0] is then_region
            assert regions[1] is else_region
        else:
            assert regions == ()


def test_nested_regions_preserve_identity_and_are_not_flattened() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    nested = ir.If((), condition, ir.Region([ir.Block()]), ir.Region([ir.Block()]))
    block = ir.Block(operations=[nested])
    outer = ir.If((), condition, ir.Region([block]), ir.Region([ir.Block()]))

    regions = ir.get_regions(outer)
    assert len(regions) == 2
    assert regions[0].blocks[0] is block
    assert regions[0].blocks[0].operations[0] is nested
    assert ir.get_regions(nested)[0] is nested.then_region
    assert ir.get_operands(outer) == (condition,)
    assert ir.get_results(outer) == ()


def test_iter_defined_values_preserves_depth_first_order_and_identity() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    argument, result, nested_argument, nested_result, else_result, next_argument = (
        ir.Value(ir.ValueId(index), ir.ScalarType.F32) for index in range(1, 7)
    )
    branch = ir.If(
        (result,),
        condition,
        ir.Region(
            [
                ir.Block(
                    arguments=(nested_argument,),
                    operations=[
                        ir.Add(nested_result, nested_argument, argument),
                        ir.Yield((nested_result,)),
                    ],
                )
            ]
        ),
        ir.Region(
            [
                ir.Block(
                    operations=[ir.Const(else_result, 1.0), ir.Yield((else_result,))]
                )
            ]
        ),
    )
    region = ir.Region(
        [
            ir.Block(arguments=(condition, argument), operations=[branch]),
            ir.Block(arguments=(next_argument,), operations=[ir.Return((result,))]),
        ]
    )

    expected = [
        condition,
        argument,
        result,
        nested_argument,
        nested_result,
        else_result,
        next_argument,
    ]
    assert all(
        actual is value
        for actual, value in zip(ir.iter_defined_values(region), expected, strict=True)
    )


def test_iter_defined_values_preserves_repeated_and_multiple_results() -> None:
    first = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    second = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    region = ir.Region(
        [
            ir.Block(
                arguments=(first,),
                operations=[ir.Call("callee", (first,), (second, first))],
            )
        ]
    )

    assert list(ir.iter_defined_values(region)) == [first, second, first]


def test_iter_defined_values_ignores_operand_only_references() -> None:
    external = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    for region in (
        ir.Region(),
        ir.Region([ir.Block()]),
        ir.Region([ir.Block(operations=[ir.Return((external,))])]),
    ):
        assert list(ir.iter_defined_values(region)) == []


def test_iter_blocks_preserves_depth_first_order_and_identity() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    leaf_then, leaf_else, outer_else, sibling, last = (ir.Block() for _ in range(5))
    nested = ir.If((), condition, ir.Region([leaf_then]), ir.Region([leaf_else]))
    then_block = ir.Block(operations=[nested])
    outer = ir.If(
        (), condition, ir.Region([then_block, sibling]), ir.Region([outer_else])
    )
    next_block = ir.Block()
    next_op = ir.If((), condition, ir.Region(), ir.Region([next_block]))
    entry = ir.Block(operations=[outer, next_op])
    region = ir.Region([entry, last])

    expected = [
        entry,
        then_block,
        leaf_then,
        leaf_else,
        sibling,
        outer_else,
        next_block,
        last,
    ]
    assert all(
        actual is block
        for actual, block in zip(ir.iter_blocks(region), expected, strict=True)
    )


def test_iter_blocks_handles_empty_regions_and_blocks() -> None:
    assert list(ir.iter_blocks(ir.Region())) == []
    block = ir.Block()
    blocks = list(ir.iter_blocks(ir.Region([block])))
    assert len(blocks) == 1
    assert blocks[0] is block


def test_iter_ops_preserves_depth_first_order_and_identity() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    leaf = ir.Yield()
    nested = ir.If((), condition, ir.Region([ir.Block(operations=[leaf])]), ir.Region())
    then_end = ir.Yield()
    else_end = ir.Yield()
    outer = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[nested, then_end])]),
        ir.Region([ir.Block(operations=[else_end])]),
    )
    first = ir.Const(condition, True)
    last = ir.Return()
    region = ir.Region(
        [ir.Block(operations=[first, outer]), ir.Block(operations=[last])]
    )

    expected = [first, outer, nested, leaf, then_end, else_end, last]
    assert all(
        actual is operation
        for actual, operation in zip(ir.iter_ops(region), expected, strict=True)
    )
    assert list(ir.iter_ops(ir.Region())) == []
    assert list(ir.iter_ops(ir.Region([ir.Block()]))) == []


def test_branch_accessors_preserve_edge_and_operand_order() -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    x = ir.Value(ir.ValueId(1), ir.ScalarType.I32)
    target = ir.Block()
    jump = ir.Branch(target, (x,))
    cond = ir.CondBranch(flag, target, target, (x, x), (x,))
    target.operations = [jump]
    body = ir.Region([ir.Block((flag, x), [cond]), target])
    fn = ir.Function("f", ir.FunctionType((flag.type, x.type), ()), body)
    assert ir.get_operands(jump) == (x,)
    assert ir.get_operands(cond) == (flag, x, x, x)
    assert ir.get_successors(cond) == (target, target)
    assert ir.get_successors(jump) == (target,)
    assert ir.get_successors(ir.Return()) == ()
    for op in (jump, cond):
        assert ir.get_results(op) == ()
        assert ir.get_regions(op) == ()
    assert list(ir.iter_ops(body)) == [cond, jump]
    assert list(ir.iter_uses(fn, x.id)) == [
        ir.Use(cond, 1),
        ir.Use(cond, 2),
        ir.Use(cond, 3),
        ir.Use(jump, 0),
    ]
