from niro import ir

Operands = tuple[ir.Value, ...]
Results = tuple[ir.Value, ...]


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
