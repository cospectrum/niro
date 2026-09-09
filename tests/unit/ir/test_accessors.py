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
