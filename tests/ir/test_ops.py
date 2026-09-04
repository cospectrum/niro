from niro import ir

type Operands = tuple[ir.Value, ...]
type Results = tuple[ir.Value, ...]


def test_operations_expose_generic_operands_and_results() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    scalar_result = ir.Value(ir.ValueId(2), ir.ScalarType.F32)
    condition = ir.Value(ir.ValueId(3), ir.ScalarType.BOOL)
    tensor_lhs = ir.Value(ir.ValueId(4), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    tensor_rhs = ir.Value(ir.ValueId(5), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    tensor_result = ir.Value(ir.ValueId(6), ir.TensorType(ir.ScalarType.F32, (2, 4)))
    transpose_result = ir.Value(ir.ValueId(7), ir.TensorType(ir.ScalarType.F32, (3, 2)))
    empty_region = ir.Region([ir.Block()])

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
                empty_region,
                empty_region,
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

    for operation, operands, results in operations:
        assert operation.get_operands() == operands
        assert operation.get_results() == results
