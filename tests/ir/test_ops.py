import pytest

from niro import ir


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
    operations = [
        (ir.Const(ir.OpId(0), scalar_result, 1.0), (), (scalar_result,)),
        (
            ir.Transpose(ir.OpId(0), transpose_result, tensor_lhs, (1, 0)),
            (tensor_lhs,),
            (transpose_result,),
        ),
        (ir.Add(ir.OpId(0), scalar_result, lhs, rhs), (lhs, rhs), (scalar_result,)),
        (ir.Mul(ir.OpId(0), scalar_result, lhs, rhs), (lhs, rhs), (scalar_result,)),
        (
            ir.MatMul(ir.OpId(0), tensor_result, tensor_lhs, tensor_rhs),
            (tensor_lhs, tensor_rhs),
            (tensor_result,),
        ),
        (
            ir.Call(ir.OpId(0), "callee", (lhs, rhs), (scalar_result,)),
            (lhs, rhs),
            (scalar_result,),
        ),
        (ir.Return(ir.OpId(0), (lhs,)), (lhs,), ()),
        (ir.Yield(ir.OpId(0), (rhs,)), (rhs,), ()),
        (
            ir.If(
                ir.OpId(0),
                (scalar_result,),
                condition,
                empty_region,
                empty_region,
            ),
            (condition,),
            (scalar_result,),
        ),
        (
            ir.UnknownOp(ir.OpId(0), "test.op", (lhs,), (scalar_result,)),
            (lhs,),
            (scalar_result,),
        ),
    ]

    # Constructing every operation above also exercises successful __post_init__.
    for operation, operands, results in operations:
        assert operation.get_operands() == operands
        assert operation.get_results() == results


def test_rejects_invalid_constant_at_construction() -> None:
    result = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, (2,)))

    with pytest.raises(ValueError, match="4 bytes, expected 8"):
        ir.Const(id=ir.OpId(0), result=result, literal=bytes(4))


def test_rejects_boolean_arithmetic_at_construction() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(2), ir.ScalarType.BOOL)

    with pytest.raises(TypeError, match="does not support boolean"):
        ir.Add(id=ir.OpId(0), result=result, lhs=lhs, rhs=rhs)


def test_matmul_result_type_is_an_invariant() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = ir.Value(ir.ValueId(1), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    valid = ir.Value(ir.ValueId(2), ir.TensorType(ir.ScalarType.F32, (2, 4)))
    invalid = ir.Value(ir.ValueId(2), ir.TensorType(ir.ScalarType.F32, (2, 3)))

    ir.MatMul(id=ir.OpId(0), result=valid, lhs=lhs, rhs=rhs)
    with pytest.raises(TypeError, match="result type does not match"):
        ir.MatMul(id=ir.OpId(0), result=invalid, lhs=lhs, rhs=rhs)


def test_rejects_negative_operation_id_at_construction() -> None:
    result = ir.Value(ir.ValueId(0), ir.ScalarType.I32)

    with pytest.raises(ValueError, match="operation ID cannot be negative"):
        ir.Const(id=ir.OpId(-1), result=result, literal=1)
