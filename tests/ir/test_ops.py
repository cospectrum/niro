import pytest

from niro import ir

type Operands = tuple[ir.Value, ...]
type Results = tuple[ir.Value, ...]


def test_operation_is_the_runtime_base_for_op_variants() -> None:
    operation = ir.Return()

    assert isinstance(operation, ir.Operation)
    with pytest.raises(TypeError, match="abstract class"):
        ir.Operation()


def test_as_op_rejects_extension_operations() -> None:
    class ExtensionOp(ir.Operation):
        def get_operands(self) -> tuple[ir.Value, ...]:
            return ()

        def get_results(self) -> tuple[ir.Value, ...]:
            return ()

        def is_terminator(self) -> bool:
            return False

    assert ir.as_op(ir.Return()) == ir.Return()
    with pytest.raises(TypeError, match="not a built-in Niro operation"):
        ir.as_op(ExtensionOp())


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


def test_rejects_invalid_constant_at_construction() -> None:
    result = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, (2,)))

    with pytest.raises(ValueError, match="4 bytes, expected 8"):
        ir.Const(result=result, literal=bytes(4))


def test_rejects_boolean_arithmetic_at_construction() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(2), ir.ScalarType.BOOL)

    with pytest.raises(TypeError, match="does not support boolean"):
        ir.Add(result=result, lhs=lhs, rhs=rhs)


def test_matmul_result_type_is_an_invariant() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = ir.Value(ir.ValueId(1), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    valid = ir.Value(ir.ValueId(2), ir.TensorType(ir.ScalarType.F32, (2, 4)))
    invalid = ir.Value(ir.ValueId(2), ir.TensorType(ir.ScalarType.F32, (2, 3)))

    ir.MatMul(result=valid, lhs=lhs, rhs=rhs)
    with pytest.raises(TypeError, match="result type does not match"):
        ir.MatMul(result=invalid, lhs=lhs, rhs=rhs)


def test_rejects_empty_global_name_at_construction() -> None:
    result = ir.Value(ir.ValueId(0), ir.ScalarType.I32)

    with pytest.raises(ValueError, match="global name cannot be empty"):
        ir.GetGlobal(name="", result=result)
