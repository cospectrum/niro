import pytest

from niro.ir import (
    Add,
    Block,
    Const,
    Function,
    FunctionType,
    MatMul,
    Region,
    ScalarType,
    TensorType,
    Value,
    ValueId,
)


def test_function_arguments_come_from_entry_block() -> None:
    lhs = Value(ValueId(0), ScalarType.F32)
    rhs = Value(ValueId(1), ScalarType.F32)
    entry = Block(arguments=(lhs, rhs))
    function = Function(
        "add",
        FunctionType((ScalarType.F32, ScalarType.F32), (ScalarType.F32,)),
        Region([entry]),
    )

    assert function.arguments == (lhs, rhs)


def test_external_function_has_no_arguments_until_called() -> None:
    external = Function(
        "print_f32",
        FunctionType((ScalarType.F32,), ()),
    )

    assert external.body is None
    assert external.arguments == ()


def test_rejects_invalid_tensor_dimension_at_construction() -> None:
    with pytest.raises(ValueError, match="dimensions cannot be negative"):
        TensorType(element_type=ScalarType.F32, shape=(2, -1))


def test_rejects_invalid_constant_at_construction() -> None:
    result = Value(ValueId(0), TensorType(ScalarType.F32, (2,)))

    with pytest.raises(ValueError, match="4 bytes, expected 8"):
        Const(result=result, value=bytes(4))


def test_rejects_boolean_arithmetic_at_construction() -> None:
    lhs = Value(ValueId(0), ScalarType.BOOL)
    rhs = Value(ValueId(1), ScalarType.BOOL)
    result = Value(ValueId(2), ScalarType.BOOL)

    with pytest.raises(TypeError, match="does not support boolean"):
        Add(result=result, lhs=lhs, rhs=rhs)


def test_matmul_result_type_is_an_invariant() -> None:
    lhs = Value(ValueId(0), TensorType(ScalarType.F32, (2, 3)))
    rhs = Value(ValueId(1), TensorType(ScalarType.F32, (3, 4)))
    valid = Value(ValueId(2), TensorType(ScalarType.F32, (2, 4)))
    invalid = Value(ValueId(2), TensorType(ScalarType.F32, (2, 3)))

    MatMul(result=valid, lhs=lhs, rhs=rhs)
    with pytest.raises(TypeError, match="result type does not match"):
        MatMul(result=invalid, lhs=lhs, rhs=rhs)
