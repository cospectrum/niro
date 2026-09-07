import pytest

from niro import ir


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "expected_shape"),
    [
        ((2, 3), (3, 4), (2, 4)),
        ((None, 3), (3, None), (None, None)),
        ((2, None), (3, 4), (2, 4)),
    ],
)
def test_matmul_result_type(
    lhs_shape: ir.Shape, rhs_shape: ir.Shape, expected_shape: ir.Shape
) -> None:
    lhs = ir.TensorType(ir.ScalarType.F32, lhs_shape)
    rhs = ir.TensorType(ir.ScalarType.F32, rhs_shape)

    assert ir.infer.matmul_result_type(lhs, rhs) == ir.TensorType(
        ir.ScalarType.F32, expected_shape
    )


def test_matmul_rejects_incompatible_contracting_dimensions() -> None:
    lhs = ir.TensorType(ir.ScalarType.F32, (2, 3))
    rhs = ir.TensorType(ir.ScalarType.F32, (5, 4))

    with pytest.raises(ValueError, match="contracting dimensions must match"):
        ir.infer.matmul_result_type(lhs, rhs)


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [((2, 3), (3, 2)), ((None, 3), (3, None)), (None, None)],
)
def test_transpose_result_type(shape: ir.Shape, expected_shape: ir.Shape) -> None:
    operand = ir.TensorType(ir.ScalarType.F32, shape)

    assert ir.infer.transpose_result_type(operand, (1, 0)) == ir.TensorType(
        ir.ScalarType.F32, expected_shape
    )


def test_transpose_rejects_repeated_axes() -> None:
    operand = ir.TensorType(ir.ScalarType.F32, (2, 3))

    with pytest.raises(ValueError, match="must contain every dimension once"):
        ir.infer.transpose_result_type(operand, (0, 0))
