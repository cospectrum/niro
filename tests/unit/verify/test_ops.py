import pytest

from niro import ir, verify


def _module_for_op(op: ir.Op) -> ir.Module:
    arguments = ir.get_operands(op)
    results = ir.get_results(op)
    return ir.Module(
        functions=[
            ir.Function(
                name="main",
                type=ir.FunctionType(
                    tuple(value.type for value in arguments),
                    tuple(value.type for value in results),
                ),
                body=ir.Region([ir.Block(arguments, [op, ir.Return(results)])]),
            )
        ]
    )


@pytest.mark.parametrize(
    ("type", "literal"),
    [
        (ir.ScalarType.BOOL, True),
        (ir.ScalarType.I32, -1),
        (ir.ScalarType.I64, 42),
        (ir.ScalarType.F32, 1.5),
        (ir.ScalarType.F64, 2.5),
        (ir.TensorType(ir.ScalarType.F32, (2, 3)), bytes(24)),
        (ir.TensorType(ir.ScalarType.I64, ()), bytes(8)),
        (ir.TensorType(ir.ScalarType.BOOL, (2,)), bytes(2)),
        (ir.TensorType(ir.ScalarType.I32, (0, 3)), b""),
    ],
)
def test_valid_constants(type: ir.Type, literal: ir.Literal) -> None:
    module = _module_for_op(ir.Const(ir.Value(ir.ValueId(0), type), literal))

    assert verify.module(module) is module


@pytest.mark.parametrize(
    ("type", "literal", "error", "message"),
    [
        (ir.ScalarType.BOOL, 1, TypeError, "constant value"),
        (ir.ScalarType.I32, True, TypeError, "constant value"),
        (ir.ScalarType.I64, 1.5, TypeError, "constant value"),
        (ir.ScalarType.F32, 1, TypeError, "constant value"),
        (ir.ScalarType.F64, b"", TypeError, "constant value"),
        (
            ir.TensorType(ir.ScalarType.F32, (2,)),
            bytes(4),
            ValueError,
            "4 bytes, expected 8",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2,)),
            bytes(12),
            ValueError,
            "12 bytes, expected 8",
        ),
        (ir.TensorType(ir.ScalarType.F32, (None,)), b"", TypeError, "static shape"),
        (ir.TensorType(ir.ScalarType.F32, None), b"", TypeError, "static shape"),
        (ir.TensorType(ir.ScalarType.F32, ()), 1.0, TypeError, "packed bytes"),
    ],
)
def test_invalid_constants(
    type: ir.Type, literal: ir.Literal, error: type[Exception], message: str
) -> None:
    module = _module_for_op(ir.Const(ir.Value(ir.ValueId(0), type), literal))

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize("operation", [ir.Add, ir.Mul])
@pytest.mark.parametrize(
    "type",
    [ir.ScalarType.I32, ir.ScalarType.F64, ir.TensorType(ir.ScalarType.F32, (None, 3))],
)
def test_valid_arithmetic(operation: type[ir.Add | ir.Mul], type: ir.Type) -> None:
    lhs, rhs, result = (ir.Value(ir.ValueId(index), type) for index in range(3))
    module = _module_for_op(operation(result, lhs, rhs))

    assert verify.module(module) is module


@pytest.mark.parametrize("operation", [ir.Add, ir.Mul])
@pytest.mark.parametrize(
    ("lhs_type", "rhs_type", "result_type", "message"),
    [
        (ir.ScalarType.I32, ir.ScalarType.F32, ir.ScalarType.I32, "same type"),
        (ir.ScalarType.I32, ir.ScalarType.I32, ir.ScalarType.I64, "same type"),
        (ir.ScalarType.BOOL, ir.ScalarType.BOOL, ir.ScalarType.BOOL, "boolean"),
        (*[ir.TensorType(ir.ScalarType.BOOL, (2,))] * 3, "boolean"),
        (
            ir.TensorType(ir.ScalarType.F32, (2,)),
            ir.TensorType(ir.ScalarType.F32, (3,)),
            ir.TensorType(ir.ScalarType.F32, (2,)),
            "same type",
        ),
    ],
)
def test_invalid_arithmetic(
    operation: type[ir.Add | ir.Mul],
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    result_type: ir.Type,
    message: str,
) -> None:
    op = operation(
        ir.Value(ir.ValueId(2), result_type),
        ir.Value(ir.ValueId(0), lhs_type),
        ir.Value(ir.ValueId(1), rhs_type),
    )

    with pytest.raises(TypeError, match=message):
        verify.module(_module_for_op(op))


def _matmul(lhs: ir.Type, rhs: ir.Type, result: ir.Type) -> ir.MatMul:
    return ir.MatMul(
        ir.Value(ir.ValueId(2), result),
        ir.Value(ir.ValueId(0), lhs),
        ir.Value(ir.ValueId(1), rhs),
    )


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "result_shape"),
    [((2, 3), (3, 4), (2, 4)), ((None, 3), (None, 4), (None, 4))],
)
def test_valid_matmul(
    lhs_shape: ir.Shape, rhs_shape: ir.Shape, result_shape: ir.Shape
) -> None:
    op = _matmul(
        *(
            ir.TensorType(ir.ScalarType.F32, shape)
            for shape in (lhs_shape, rhs_shape, result_shape)
        )
    )
    module = _module_for_op(op)

    assert verify.module(module) is module


@pytest.mark.parametrize(
    ("lhs", "rhs", "result", "error", "message"),
    [
        (
            ir.ScalarType.F32,
            ir.ScalarType.F32,
            ir.ScalarType.F32,
            TypeError,
            "must be tensors",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, None),
            ir.TensorType(ir.ScalarType.F32, (3, 4)),
            ir.TensorType(ir.ScalarType.F32, None),
            TypeError,
            "ranked tensors",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (3,)),
            ir.TensorType(ir.ScalarType.F32, (3, 4)),
            ir.TensorType(ir.ScalarType.F32, (4,)),
            TypeError,
            "rank-two",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.I32, (3, 4)),
            ir.TensorType(ir.ScalarType.F32, (2, 4)),
            TypeError,
            "element types",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (2, 4)),
            ir.TensorType(ir.ScalarType.F32, (2, 4)),
            ValueError,
            "contracting dimensions",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (3, 4)),
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            TypeError,
            "result type",
        ),
    ],
)
def test_invalid_matmul(
    lhs: ir.Type, rhs: ir.Type, result: ir.Type, error: type[Exception], message: str
) -> None:
    module = _module_for_op(_matmul(lhs, rhs, result))

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize(
    ("shape", "permutation", "result_shape"),
    [
        ((2, 3), (1, 0), (3, 2)),
        ((None, 3), (1, 0), (3, None)),
        ((), (), ()),
        (None, (1, 0), None),
    ],
)
def test_valid_transpose(
    shape: ir.Shape | None, permutation: tuple[int, ...], result_shape: ir.Shape | None
) -> None:
    operand = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, shape))
    result = ir.Value(ir.ValueId(1), ir.TensorType(ir.ScalarType.F32, result_shape))
    module = _module_for_op(ir.Transpose(result, operand, permutation))

    assert verify.module(module) is module


@pytest.mark.parametrize(
    ("operand_type", "result_type", "permutation", "error", "message"),
    [
        (ir.ScalarType.F32, ir.ScalarType.F32, (), TypeError, "must be a tensor"),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (3, 2)),
            (0, 0),
            ValueError,
            "every dimension once",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (3, 2)),
            (0,),
            ValueError,
            "every dimension once",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            (1, 0),
            TypeError,
            "result type",
        ),
    ],
)
def test_invalid_transpose(
    operand_type: ir.Type,
    result_type: ir.Type,
    permutation: tuple[int, ...],
    error: type[Exception],
    message: str,
) -> None:
    op = ir.Transpose(
        ir.Value(ir.ValueId(1), result_type),
        ir.Value(ir.ValueId(0), operand_type),
        permutation,
    )

    with pytest.raises(error, match=message):
        verify.module(_module_for_op(op))


def test_unknown_operation_requires_a_name() -> None:
    with pytest.raises(ValueError, match="UnknownOp name cannot be empty"):
        verify.module(_module_for_op(ir.UnknownOp("", (), ())))


def test_invalid_constant_in_nested_branch_is_verified() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    conditional = ir.If(
        (),
        condition,
        ir.Region([ir.Block(operations=[ir.Const(result, 1), ir.Yield()])]),
        ir.Region(),
    )

    with pytest.raises(TypeError, match="constant value"):
        verify.module(_module_for_op(conditional))
