import pytest

from niro import ir


def test_function_arguments_come_from_entry_block() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    entry = ir.Block(arguments=(lhs, rhs))
    function = ir.Function(
        id=ir.FuncId(0),
        name="add",
        type=ir.FunctionType(
            (ir.ScalarType.F32, ir.ScalarType.F32), (ir.ScalarType.F32,)
        ),
        body=ir.Region([entry]),
    )

    assert function.arguments == (lhs, rhs)


def test_external_function_has_no_arguments_until_called() -> None:
    external = ir.Function(
        id=ir.FuncId(0),
        name="print_f32",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )

    assert external.body is None
    assert external.arguments == ()


def test_rejects_negative_function_id_at_construction() -> None:
    with pytest.raises(ValueError, match="function ID cannot be negative"):
        ir.Function(id=ir.FuncId(-1), name="main", type=ir.FunctionType((), ()))


def test_validates_optional_interface_names() -> None:
    function_type = ir.FunctionType(
        (ir.ScalarType.F32, ir.ScalarType.I64), (ir.ScalarType.F32,)
    )
    function = ir.Function(
        id=ir.FuncId(0),
        name="main",
        type=function_type,
        input_names=("value", None),
        output_names=(None,),
    )

    assert function.input_names == ("value", None)
    with pytest.raises(ValueError, match="input names must match input arity"):
        ir.Function(
            id=ir.FuncId(0), name="main", type=function_type, input_names=("x",)
        )
    with pytest.raises(ValueError, match="output names cannot be empty"):
        ir.Function(
            id=ir.FuncId(0), name="main", type=function_type, output_names=("",)
        )
