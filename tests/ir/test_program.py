import pytest

from niro import ir


def test_function_arguments_come_from_entry_block() -> None:
    lhs = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    rhs = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    entry = ir.Block(arguments=(lhs, rhs))
    function = ir.Function(
        name="add",
        type=ir.FunctionType(
            (ir.ScalarType.F32, ir.ScalarType.F32), (ir.ScalarType.F32,)
        ),
        body=ir.Region([entry]),
    )

    assert function.first_block
    assert function.first_block.arguments == (lhs, rhs)


def test_external_function_has_no_arguments_until_called() -> None:
    external = ir.Function(
        name="print_f32",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )

    assert external.body is None
    assert external.first_block is None


def test_rejects_empty_function_name() -> None:
    with pytest.raises(ValueError, match="function name cannot be empty"):
        ir.Function(name="", type=ir.FunctionType((), ()))


def test_validates_optional_interface_names() -> None:
    function_type = ir.FunctionType(
        (ir.ScalarType.F32, ir.ScalarType.I64), (ir.ScalarType.F32,)
    )
    function = ir.Function(
        name="main",
        type=function_type,
        input_names=("value", None),
        output_names=(None,),
    )

    assert function.input_names == ("value", None)
    with pytest.raises(ValueError, match="input names must match input arity"):
        ir.Function(name="main", type=function_type, input_names=("x",))
    with pytest.raises(ValueError, match="output names cannot be empty"):
        ir.Function(name="main", type=function_type, output_names=("",))


def test_functions_and_globals_share_symbol_namespace() -> None:
    function = ir.Function(name="value", type=ir.FunctionType((), ()))
    global_ = ir.Global(name="value", type=ir.ScalarType.I32, initializer=1)

    with pytest.raises(ValueError, match="symbol names must be unique"):
        ir.Module(functions=[function], globals=[global_])
