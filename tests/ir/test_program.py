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


def test_preserves_optional_interface_names() -> None:
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
    assert function.output_names == (None,)
