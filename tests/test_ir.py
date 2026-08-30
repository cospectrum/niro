from niro.ir import (
    Block,
    Function,
    FunctionType,
    Region,
    ScalarType,
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
