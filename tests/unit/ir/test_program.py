from niro import ir, verify


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


def test_optional_interface_names() -> None:
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
    assert verify.module(ir.Module(functions=[function])).functions == [function]


def test_blocks_compare_and_hash_by_identity_through_mutation_and_cycles() -> None:
    first, second = ir.Block(), ir.Block()
    assert first != second
    members = {first, second}
    first.operations.append(ir.Branch(first))
    second.operations.append(ir.Branch(second))
    assert first != second
    assert first in members and second in members
    assert ir.Branch(first) == ir.Branch(first)
    assert ir.Branch(first) != ir.Branch(second)
