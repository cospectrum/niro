import pytest

from niro import ir
from niro.builder import ENTRY_POINT_ATTR, ModuleBuilder


def test_builds_function_and_call_with_function_wide_value_ids() -> None:
    module = ModuleBuilder()
    add = module.func(
        "add",
        arg_types=[ir.ScalarType.F32, ir.ScalarType.F32],
        ret_types=[ir.ScalarType.F32],
    )
    lhs, rhs = add.args
    result = add.add(lhs, rhs)
    add.return_(result)

    main = module.func("main", ret_types=[ir.ScalarType.F32])
    call_result = main.call(add, [main.f32(1.0), main.f32(2.0)])
    assert isinstance(call_result, ir.Value)
    main.return_(call_result)
    module.entry_point(main)

    assert [value.id for value in add.args] == [ir.ValueId(0), ir.ValueId(1)]
    assert result.id == ir.ValueId(2)
    assert call_result.id == ir.ValueId(2)
    assert module.ir.attributes[ENTRY_POINT_ATTR] == "main"
    assert main.function.body is not None
    assert isinstance(main.function.body.blocks[0].operations[-2], ir.Call)


def test_calls_external_function_by_name() -> None:
    module = ModuleBuilder()
    module.extern("print_f32", [ir.ScalarType.F32])
    main = module.func("main")

    assert main.call("print_f32", main.f32(1.0)) is None
    main.return_()


def test_rejects_values_from_another_function() -> None:
    module = ModuleBuilder()
    first = module.func("first", [ir.ScalarType.F32])
    second = module.func("second", [ir.ScalarType.F32])

    with pytest.raises(ValueError, match="does not belong"):
        second.add(first.args[0], second.args[0])


def test_rejects_operation_after_return() -> None:
    module = ModuleBuilder()
    function = module.func("main")
    function.return_()

    with pytest.raises(ValueError, match="after return"):
        function.f32(1.0)
