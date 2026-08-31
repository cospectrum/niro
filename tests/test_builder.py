from typing import cast

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


def test_tensor_constant_requires_bytes() -> None:
    function = ModuleBuilder().func("main")
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))

    value = function.tensor(
        data=b"\x00\x00\x00@\x00\x00@@",
        result_type=tensor_type,
    )

    assert value.type == tensor_type
    assert function.function.body is not None
    assert function.function.body.blocks[0].operations == [
        ir.Const(result=value, value=b"\x00\x00\x00@\x00\x00@@")
    ]

    with pytest.raises(TypeError, match="requires packed bytes"):
        function.tensor(
            data=cast(bytes, (2.0, 3.0)),
            result_type=tensor_type,
        )


def test_builds_unknown_operation() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))
    function = ModuleBuilder().func(
        name="main",
        arg_types=[tensor_type],
        ret_types=[tensor_type],
    )

    (result,) = function.unknown(
        name="onnx.Relu",
        operands=function.args,
        result_types=[tensor_type],
        attributes={"alpha": 1.0},
    )
    function.return_(result)

    assert function.function.body is not None
    operation = function.function.body.blocks[0].operations[0]
    assert operation == ir.UnknownOp(
        name="onnx.Relu",
        operands=function.args,
        results=(result,),
        attributes={"alpha": 1.0},
    )


def test_failed_operation_does_not_allocate_a_value() -> None:
    function = ModuleBuilder().func(name="main")
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))

    with pytest.raises(ValueError, match="expected 8"):
        function.tensor(data=bytes(4), result_type=tensor_type)

    assert function.i32(1).id == ir.ValueId(0)


def test_matmul_derives_its_result_type() -> None:
    function = ModuleBuilder().func(
        name="main",
        arg_types=[
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (3, 4)),
        ],
    )

    result = function.matmul(function.args[0], function.args[1])

    assert result.type == ir.TensorType(ir.ScalarType.F32, (2, 4))
