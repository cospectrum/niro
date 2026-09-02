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
    result = add.entry.add(lhs, rhs)
    add.entry.return_(result)

    main = module.func("main", ret_types=[ir.ScalarType.F32])
    call_result = main.entry.call(add, [main.entry.f32(1.0), main.entry.f32(2.0)])
    assert isinstance(call_result, ir.Value)
    main.entry.return_(call_result)
    module.set_entry_point(main)

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

    assert main.entry.call("print_f32", main.entry.f32(1.0)) is None
    main.entry.return_()


def test_rejects_duplicate_function_during_builder_construction() -> None:
    module = ModuleBuilder()
    module.func("main")

    with pytest.raises(ValueError, match="duplicate function"):
        module.func("main")

    assert [function.name for function in module.ir.functions] == ["main"]
    assert module.func("next").function.id == ir.FuncId(1)


def test_rejects_values_from_another_function() -> None:
    module = ModuleBuilder()
    first = module.func("first", [ir.ScalarType.F32])
    second = module.func("second", [ir.ScalarType.F32])

    with pytest.raises(ValueError, match="does not belong"):
        second.entry.add(first.args[0], second.args[0])


def test_rejects_operation_after_return() -> None:
    module = ModuleBuilder()
    function = module.func("main")
    function.entry.return_()

    with pytest.raises(ValueError, match="after a block terminator"):
        function.entry.f32(1.0)


def test_tensor_constant_requires_bytes() -> None:
    function = ModuleBuilder().func("main")
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))

    value = function.entry.tensor(
        data=b"\x00\x00\x00@\x00\x00@@",
        result_type=tensor_type,
    )

    assert value.type == tensor_type
    assert function.function.body is not None
    assert function.function.body.blocks[0].operations == [
        ir.Const(
            id=ir.OpId(0),
            result=value,
            literal=b"\x00\x00\x00@\x00\x00@@",
        )
    ]

    with pytest.raises(TypeError, match="requires packed bytes"):
        function.entry.tensor(
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

    (result,) = function.entry.unknown(
        name="onnx.Relu",
        operands=function.args,
        result_types=[tensor_type],
        attributes={"alpha": 1.0},
    )
    function.entry.return_(result)

    assert function.function.body is not None
    operation = function.function.body.blocks[0].operations[0]
    assert operation == ir.UnknownOp(
        id=ir.OpId(0),
        name="onnx.Relu",
        operands=function.args,
        results=(result,),
        attributes={"alpha": 1.0},
    )


def test_failed_operation_does_not_allocate_a_value() -> None:
    function = ModuleBuilder().func(name="main")
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))

    with pytest.raises(ValueError, match="expected 8"):
        function.entry.tensor(data=bytes(4), result_type=tensor_type)

    assert function.entry.i32(1).id == ir.ValueId(0)


def test_matmul_derives_its_result_type() -> None:
    function = ModuleBuilder().func(
        name="main",
        arg_types=[
            ir.TensorType(ir.ScalarType.F32, (2, 3)),
            ir.TensorType(ir.ScalarType.F32, (3, 4)),
        ],
    )

    result = function.entry.matmul(function.args[0], function.args[1])

    assert result.type == ir.TensorType(ir.ScalarType.F32, (2, 4))


def test_builds_multiple_blocks_with_function_wide_value_ids() -> None:
    function = ModuleBuilder().func(
        name="main",
        arg_types=[ir.ScalarType.I32],
    )

    second = function.body.block([ir.ScalarType.F32])
    result = second.f32(1.0)

    assert function.body.region is function.function.body
    assert function.entry is function.body.blocks[0]
    assert function.body.blocks == [function.entry, second]
    assert second.args[0].id == ir.ValueId(1)
    assert result.id == ir.ValueId(2)


def test_builds_if_regions_and_preserves_result_order() -> None:
    function = ModuleBuilder().func(
        name="main",
        ret_types=[ir.ScalarType.I32, ir.ScalarType.F32],
    )
    condition = function.entry.bool(True)
    conditional = function.entry.if_(
        condition,
        result_types=[ir.ScalarType.I32, ir.ScalarType.F32],
    )

    then_block = conditional.then_region.block()
    then_block.yield_(then_block.i32(1), then_block.f32(2.0))
    else_block = conditional.else_region.block()
    else_block.yield_(else_block.i32(3), else_block.f32(4.0))
    function.entry.return_(*conditional.results)

    assert conditional.operation.results == conditional.results
    assert [value.id for value in conditional.results] == [
        ir.ValueId(1),
        ir.ValueId(2),
    ]
    assert conditional.operation.then_region is conditional.then_region.region
    assert conditional.operation.else_region is conditional.else_region.region
    assert function.entry.block.operations == [
        ir.Const(id=ir.OpId(0), result=condition, literal=True),
        conditional.operation,
        ir.Return(id=ir.OpId(8), operands=conditional.results),
    ]


def test_allocates_function_ids_within_module() -> None:
    module = ModuleBuilder()

    first = module.func("first")
    external = module.extern("external")
    second = module.func("second")

    assert first.function.id == ir.FuncId(0)
    assert external.id == ir.FuncId(1)
    assert second.function.id == ir.FuncId(2)


def test_allocates_operation_ids_across_nested_regions() -> None:
    function = ModuleBuilder().func("main")
    condition = function.entry.bool(True)
    conditional = function.entry.if_(condition)
    conditional.then_region.block().yield_()
    conditional.else_region.block().yield_()
    function.entry.return_()

    assert function.function.body is not None
    assert [op.id for op in function.function.body.blocks[0].operations] == [
        ir.OpId(0),
        ir.OpId(1),
        ir.OpId(4),
    ]
    assert conditional.then_region.region.blocks[0].operations[0].id == ir.OpId(2)
    assert conditional.else_region.region.blocks[0].operations[0].id == ir.OpId(3)


def test_failed_operation_does_not_allocate_an_operation_id() -> None:
    function = ModuleBuilder().func("main")
    tensor_type = ir.TensorType(ir.ScalarType.F32, (2,))

    with pytest.raises(ValueError, match="expected 8"):
        function.entry.tensor(data=bytes(4), result_type=tensor_type)

    function.entry.i32(1)
    assert function.entry.block.operations[0].id == ir.OpId(0)


def test_if_requires_boolean_condition_without_allocating_results() -> None:
    function = ModuleBuilder().func(name="main")
    condition = function.entry.i32(1)

    with pytest.raises(TypeError, match="condition must be boolean"):
        function.entry.if_(condition, result_types=[ir.ScalarType.I32])

    assert function.entry.i32(2).id == ir.ValueId(1)
