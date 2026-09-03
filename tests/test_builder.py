import pytest

from niro import ir
from niro.builder import FunctionBuilder, ModuleBuilder


def function_builder() -> FunctionBuilder:
    return ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((), ()),
    )


def test_rejects_second_block_in_region() -> None:
    region = function_builder().region()
    region.block()

    with pytest.raises(ValueError, match="multiple blocks"):
        region.block()


def test_rejects_operation_after_terminator() -> None:
    block = function_builder().region().block()
    block.return_()

    with pytest.raises(ValueError, match="after a block terminator"):
        block.i32(1)

    assert block.ir.operations == [ir.Return()]


def test_rejects_call_to_unknown_function_name() -> None:
    caller = function_builder().region().block()
    callee = ModuleBuilder().function(name="callee", type=ir.FunctionType((), ()))

    with pytest.raises(ValueError, match="unknown function"):
        caller.call(callee)


def test_const() -> None:
    block = function_builder().region().block()
    result = block.const(1, ir.ScalarType.I32)

    assert block.ir.operations == [ir.Const(result, 1)]


def test_transpose() -> None:
    block = function_builder().region().block()
    operand = block.tensor(
        bytes(24),
        ir.TensorType(ir.ScalarType.F32, (2, 3)),
    )
    result = block.transpose(operand, (1, 0))

    assert block.ir.operations[-1] == ir.Transpose(result, operand, (1, 0))


def test_add() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.add(lhs, rhs)

    assert block.ir.operations[-1] == ir.Add(result, lhs, rhs)


def test_mul() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.mul(lhs, rhs)

    assert block.ir.operations[-1] == ir.Mul(result, lhs, rhs)


def test_matmul() -> None:
    block = function_builder().region().block()
    lhs = block.tensor(bytes(24), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = block.tensor(bytes(48), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    result = block.matmul(lhs, rhs)

    assert block.ir.operations[-1] == ir.MatMul(result, lhs, rhs)


def test_call() -> None:
    module = ModuleBuilder()
    callee = module.function(name="callee", type=ir.FunctionType((), ()))
    caller = module.function(name="caller", type=ir.FunctionType((), ()))
    block = caller.region().block()

    results = block.call(callee)

    assert results == ()
    assert block.ir.operations == [ir.Call("callee", (), ())]


def test_return() -> None:
    block = function_builder().region().block()
    block.return_()

    assert block.ir.operations == [ir.Return()]


def test_yield() -> None:
    block = function_builder().region().block()
    block.yield_()

    assert block.ir.operations == [ir.Yield()]


def test_if() -> None:
    block = function_builder().region().block()
    condition = block.bool(True)
    conditional = block.if_(condition)

    assert block.ir.operations[-1] is conditional.ir
    assert isinstance(conditional.ir, ir.If)


def test_unknown_op() -> None:
    block = function_builder().region().block()

    results = block.unknown_op("example.op")

    assert results == ()
    assert block.ir.operations == [ir.UnknownOp("example.op", (), ())]


def test_get_global() -> None:
    module = ModuleBuilder()
    global_ = module.global_("answer", ir.ScalarType.I32, 42)
    function = module.function(name="main", type=ir.FunctionType((), ()))
    block = function.region().block()

    result = block.get_global(global_)

    assert result.type is ir.ScalarType.I32
    assert block.ir.operations == [ir.GetGlobal("answer", result)]


def test_global_and_function_names_share_namespace() -> None:
    module = ModuleBuilder()
    module.global_("main", ir.ScalarType.I32, 42)

    with pytest.raises(ValueError, match="symbol names must be unique"):
        module.function(name="main", type=ir.FunctionType((), ()))
