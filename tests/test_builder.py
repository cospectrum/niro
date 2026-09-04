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


def test_function_first_block_arguments_match_function_inputs() -> None:
    function = ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32, ir.ScalarType.I64), ()),
    )

    block = function.region().first_block()

    assert tuple(argument.type for argument in block.inner.arguments) == (
        ir.ScalarType.F32,
        ir.ScalarType.I64,
    )


def test_function_block_rejects_arguments_that_do_not_match_inputs() -> None:
    function = ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )

    with pytest.raises(ir.VerificationError, match="must match region input types"):
        function.region().block((ir.ScalarType.I64,))


def test_nested_region_first_block_has_no_function_arguments() -> None:
    function = ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )
    block = function.region().first_block()
    conditional = block.if_(block.bool(True))

    then_block = conditional.then_region.first_block()
    else_block = conditional.else_region.first_block()

    assert then_block.inner.arguments == ()
    assert else_block.inner.arguments == ()


def test_rejects_operation_after_terminator() -> None:
    block = function_builder().region().block()
    block.return_()

    with pytest.raises(ValueError, match="after a block terminator"):
        block.i32(1)

    assert block.inner.operations == [ir.Return()]


def test_rejects_call_to_unknown_function_name() -> None:
    caller = function_builder().region().block()
    callee = ModuleBuilder().function(name="callee", type=ir.FunctionType((), ()))

    with pytest.raises(ValueError, match="unknown function"):
        caller.call(callee)


def test_const() -> None:
    block = function_builder().region().block()
    result = block.const(1, ir.ScalarType.I32)

    assert block.inner.operations == [ir.Const(result, 1)]


def test_transpose() -> None:
    block = function_builder().region().block()
    operand = block.tensor(
        bytes(24),
        ir.TensorType(ir.ScalarType.F32, (2, 3)),
    )
    result = block.transpose(operand, (1, 0))

    assert block.inner.operations[-1] == ir.Transpose(result, operand, (1, 0))


def test_add() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.add(lhs, rhs)

    assert block.inner.operations[-1] == ir.Add(result, lhs, rhs)


def test_mul() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.mul(lhs, rhs)

    assert block.inner.operations[-1] == ir.Mul(result, lhs, rhs)


def test_matmul() -> None:
    block = function_builder().region().block()
    lhs = block.tensor(bytes(24), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = block.tensor(bytes(48), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    result = block.matmul(lhs, rhs)

    assert block.inner.operations[-1] == ir.MatMul(result, lhs, rhs)


def test_call() -> None:
    module = ModuleBuilder()
    callee = module.function(name="callee", type=ir.FunctionType((), ()))
    caller = module.function(name="caller", type=ir.FunctionType((), ()))
    block = caller.region().block()

    results = block.call(callee)

    assert results == ()
    assert block.inner.operations == [ir.Call("callee", (), ())]


def test_return() -> None:
    block = function_builder().region().block()
    block.return_()

    assert block.inner.operations == [ir.Return()]


def test_yield() -> None:
    entry = function_builder().region().block()
    block = entry.if_(entry.bool(True)).then_region.first_block()
    block.yield_()

    assert block.inner.operations == [ir.Yield()]


def test_if() -> None:
    block = function_builder().region().block()
    condition = block.bool(True)
    conditional = block.if_(condition)

    assert block.inner.operations[-1] is conditional.inner
    assert isinstance(conditional.inner, ir.If)


def test_unknown_op() -> None:
    block = function_builder().region().block()

    results = block.unknown_op("example.op")

    assert results == ()
    assert block.inner.operations == [ir.UnknownOp("example.op", (), ())]


def test_get_global() -> None:
    module = ModuleBuilder()
    global_ = module.global_("answer", ir.ScalarType.I32, 42)
    function = module.function(name="main", type=ir.FunctionType((), ()))
    block = function.region().block()

    result = block.get_global(global_)

    assert result.type is ir.ScalarType.I32
    assert block.inner.operations == [ir.GetGlobal("answer", result)]


def test_global_and_function_names_share_namespace() -> None:
    module = ModuleBuilder()
    module.global_("main", ir.ScalarType.I32, 42)

    with pytest.raises(ValueError, match="symbol names must be unique"):
        module.function(name="main", type=ir.FunctionType((), ()))


def test_rejected_declarations_leave_module_unchanged() -> None:
    module = ModuleBuilder()
    with pytest.raises(ir.VerificationError, match="input names must match"):
        module.function(name="main", type=ir.FunctionType((), ()), input_names=("x",))
    with pytest.raises(ir.VerificationError, match="4 bytes, expected 8"):
        module.global_("weight", ir.TensorType(ir.ScalarType.F32, (2,)), bytes(4))
    with pytest.raises(ir.VerificationError, match="non-negative integer"):
        module.function(
            name="bad",
            type=ir.FunctionType((ir.TensorType(ir.ScalarType.F32, (-1,)),), ()),
        )
    assert module.inner == ir.Module()


def test_rejected_operations_are_not_appended() -> None:
    block = function_builder().region().block()
    boolean = block.bool(True)
    with pytest.raises(ir.VerificationError, match="does not support boolean"):
        block.add(boolean, boolean)
    with pytest.raises(ir.VerificationError, match="out of range"):
        block.i32(1 << 31)
    with pytest.raises(ir.VerificationError, match="expected Return"):
        block.yield_()
    with pytest.raises(ir.VerificationError, match="Return operand types"):
        block.return_(boolean)
    assert block.inner.operations == [ir.Const(boolean, True)]


def test_builder_checks_call_signature_before_append() -> None:
    module = ModuleBuilder()
    callee = module.function(
        name="callee", type=ir.FunctionType((ir.ScalarType.I32,), ())
    )
    block = (
        module.function(name="main", type=ir.FunctionType((), ()))
        .region()
        .first_block()
    )
    with pytest.raises(ir.VerificationError, match="call argument types"):
        block.call(callee)
    assert block.inner.operations == []


def test_builder_checks_if_branch_arguments_and_yields() -> None:
    entry = function_builder().region().first_block()
    conditional = entry.if_(entry.bool(True), (ir.ScalarType.I32,))
    with pytest.raises(ir.VerificationError, match="region input types"):
        conditional.then_region.block((ir.ScalarType.I32,))
    assert conditional.then_region.inner.blocks == []
    branch = conditional.then_region.first_block()
    with pytest.raises(ir.VerificationError, match="expected Yield"):
        branch.return_()
    with pytest.raises(ir.VerificationError, match="Yield operand types"):
        branch.yield_()
    branch.yield_(branch.i32(1))
