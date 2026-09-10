from typing import assert_type

import pytest

from niro import builder, ir


def function_builder() -> builder.FunctionBuilder:
    return builder.ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((), ()),
    )


def test_appends_multiple_blocks_in_region() -> None:
    region = function_builder().region()
    first = region.block()
    second = region.block((ir.ScalarType.I32,))

    assert region.raw.blocks == [first.raw, second.raw]
    assert second.raw.arguments[0].type is ir.ScalarType.I32


def test_function_first_block_arguments_match_function_inputs() -> None:
    function = builder.ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32, ir.ScalarType.I64), ()),
    )

    block = function.region().first_block()

    assert tuple(argument.type for argument in block.raw.arguments) == (
        ir.ScalarType.F32,
        ir.ScalarType.I64,
    )


def test_function_block_accepts_explicit_argument_types() -> None:
    function = builder.ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )

    block = function.region().block((ir.ScalarType.I64,))

    assert block.raw.arguments[0].type is ir.ScalarType.I64


def test_nested_region_block_has_no_function_arguments() -> None:
    function = builder.ModuleBuilder().function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.F32,), ()),
    )
    block = function.region().first_block()
    conditional = block.if_(block.bool(True))

    then_block = conditional.then_region.block()
    else_block = conditional.else_region.block()

    assert then_block.raw.arguments == ()
    assert else_block.raw.arguments == ()


def test_value_supply_is_shared_across_regions_but_not_functions() -> None:
    module = builder.ModuleBuilder()
    function = module.function(
        name="f", type=ir.FunctionType((ir.ScalarType.BOOL,), (ir.ScalarType.I32,))
    )
    block = function.region().first_block()
    conditional = block.if_(block.raw.arguments[0], (ir.ScalarType.I32,))
    then_block = conditional.then_region.block()
    then_block.yield_(then_block.i32(1))
    else_block = conditional.else_region.block()
    else_block.yield_(else_block.i32(2))
    block.return_(conditional.raw.results[0])
    assert function.raw.body is not None
    ids = [value.id for value in ir.iter_defined_values(function.raw.body)]
    assert ids == [0, 1, 2, 3]

    other = module.function(name="g", type=ir.FunctionType((), ()))
    other_block = other.region().first_block()
    assert other_block.i32(3).id == 0
    other_block.return_()
    module.verify()


def test_appends_operation_after_terminator() -> None:
    block = function_builder().region().block()
    block.return_()

    result = block.i32(1)

    assert block.raw.operations == [ir.Return(), ir.Const(result, 1)]


def test_call_to_undeclared_function_builder_with_explicit_empty_results() -> None:
    caller = function_builder().region().block()
    callee = builder.ModuleBuilder().function(
        name="callee", type=ir.FunctionType((), ())
    )

    results = caller.call(callee, result_types=())

    assert results == ()
    assert caller.raw.operations == [ir.Call("callee", (), ())]


def test_const() -> None:
    block = function_builder().region().block()
    result = block.const(1, ir.ScalarType.I32)

    assert block.raw.operations == [ir.Const(result, 1)]


def test_transpose() -> None:
    block = function_builder().region().block()
    operand = block.tensor(
        bytes(24),
        ir.TensorType(ir.ScalarType.F32, (2, 3)),
    )
    result = block.transpose(operand, (1, 0))

    assert block.raw.operations[-1] == ir.Transpose(result, operand, (1, 0))


def test_add() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.add(lhs, rhs)

    assert block.raw.operations[-1] == ir.Add(result, lhs, rhs)


def test_mul() -> None:
    block = function_builder().region().block()
    lhs, rhs = block.i32(1), block.i32(2)
    result = block.mul(lhs, rhs)

    assert block.raw.operations[-1] == ir.Mul(result, lhs, rhs)


def test_matmul() -> None:
    block = function_builder().region().block()
    lhs = block.tensor(bytes(24), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = block.tensor(bytes(48), ir.TensorType(ir.ScalarType.F32, (3, 4)))
    result = block.matmul(lhs, rhs)

    assert block.raw.operations[-1] == ir.MatMul(result, lhs, rhs)


def test_call() -> None:
    module = builder.ModuleBuilder()
    callee = module.function(name="callee", type=ir.FunctionType((), ()))
    caller = module.function(name="caller", type=ir.FunctionType((), ()))
    block = caller.region().block()

    results = block.call(callee)

    assert results == ()
    assert block.raw.operations == [ir.Call("callee", (), ())]


def test_return() -> None:
    block = function_builder().region().block()
    block.return_()

    assert block.raw.operations == [ir.Return()]


def test_yield() -> None:
    block = function_builder().region().block()
    block.yield_()

    assert block.raw.operations == [ir.Yield()]


def test_if() -> None:
    block = function_builder().region().block()
    condition = block.bool(True)
    conditional = block.if_(condition)

    assert block.raw.operations[-1] is conditional.raw
    assert isinstance(conditional.raw, ir.If)


def test_unknown_op() -> None:
    block = function_builder().region().block()

    results = block.unknown_op("example.op")

    assert results == ()
    assert block.raw.operations == [ir.UnknownOp("example.op", (), ())]


def test_get_global() -> None:
    module = builder.ModuleBuilder()
    global_ = module.global_("answer", ir.ScalarType.I32, 42)
    function = module.function(name="main", type=ir.FunctionType((), ()))
    block = function.region().block()

    result = block.get_global(global_)

    assert result.type is ir.ScalarType.I32
    assert block.raw.operations == [ir.GetGlobal("answer", result)]


def test_allows_duplicate_symbols_during_construction() -> None:
    module = builder.ModuleBuilder()
    first_global = module.global_("main", ir.ScalarType.I32, 42)
    first_function = module.function(name="main", type=ir.FunctionType((), ()))
    second_global = module.global_("main", ir.ScalarType.I64, 43)
    second_function = module.function(name="main", type=ir.FunctionType((), ()))

    assert module.raw.globals == [first_global, second_global]
    assert module.raw.functions == [first_function.raw, second_function.raw]


def test_call_infers_results_without_checking_arguments() -> None:
    module = builder.ModuleBuilder()
    callee = module.function(
        name="callee", type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.F32,))
    )
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    argument = block.bool(True)

    for target in (callee, callee.raw, "callee"):
        results = block.call(target, (argument, argument))
        assert tuple(value.type for value in results) == (ir.ScalarType.F32,)
        assert block.raw.operations[-1] == ir.Call(
            "callee", (argument, argument), results
        )
    explicit = block.call(callee, result_types=[ir.ScalarType.F32])
    assert explicit[0].type is ir.ScalarType.F32


def test_call_forward_reference() -> None:
    module = builder.ModuleBuilder()
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    results = block.call("later", result_types=[ir.ScalarType.I32])
    operation = block.raw.operations[-1]
    module.function(name="later", type=ir.FunctionType((), (ir.ScalarType.I32,)))

    assert results[0].type is ir.ScalarType.I32
    assert block.raw.operations[-1] is operation
    assert operation == ir.Call("later", (), results)


def test_get_global_forward_reference() -> None:
    module = builder.ModuleBuilder()
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    result = block.get_global("later", type=ir.ScalarType.I32)
    operation = block.raw.operations[-1]
    global_ = module.global_("later", ir.ScalarType.I32, 42)

    assert result.type is ir.ScalarType.I32
    assert block.raw.operations[-1] is operation
    assert operation == ir.GetGlobal("later", result)
    assert block.get_global(global_, type=ir.ScalarType.I32).type is ir.ScalarType.I32
    assert block.get_global("later").type is ir.ScalarType.I32


def test_unresolved_object_targets_use_explicit_types() -> None:
    block = function_builder().region().block()
    function = ir.Function("later", ir.FunctionType((), (ir.ScalarType.F32,)))
    global_ = ir.Global("weight", ir.ScalarType.F32, 1.0)

    results = block.call(function, result_types=(ir.ScalarType.I32,))
    result = block.get_global(global_, type=ir.ScalarType.I32)

    assert results[0].type is ir.ScalarType.I32
    assert result.type is ir.ScalarType.I32


def test_resolution_uses_first_matching_module_declaration() -> None:
    module = builder.ModuleBuilder()
    module.function(name="callee", type=ir.FunctionType((), (ir.ScalarType.I32,)))
    duplicate = module.function(
        name="callee", type=ir.FunctionType((), (ir.ScalarType.F32,))
    )
    module.global_("weight", ir.ScalarType.I32, 1)
    duplicate_global = module.global_("weight", ir.ScalarType.F32, 1.0)
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()

    assert block.call(duplicate)[0].type is ir.ScalarType.I32
    assert block.get_global(duplicate_global).type is ir.ScalarType.I32


def test_verify_returns_raw_module_with_verified_type() -> None:
    module = builder.ModuleBuilder()
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    block.return_()

    verified = module.verify()

    assert_type(verified, ir.VerifiedModule)
    assert verified is module.raw


def test_verify_checks_current_builder_contents() -> None:
    module = builder.ModuleBuilder()
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    block.return_()
    module.verify()
    block.i32(1)

    with pytest.raises(ValueError, match="must end with Return"):
        module.verify()


def test_builds_function_cfg_containing_single_block_if() -> None:
    module = builder.ModuleBuilder()
    function = module.function(
        name="choose",
        type=ir.FunctionType(
            (ir.ScalarType.BOOL, ir.ScalarType.I32), (ir.ScalarType.I32,)
        ),
    )
    body = function.region()
    entry = body.first_block()
    flag, x = entry.raw.arguments
    join = body.block((x.type,))
    conditional = entry.if_(flag, (x.type,))
    start = conditional.then_region.block()
    start.yield_(x)
    conditional.else_region.block().yield_(x)
    entry.branch(join, *conditional.raw.results)
    join.return_(*join.raw.arguments)
    module.verify()
    assert start.raw.operations == [ir.Yield((x,))]
    assert entry.raw.operations[-1] == ir.Branch(join.raw, conditional.raw.results)
    assert start.raw.arguments == ()


def test_if_region_rejects_a_second_block() -> None:
    entry = function_builder().region().first_block()
    arm = entry.if_(entry.bool(True), (ir.ScalarType.I32,)).then_region
    start = arm.block()
    with pytest.raises(ValueError, match="already has a block"):
        arm.block()
    assert arm.raw.blocks == [start.raw]


def test_if_builder_allows_zero_results_with_both_arms() -> None:
    module = builder.ModuleBuilder()
    entry = (
        module.function(name="f", type=ir.FunctionType((), ())).region().first_block()
    )
    conditional = entry.if_(entry.bool(True))
    conditional.then_region.block().yield_()
    conditional.else_region.block().yield_()
    entry.return_()
    module.verify()
    assert conditional.raw.results == ()


def test_builder_branch_errors_are_reported_by_module_verification() -> None:
    module = builder.ModuleBuilder()
    region = module.function(name="f", type=ir.FunctionType((), ())).region()
    entry = region.first_block()
    target = region.block((ir.ScalarType.I32,))
    entry.branch(target.raw)
    target.return_()
    with pytest.raises(TypeError, match="branch argument types"):
        module.verify()
