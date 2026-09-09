from niro import ir


def test_verify_returns_same_module() -> None:
    module = ir.Module(
        functions=[ir.Function("external", ir.FunctionType((), ()))],
        globals=[ir.Global("value", ir.ScalarType.I32, 1)],
    )

    assert ir.verify(module) is module


def test_verify_forward_references_and_nested_scopes() -> None:
    module = ir.ModuleBuilder()
    function = module.function(
        name="main",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
    )
    block = function.region().first_block()
    (argument,) = block.raw.arguments
    weight = block.get_global("weight", type=ir.ScalarType.I32)
    condition = block.bool(True)
    conditional = block.if_(condition, (ir.ScalarType.I32,))
    then_block = conditional.then_region.block()
    nested = then_block.if_(condition, (ir.ScalarType.I32,))
    nested.then_region.block().yield_(argument)
    nested.else_region.block().yield_(weight)
    then_block.yield_(*nested.raw.results)
    else_block = conditional.else_region.block()
    (called,) = else_block.call("later", (argument,), result_types=(ir.ScalarType.I32,))
    else_block.yield_(called)
    block.return_(*conditional.raw.results)
    module.global_("weight", ir.ScalarType.I32, 42)
    later = (
        module.function(
            name="later",
            type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
        )
        .region()
        .first_block()
    )
    later.return_(*later.raw.arguments)

    assert ir.verify(module.raw) is module.raw


def test_verify_resultless_if_without_else_and_unknown_operation() -> None:
    module = ir.ModuleBuilder()
    block = module.function(name="main", type=ir.FunctionType((), ())).region().block()
    (value,) = block.unknown_op("example.source", result_types=(ir.ScalarType.I32,))
    branch = block.if_(block.bool(True)).then_region.block()
    branch.unknown_op("example.consume", operands=(value,))
    branch.yield_()
    block.return_()

    assert ir.verify(module.raw) is module.raw
