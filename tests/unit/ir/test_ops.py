"""Check opaque operation structure and functional tensor element reads."""

import dataclasses

import pytest

from niro import builder, ir, rewrite, verify


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_tensor_extract_preserves_input(shape: tuple[int, ...]) -> None:
    tensor_type = ir.TensorType(ir.ScalarType.F32, shape)
    module = builder.ModuleBuilder()
    function = module.function(
        name="extract",
        type=ir.FunctionType((tensor_type,), (ir.ScalarType.F32, tensor_type)),
    )
    block = function.region().first_block()
    (operand,) = block.raw.arguments
    indices = tuple(block.const(0, ir.ScalarType.I64) for _ in shape)
    result = block.tensor_extract(operand, indices)
    block.return_(result, operand)
    module.verify()
    operation = block.raw.operations[-2]
    assert isinstance(operation, ir.TensorExtract)
    assert ir.get_operands(operation) == (operand, *indices)
    assert ir.get_results(operation) == (result,)
    assert function.raw.body is not None
    copied, mapping = rewrite.clone_region(
        function.raw.body, ir.ValueSupply(next_id=20)
    )
    assert copied is not function.raw.body
    assert result.id in mapping


@pytest.mark.parametrize(
    ("operand_type", "index_types", "error"),
    [
        (ir.ScalarType.F32, (), TypeError),
        (ir.TensorType(ir.ScalarType.F32, (2,)), (), ValueError),
        (ir.TensorType(ir.ScalarType.F32, (2,)), (ir.ScalarType.F32,), TypeError),
    ],
)
def test_tensor_extract_rejects_invalid_operands(
    operand_type: ir.Type, index_types: tuple[ir.Type, ...], error: type[Exception]
) -> None:
    module = builder.ModuleBuilder()
    function = module.function(
        name="extract", type=ir.FunctionType((operand_type, *index_types), ())
    )
    block = function.region().first_block()
    operand, *indices = block.raw.arguments
    with pytest.raises(error):
        block.tensor_extract(operand, indices)


@pytest.mark.parametrize(
    "scenario",
    ["capture", "foreign-target", "early-terminator", "shared-region", "self-capture"],
)
def test_unknown_regions_and_successors(scenario: str) -> None:
    module = builder.ModuleBuilder()
    function = module.function(
        name="opaque", type=ir.FunctionType((ir.ScalarType.I32,), ())
    )
    entry = function.region().first_block()
    (captured,) = entry.raw.arguments
    exit_block = ir.Block(operations=[ir.Yield((captured,))])
    jump = ir.UnknownOp("example.jump", (captured,), (), successors=(exit_block,))
    nested_entry = ir.Block(operations=[jump])
    nested = ir.Region([nested_entry, exit_block])
    results = entry.unknown_op(
        "example.region", result_types=(ir.ScalarType.I32,), regions=(nested,)
    )
    entry.return_()
    if scenario == "foreign-target":
        nested.blocks.remove(exit_block)
    elif scenario == "early-terminator":
        nested_entry.operations.append(ir.Yield())
    elif scenario == "shared-region":
        op = entry.raw.operations[0]
        assert isinstance(op, ir.UnknownOp)
        entry.raw.operations[0] = dataclasses.replace(op, regions=(nested, nested))
    elif scenario == "self-capture":
        exit_block.operations = [ir.Yield(results)]
    if scenario != "capture":
        with pytest.raises(ValueError):
            module.verify()
        return
    module.verify()
    assert ir.get_successors(jump) == (exit_block,)
    assert ir.get_regions(entry.raw.operations[0]) == (nested,)
    assert function.raw.body is not None
    assert len(list(ir.iter_blocks(function.raw.body))) == 3
    copied, _ = rewrite.clone_region(function.raw.body, ir.ValueSupply(next_id=20))
    copied_op = copied.blocks[0].operations[0]
    assert isinstance(copied_op, ir.UnknownOp)
    copied_nested = copied_op.regions[0]
    copied_jump = copied_nested.blocks[0].operations[0]
    assert ir.get_successors(copied_jump) == (copied_nested.blocks[1],)
    assert copied_nested.blocks[1] is not exit_block
    transformed = dataclasses.replace(function.raw, body=copied)
    verify.module(ir.Module(functions=[transformed]))


def test_unknown_successor_participates_in_function_cfg() -> None:
    exit_block = ir.Block(operations=[ir.Return()])
    jump = ir.UnknownOp("example.jump", (), (), successors=(exit_block,))
    function = ir.Function(
        "cfg",
        ir.FunctionType((), ()),
        body=ir.Region([ir.Block(operations=[jump]), exit_block]),
    )
    verify.module(ir.Module(functions=[function]))
