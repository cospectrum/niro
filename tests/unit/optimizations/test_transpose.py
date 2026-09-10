import dataclasses
import itertools
import pickle
from collections.abc import Sequence

import pytest

from niro import ir, optimizations, verify


def chain(
    shape: ir.Shape | None,
    permutations: Sequence[tuple[int, ...]],
    *,
    name: str = "f",
) -> ir.Function:
    argument = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.F32, shape))
    operand = argument
    operations: list[ir.Op] = []
    for index, permutation in enumerate(permutations, start=1):
        result = ir.Value(
            ir.ValueId(index),
            ir.infer.transpose_result_type(operand.type, permutation),
        )
        operations.append(ir.Transpose(result, operand, permutation))
        operand = result
    operations.append(ir.Return((operand,)))
    return ir.Function(
        name,
        ir.FunctionType((argument.type,), (operand.type,)),
        ir.Region([ir.Block((argument,), operations)]),
    )


def block(function: ir.Function) -> ir.Block:
    assert function.first_block is not None
    return function.first_block


def evaluate_transposes(function: ir.Function) -> dict[tuple[int, ...], int]:
    """Evaluate each transpose by moving individual tensor elements."""
    body = block(function)
    (argument,) = body.arguments
    assert isinstance(argument.type, ir.TensorType)
    assert argument.type.shape is not None
    axes = []
    for size in argument.type.shape:
        assert size is not None
        axes.append(range(size))
    tensors = {
        argument.id: {
            index: value for value, index in enumerate(itertools.product(*axes))
        }
    }
    for op in body.operations:
        if isinstance(op, ir.Return):
            return tensors[op.operands[0].id]
        assert isinstance(op, ir.Transpose)
        tensors[op.result.id] = {
            tuple(index[axis] for axis in op.permutation): value
            for index, value in tensors[op.operand.id].items()
        }
    raise AssertionError("missing return")


@pytest.mark.parametrize("shape", [(), (7,), (0, 3), (None, 3), (None, None, None)])
def test_removes_identity_for_known_rank_including_dynamic_dimensions(
    shape: ir.Shape,
) -> None:
    function = chain(shape, (tuple(range(len(shape))),))
    original = verify.module(ir.Module(functions=[function]))
    snapshot = pickle.dumps(original)
    updated = optimizations.simplify_transposes(original)
    assert block(updated.functions[0]).operations == [
        ir.Return(block(function).arguments)
    ]
    assert pickle.dumps(original) == snapshot
    assert optimizations.simplify_transposes(updated) is updated


@pytest.mark.parametrize("first", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("second", list(itertools.permutations(range(3))))
def test_all_rank_three_permutation_pairs_preserve_tensor_elements(
    first: tuple[int, ...], second: tuple[int, ...]
) -> None:
    # Equal dimensions ensure type verification alone cannot catch a wrong order.
    function = chain((2, 2, 2), (first, second))
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    optimized = updated.functions[0]
    assert evaluate_transposes(optimized) == evaluate_transposes(function)
    assert len(block(optimized).operations) <= 2
    assert optimizations.simplify_transposes(updated) is updated


def test_composes_noncommuting_permutations_and_preserves_result_id() -> None:
    function = chain((2, 3, 5), ((1, 0, 2), (0, 2, 1)))
    original_ops = block(function).operations
    consumer = original_ops[1]
    assert isinstance(consumer, ir.Transpose)
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    assert block(updated.functions[0]).operations == [
        ir.Transpose(consumer.result, block(function).arguments[0], (1, 2, 0)),
        original_ops[-1],
    ]


def test_simplifies_long_chain_in_one_pass_invocation() -> None:
    permutations = ((1, 0, 2), (0, 2, 1), (2, 1, 0)) * 3
    function = chain((2, 3, 4), permutations)
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    assert len(block(updated.functions[0]).operations) <= 2
    assert evaluate_transposes(updated.functions[0]) == evaluate_transposes(function)
    assert optimizations.simplify_transposes(updated) is updated


@pytest.mark.parametrize("second", [(1, 0, 2), (0, 2, 1)])
def test_preserves_producer_with_other_uses(second: tuple[int, ...]) -> None:
    function = chain((2, 3, 4), ((1, 0, 2), second))
    first, last, _ = block(function).operations
    assert isinstance(first, ir.Transpose)
    assert isinstance(last, ir.Transpose)
    returns = (first.result, last.result)
    function = dataclasses.replace(
        function,
        type=ir.FunctionType(function.type.inputs, tuple(v.type for v in returns)),
        body=ir.Region(
            [ir.Block(block(function).arguments, [first, last, ir.Return(returns)])]
        ),
    )
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    operations = block(updated.functions[0]).operations
    assert operations[0] is first
    if second == (1, 0, 2):
        assert operations == [first, ir.Return((first.result, first.operand))]
    else:
        assert operations == [
            first,
            ir.Transpose(last.result, first.operand, (1, 2, 0)),
            ir.Return(returns),
        ]


@pytest.mark.parametrize("second", [(1, 0, 2), (0, 2, 1)])
def test_nonadjacent_transposes_preserve_intervening_operations(
    second: tuple[int, ...],
) -> None:
    function = chain((2, 3, 4), ((1, 0, 2), second))
    first, last, ret = block(function).operations
    assert isinstance(first, ir.Transpose)
    assert isinstance(last, ir.Transpose)
    effect = ir.UnknownOp("effect", (), ())
    function = dataclasses.replace(
        function,
        body=ir.Region(
            [ir.Block(block(function).arguments, [first, effect, last, ret])]
        ),
    )
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    operations = block(updated.functions[0]).operations
    assert operations[0] is effect
    if second == (1, 0, 2):
        assert operations == [effect, ir.Return((first.operand,))]
    else:
        assert operations == [
            effect,
            ir.Transpose(last.result, first.operand, (1, 2, 0)),
            ret,
        ]


def test_composes_captures_in_both_if_regions_and_removes_unused_producer() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    swapped = ir.TensorType(ir.ScalarType.F32, (3, 2))
    x = ir.Value(ir.ValueId(0), tensor)
    condition = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    transposed = ir.Value(ir.ValueId(2), swapped)
    then_value, else_value, result = (
        ir.Value(ir.ValueId(i), tensor) for i in range(3, 6)
    )
    producer = ir.Transpose(transposed, x, (1, 0))
    branch = ir.If(
        (result,),
        condition,
        ir.Region(
            [
                ir.Block(
                    operations=[
                        ir.Transpose(then_value, transposed, (1, 0)),
                        ir.Yield((then_value,)),
                    ]
                )
            ]
        ),
        ir.Region(
            [
                ir.Block(
                    operations=[
                        ir.Transpose(else_value, transposed, (1, 0)),
                        ir.Yield((else_value,)),
                    ]
                )
            ]
        ),
    )
    function = ir.Function(
        "branches",
        ir.FunctionType((tensor, condition.type), (tensor,)),
        ir.Region([ir.Block((x, condition), [producer, branch, ir.Return((result,))])]),
    )
    original = verify.module(ir.Module(functions=[function]))
    snapshot = pickle.dumps(original)
    updated = optimizations.simplify_transposes(original)
    operations = block(updated.functions[0]).operations
    assert len(operations) == 2
    updated_branch = operations[0]
    assert isinstance(updated_branch, ir.If)
    assert updated_branch.then_region.blocks[0].operations == [ir.Yield((x,))]
    assert updated_branch.else_region.blocks[0].operations == [ir.Yield((x,))]
    assert pickle.dumps(original) == snapshot


def test_identity_redirects_repeated_operands_and_nested_captures() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    x, identity, total, result = (ir.Value(ir.ValueId(i), tensor) for i in range(4))
    condition = ir.Value(ir.ValueId(4), ir.ScalarType.BOOL)
    branch = ir.If(
        (result,),
        condition,
        ir.Region([ir.Block(operations=[ir.Yield((identity,))])]),
        ir.Region([ir.Block(operations=[ir.Yield((total,))])]),
    )
    function = ir.Function(
        "captures",
        ir.FunctionType((tensor, condition.type), (tensor,)),
        ir.Region(
            [
                ir.Block(
                    (x, condition),
                    [
                        ir.Transpose(identity, x, (0, 1)),
                        ir.Add(total, identity, identity),
                        branch,
                        ir.Return((result,)),
                    ],
                )
            ]
        ),
    )
    original = verify.module(ir.Module(functions=[function]))
    updated = optimizations.simplify_transposes(original)
    operations = block(updated.functions[0]).operations
    assert operations[0] == ir.Add(total, x, x)
    updated_branch = operations[1]
    assert isinstance(updated_branch, ir.If)
    assert updated_branch.then_region.blocks[0].operations == [ir.Yield((x,))]


@pytest.mark.parametrize(
    "permutations",
    [((0, 1), (0, 1)), ((1, 0), (1, 0)), ((0, 0), (1, 0)), ((), ())],
)
def test_unknown_rank_is_unchanged(
    permutations: tuple[tuple[int, ...], ...],
) -> None:
    function = chain(None, permutations)
    original = verify.module(ir.Module(functions=[function]))
    assert optimizations.simplify_transposes(original) is original


def test_single_nonidentity_transpose_of_an_operation_result_is_unchanged() -> None:
    function = chain((3, 3), ((1, 0),))
    x = block(function).arguments[0]
    produced = ir.Value(ir.ValueId(2), x.type)
    original_transpose, ret = block(function).operations
    assert isinstance(original_transpose, ir.Transpose)
    function = dataclasses.replace(
        function,
        body=ir.Region(
            [
                ir.Block(
                    (x,),
                    [
                        ir.UnknownOp("tensor_source", (x,), (produced,)),
                        dataclasses.replace(original_transpose, operand=produced),
                        ret,
                    ],
                )
            ]
        ),
    )
    original = verify.module(ir.Module(functions=[function]))
    assert optimizations.simplify_transposes(original) is original


def test_module_preserves_metadata_globals_and_unchanged_functions() -> None:
    first = dataclasses.replace(
        chain((2, 3), ((1, 0), (1, 0)), name="first"),
        input_names=("input",),
        output_names=("output",),
        attributes={"tag": "first"},
    )
    second = chain((None, 3), ((0, 1),), name="second")
    unchanged = chain((2, 3), (), name="unchanged")
    external = ir.Function("external", first.type)
    global_ = ir.Global("empty", ir.TensorType(ir.ScalarType.F32, (0,)), b"")
    original = verify.module(
        ir.Module(
            functions=[first, external, second, unchanged],
            globals=[global_],
            attributes={"tag": "module"},
        )
    )
    snapshot = pickle.dumps(original)
    updated = optimizations.simplify_transposes(original)
    for index in (0, 2):
        assert block(updated.functions[index]).operations == [
            ir.Return(block(original.functions[index]).arguments)
        ]
    assert updated.functions[1] is external
    assert updated.functions[3] is unchanged
    assert updated.functions[0].input_names == first.input_names
    assert updated.functions[0].output_names == first.output_names
    assert updated.functions[0].attributes == first.attributes
    assert updated.attributes == original.attributes
    assert updated.globals[0] is global_
    assert pickle.dumps(original) == snapshot


def test_empty_module_is_unchanged() -> None:
    original = verify.module(ir.Module())
    assert optimizations.simplify_transposes(original) is original


def test_simplifies_transposes_across_blocks_and_in_branch_arguments() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    swapped = ir.TensorType(ir.ScalarType.F32, (3, 2))
    x, first, second, returned = (
        ir.Value(ir.ValueId(i), type_)
        for i, type_ in enumerate((tensor, swapped, tensor, tensor))
    )
    exit_ = ir.Block((returned,), [ir.Return((returned,))])
    middle = ir.Block(
        operations=[ir.Transpose(second, first, (1, 0)), ir.Branch(exit_, (second,))]
    )
    entry = ir.Block((x,), [ir.Transpose(first, x, (1, 0)), ir.Branch(middle)])
    fn = ir.Function(
        "f", ir.FunctionType((tensor,), (tensor,)), ir.Region([entry, middle, exit_])
    )
    original = verify.module(ir.Module(functions=[fn]))
    snapshot = pickle.dumps(original)
    updated = optimizations.simplify_transposes(original)
    body = updated.functions[0].body
    assert body is not None
    assert not any(isinstance(op, ir.Transpose) for op in ir.iter_ops(body))
    assert body.blocks[1].operations == [ir.Branch(body.blocks[2], (x,))]
    assert pickle.dumps(original) == snapshot
    assert optimizations.simplify_transposes(updated) is updated
