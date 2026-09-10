"""Properties of transpose simplification on generated tensor graphs."""

import copy
import itertools

import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from niro import ir, optimizations, verify


@st.composite
def modules(draw: DrawFn) -> ir.VerifiedModule:
    """Build transpose DAGs with shared operands and arbitrary returned values."""
    shape = tuple(draw(st.lists(st.integers(0, 3), max_size=4)))
    argument = ir.Value(ir.ValueId(0), ir.TensorType(ir.ScalarType.I32, shape))
    values = [argument]
    operations: list[ir.Op] = []
    for index in range(draw(st.integers(0, 12))):
        operand = draw(st.sampled_from(values))
        permutation = tuple(draw(st.permutations(tuple(range(len(shape))))))
        assert isinstance(operand.type, ir.TensorType)
        assert operand.type.shape is not None
        result = ir.Value(
            ir.ValueId(index + 1),
            ir.TensorType(
                ir.ScalarType.I32,
                tuple(operand.type.shape[axis] for axis in permutation),
            ),
        )
        operations.append(ir.Transpose(result, operand, permutation))
        values.append(result)
    returned = tuple(draw(st.lists(st.sampled_from(values), min_size=1, max_size=4)))
    operations.append(ir.Return(returned))
    function = ir.Function(
        "main",
        ir.FunctionType((argument.type,), tuple(v.type for v in returned)),
        ir.Region([ir.Block((argument,), operations)]),
    )
    return verify.module(ir.Module(functions=[function]))


def evaluate(function: ir.Function) -> tuple[dict[tuple[int, ...], int], ...]:
    """Move uniquely numbered tensor elements to evaluate the returned tensors."""
    block = function.first_block
    assert block is not None
    (argument,) = block.arguments
    assert isinstance(argument.type, ir.TensorType)
    assert argument.type.shape is not None
    axes = []
    for dimension in argument.type.shape:
        assert dimension is not None
        axes.append(range(dimension))
    tensors = {
        argument.id: {
            index: value for value, index in enumerate(itertools.product(*axes))
        }
    }
    for op in block.operations:
        if isinstance(op, ir.Return):
            return tuple(tensors[value.id] for value in op.operands)
        assert isinstance(op, ir.Transpose)
        tensors[op.result.id] = {
            tuple(index[axis] for axis in op.permutation): value
            for index, value in tensors[op.operand.id].items()
        }
    raise AssertionError("missing return")


@hypothesis.given(modules())
def test_simplify_transposes_preserves_semantics(module: ir.VerifiedModule) -> None:
    snapshot = copy.deepcopy(module)
    updated = optimizations.simplify_transposes(module)
    verify.module(updated)
    hypothesis.event(f"rewritten={updated is not module}")
    assert evaluate(updated.functions[0]) == evaluate(module.functions[0])
    assert updated.functions[0].type == module.functions[0].type
    assert module == snapshot
    assert optimizations.simplify_transposes(updated) is updated
    body = updated.functions[0].body
    assert body is not None
    transposes = [op for op in ir.iter_ops(body) if isinstance(op, ir.Transpose)]
    produced = {op.result.id for op in transposes}
    for op in transposes:
        assert op.permutation != tuple(range(len(op.permutation)))
        assert op.operand.id not in produced
