"""Properties of transpose simplification on generated tensor graphs."""

import itertools
import pickle

import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from niro import ir, optimizations, verify

from .strategies import mixed_modules, with_control_flow


@st.composite
def modules(draw: DrawFn) -> ir.VerifiedModule:
    """Build nonempty tensor DAGs and branches that consume captured results."""
    shape = tuple(draw(st.lists(st.integers(1, 3), max_size=4)))
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
    returned = (values[-1], *draw(st.lists(st.sampled_from(values), max_size=3)))
    condition = ir.Value(ir.ValueId(len(values)), ir.ScalarType.BOOL)
    next_id = len(values) + 1
    branches = []
    for _ in range(2):
        branch_ops: list[ir.Op] = []
        branch_results = []
        for operand in returned:
            permutation = tuple(draw(st.permutations(tuple(range(len(shape))))))
            inverse = tuple(permutation.index(axis) for axis in range(len(shape)))
            transposed_type = ir.infer.transpose_result_type(operand.type, permutation)
            transposed = ir.Value(ir.ValueId(next_id), transposed_type)
            restored = ir.Value(ir.ValueId(next_id + 1), operand.type)
            next_id += 2
            branch_ops.extend(
                [
                    ir.Transpose(transposed, operand, permutation),
                    ir.Transpose(restored, transposed, inverse),
                ]
            )
            if draw(st.booleans()):
                total = ir.Value(ir.ValueId(next_id), operand.type)
                next_id += 1
                branch_ops.append(ir.Add(total, restored, restored))
                restored = total
            branch_results.append(restored)
        branch_ops.append(ir.Yield(tuple(branch_results)))
        branches.append(ir.Region([ir.Block(operations=branch_ops)]))
    results = tuple(
        ir.Value(ir.ValueId(next_id + i), v.type) for i, v in enumerate(returned)
    )
    operations.extend([ir.If(results, condition, *branches), ir.Return(results)])
    function = ir.Function(
        "main",
        ir.FunctionType(
            (argument.type, condition.type), tuple(v.type for v in returned)
        ),
        ir.Region([ir.Block((argument, condition), operations)]),
    )
    return verify.module(ir.Module(functions=[with_control_flow(draw, function)]))


def evaluate(
    function: ir.Function, condition: bool
) -> tuple[dict[tuple[int, ...], int], ...]:
    """Move uniquely numbered tensor elements to evaluate the returned tensors."""
    block = function.first_block
    assert block is not None
    argument = block.arguments[0]
    assert isinstance(argument.type, ir.TensorType)
    assert argument.type.shape is not None
    axes = []
    for dimension in argument.type.shape:
        assert dimension is not None
        axes.append(range(dimension))
    tensors = {
        argument.id: {
            index: value for value, index in enumerate(itertools.product(*axes), 1)
        }
    }

    def run(
        block: ir.Block,
        scope: dict[ir.ValueId, dict[tuple[int, ...], int]],
        flags: dict[ir.ValueId, bool],
    ) -> tuple[dict[tuple[int, ...], int], ...]:
        for _ in range(10000):
            for op in block.operations:
                if isinstance(op, (ir.Branch, ir.CondBranch)):
                    if isinstance(op, ir.Branch):
                        target, arguments = op.target, op.arguments
                    elif flags[op.condition.id]:
                        target, arguments = op.true_target, op.true_arguments
                    else:
                        target, arguments = op.false_target, op.false_arguments
                    # The generator adds boolean loop arguments only.
                    passed = tuple(flags[v.id] for v in arguments)
                    flags.update(
                        (v.id, flag)
                        for v, flag in zip(target.arguments, passed, strict=True)
                    )
                    block = target
                    break
                if isinstance(op, ir.Const):
                    assert isinstance(op.literal, bool)
                    flags[op.result.id] = op.literal
                    continue
                if isinstance(op, (ir.Return, ir.Yield)):
                    return tuple(scope[value.id] for value in op.operands)
                if isinstance(op, ir.If):
                    region = (
                        op.then_region if flags[op.condition.id] else op.else_region
                    )
                    results = run(region.blocks[0], dict(scope), dict(flags))
                    scope.update(
                        (v.id, r) for v, r in zip(op.results, results, strict=True)
                    )
                    continue
                if isinstance(op, ir.Add):
                    scope[op.result.id] = {
                        index: value + scope[op.rhs.id][index]
                        for index, value in scope[op.lhs.id].items()
                    }
                    continue
                assert isinstance(op, ir.Transpose)
                scope[op.result.id] = {
                    tuple(index[axis] for axis in op.permutation): value
                    for index, value in scope[op.operand.id].items()
                }
            else:
                raise AssertionError("missing terminator")
        raise AssertionError("execution step limit exceeded")

    return run(block, tensors, {block.arguments[1].id: condition})


@hypothesis.given(modules())
def test_simplify_transposes_preserves_semantics(module: ir.VerifiedModule) -> None:
    snapshot = pickle.dumps(module)
    updated = optimizations.simplify_transposes(module)
    verify.module(updated)
    hypothesis.event(f"rewritten={updated is not module}")
    for condition in (False, True):
        assert evaluate(updated.functions[0], condition) == evaluate(
            module.functions[0], condition
        )
    assert updated.functions[0].type == module.functions[0].type
    assert pickle.dumps(module) == snapshot
    assert optimizations.simplify_transposes(updated) is updated
    body = updated.functions[0].body
    assert body is not None
    transposes = [op for op in ir.iter_ops(body) if isinstance(op, ir.Transpose)]
    produced = {op.result.id for op in transposes}
    for op in transposes:
        assert op.permutation != tuple(range(len(op.permutation)))
        assert op.operand.id not in produced


@hypothesis.given(mixed_modules())
def test_simplify_transposes_preserves_validity(module: ir.VerifiedModule) -> None:
    snapshot = pickle.dumps(module)
    updated = optimizations.simplify_transposes(module)
    verify.module(updated)
    hypothesis.event(f"rewritten={updated is not module}")
    assert pickle.dumps(module) == snapshot
    assert updated.globals == module.globals
    assert updated.attributes == module.attributes
    assert [(f.name, f.type, f.attributes) for f in updated.functions] == [
        (f.name, f.type, f.attributes) for f in module.functions
    ]
    assert optimizations.simplify_transposes(updated) is updated
