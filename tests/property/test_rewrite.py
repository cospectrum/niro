"""Semantic and structural properties of functional rewrites over bounded CFGs."""

import dataclasses
import pickle

import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from niro import ir, rewrite, verify


@st.composite
def functions(draw: DrawFn) -> tuple[ir.Function, ir.Const, ir.Const]:
    """Generate equivalent constants, nested captures, and a terminating CFG loop."""
    supply = ir.ValueSupply(draw(st.integers(0, 20)))
    flag, x = supply.fresh(ir.ScalarType.BOOL), supply.fresh(ir.ScalarType.I32)
    literal = draw(st.integers(-20, 20))
    first = ir.Const(supply.fresh(x.type), literal)
    duplicate = ir.Const(supply.fresh(x.type), literal)
    entry = ir.Block((flag, x), [first, duplicate])
    current = x
    for use_duplicate in draw(st.lists(st.booleans(), max_size=6)):
        result = supply.fresh(x.type)
        entry.operations.append(
            ir.Add(result, current, duplicate.result if use_duplicate else first.result)
        )
        current = result
    selected = supply.fresh(x.type)
    entry.operations.append(
        ir.If(
            (selected,),
            flag,
            ir.Region([ir.Block(operations=[ir.Yield((first.result,))])]),
            ir.Region([ir.Block(operations=[ir.Yield((duplicate.result,))])]),
        )
    )
    repeat, carried = supply.fresh(flag.type), supply.fresh(x.type)
    stop, total, returned = (
        supply.fresh(flag.type),
        supply.fresh(x.type),
        supply.fresh(x.type),
    )
    loop = ir.Block((repeat, carried))
    exit_ = ir.Block((returned,), [ir.Return((returned,))])
    loop.operations = [
        ir.Const(stop, False),
        ir.Add(total, carried, selected),
        ir.CondBranch(repeat, loop, exit_, (stop, total), (total,)),
    ]
    entry.operations.append(ir.Branch(loop, (flag, current)))
    body = ir.Region([entry, *draw(st.permutations([loop, exit_]))])
    function = ir.Function("f", ir.FunctionType((flag.type, x.type), (x.type,)), body)
    verify.module(ir.Module(functions=[function]))
    return function, first, duplicate


def evaluate(function: ir.Function, flag: bool, x: int) -> tuple[int | bool, ...]:
    """Execute the generated scalar subset independently of rewrite helpers."""
    assert function.body is not None
    entry = function.body.blocks[0]

    def run(
        block: ir.Block, scope: dict[ir.ValueId, int | bool]
    ) -> tuple[int | bool, ...]:
        for _ in range(64):
            for op in block.operations:
                match op:
                    case ir.Const():
                        assert isinstance(op.literal, (int, bool))
                        scope[op.result.id] = op.literal
                    case ir.Add():
                        scope[op.result.id] = scope[op.lhs.id] + scope[op.rhs.id]
                    case ir.If():
                        arm = (
                            op.then_region if scope[op.condition.id] else op.else_region
                        )
                        results = run(arm.blocks[0], dict(scope))
                        scope.update(
                            (v.id, result)
                            for v, result in zip(op.results, results, strict=True)
                        )
                    case ir.Branch() | ir.CondBranch():
                        if isinstance(op, ir.Branch):
                            target, arguments = op.target, op.arguments
                        elif scope[op.condition.id]:
                            target, arguments = op.true_target, op.true_arguments
                        else:
                            target, arguments = op.false_target, op.false_arguments
                        values = tuple(scope[v.id] for v in arguments)
                        scope.update(
                            (v.id, result)
                            for v, result in zip(target.arguments, values, strict=True)
                        )
                        block = target
                        break
                    case ir.Yield() | ir.Return():
                        return tuple(scope[v.id] for v in op.operands)
                    case _:
                        raise AssertionError(f"unexpected operation: {op}")
            else:
                raise AssertionError("missing terminator")
        raise AssertionError("execution step limit exceeded")

    return run(entry, {entry.arguments[0].id: flag, entry.arguments[1].id: x})


def check_equivalent(original: ir.Function, updated: ir.Function, x: int) -> None:
    """Check verified output and equivalent results on both conditional paths."""
    verify.module(ir.Module(functions=[updated]))
    for flag in (False, True):
        assert evaluate(updated, flag, x) == evaluate(original, flag, x)


@hypothesis.given(functions(), st.integers(-1000, 1000), st.integers(0, 20))
def test_clone_freshens_definitions_and_preserves_cfg_semantics(
    case: tuple[ir.Function, ir.Const, ir.Const],
    x: int,
    gap: int,
) -> None:
    original, _, _ = case
    assert original.body is not None
    snapshot = pickle.dumps(original)
    supply = rewrite.value_supply(original)
    supply.next_id += gap
    start = supply.next_id
    copied, values = rewrite.clone_region(original.body, supply)
    updated = dataclasses.replace(original, body=copied)
    check_equivalent(original, updated, x)
    assert set(values) == {v.id for v in ir.iter_defined_values(original.body)}
    assert {v.id for v in values.values()} == set(range(start, supply.next_id))
    original_blocks = set(ir.iter_blocks(original.body))
    copied_blocks = set(ir.iter_blocks(copied))
    assert original_blocks.isdisjoint(copied_blocks)
    assert all(
        target in copied_blocks
        for op in ir.iter_ops(copied)
        for target in ir.get_successors(op)
    )
    assert pickle.dumps(original) == snapshot


@hypothesis.given(functions(), st.integers(-1000, 1000))
def test_replace_uses_and_erase_preserve_semantics_and_input(
    case: tuple[ir.Function, ir.Const, ir.Const],
    x: int,
) -> None:
    original, first, duplicate = case
    snapshot = pickle.dumps(original)
    redirected = rewrite.replace_uses(original, {duplicate.result.id: first.result})
    check_equivalent(original, redirected, x)
    updated = rewrite.erase_ops(redirected, (duplicate,))
    check_equivalent(original, updated, x)
    assert list(ir.iter_uses(updated, duplicate.result.id)) == []
    assert ir.get_definition(updated, duplicate.result.id) is None
    assert pickle.dumps(original) == snapshot


@hypothesis.given(functions(), st.integers(-1000, 1000), st.integers())
def test_insert_and_erase_unused_constant_preserve_semantics(
    case: tuple[ir.Function, ir.Const, ir.Const],
    x: int,
    location: int,
) -> None:
    original, _, _ = case
    assert original.body is not None
    snapshot = pickle.dumps(original)
    blocks = list(ir.iter_blocks(original.body))
    block = blocks[location % len(blocks)]
    index = location % len(block.operations)
    fresh = rewrite.value_supply(original).fresh(ir.ScalarType.I32)
    inserted = ir.Const(fresh, 7)
    updated = rewrite.insert_ops(
        original, (inserted,), at=rewrite.InsertPoint(block, index)
    )
    check_equivalent(original, updated, x)
    restored = rewrite.erase_ops(updated, (inserted,))
    check_equivalent(original, restored, x)
    assert ir.get_definition(restored, fresh.id) is None
    assert pickle.dumps(original) == snapshot
