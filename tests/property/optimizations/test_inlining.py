"""Properties of inlining on generated acyclic scalar programs."""

import pickle
from collections.abc import Sequence

import hypothesis
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from niro import ir, optimizations, verify

from .strategies import mixed_modules


@st.composite
def modules(draw: DrawFn) -> ir.VerifiedModule:
    """Build small call DAGs, including nested calls, captures and ordered effects."""
    functions: list[ir.Function] = []
    for index in range(draw(st.integers(2, 4))):
        functions.append(make_function(draw, f"f{index}", functions))
    return verify.module(ir.Module(functions=list(draw(st.permutations(functions)))))


@st.composite
def inlining_cases(
    draw: DrawFn, strategy: SearchStrategy[ir.VerifiedModule]
) -> tuple[ir.VerifiedModule, frozenset[ir.SymbolName] | None]:
    """Pair a verified module with all targets or a subset of its function symbols."""
    module = draw(strategy)
    names = [function.name for function in module.functions]
    callees = draw(st.one_of(st.none(), st.frozensets(st.sampled_from(names))))
    return module, callees


def make_function(
    draw: DrawFn, name: str, callees: Sequence[ir.Function]
) -> ir.Function:
    """Generate a function whose calls target already generated definitions."""
    next_id = draw(st.integers(0, 10))
    stride = draw(st.integers(1, 3))

    def fresh(type_: ir.Type = ir.ScalarType.I32) -> ir.Value:
        nonlocal next_id
        result = ir.Value(ir.ValueId(next_id), type_)
        next_id += stride
        return result

    x, y, condition = fresh(), fresh(), fresh(ir.ScalarType.BOOL)

    def body(values: list[ir.Value], nested: bool = False) -> list[ir.Op]:
        operations: list[ir.Op] = []
        choices = ["add", "effect"]
        if callees:
            choices.append("call")
        if not nested:
            choices.append("if")
        for _ in range(draw(st.integers(0, 3))):
            kind = draw(st.sampled_from(choices))
            lhs = draw(st.sampled_from(values))
            rhs = draw(st.sampled_from(values))
            if kind == "effect":
                operations.append(ir.UnknownOp(f"effect_{name}", (lhs, rhs), ()))
            elif kind == "add":
                result = fresh()
                operations.append(ir.Add(result, lhs, rhs))
                values.append(result)
            elif kind == "call":
                callee = draw(st.sampled_from(callees))
                results = tuple(fresh(type_) for type_ in callee.type.outputs)
                operations.append(ir.Call(callee.name, (lhs, rhs, condition), results))
                values.extend(results)
            else:
                branches = []
                for _ in range(2):
                    captured = list(values)
                    branch_ops = body(captured, nested=True)
                    branch_ops.append(ir.Yield((draw(st.sampled_from(captured)),)))
                    branches.append(ir.Region([ir.Block(operations=branch_ops)]))
                result = fresh()
                operations.append(ir.If((result,), condition, *branches))
                values.append(result)
        return operations

    values = [x, y]
    operations = body(values)
    if callees:
        callee = draw(st.sampled_from(callees))
        results = tuple(fresh(type_) for type_ in callee.type.outputs)
        operations.append(ir.Call(callee.name, (values[-1], x, condition), results))
        values.extend(results)
    # Make computed values observable even in functions with no return values.
    operations.append(ir.UnknownOp(f"observe_{name}", tuple(values), ()))
    returned = tuple(draw(st.lists(st.sampled_from(values), max_size=2)))
    operations.append(ir.Return(returned))
    return ir.Function(
        name,
        ir.FunctionType(
            (x.type, y.type, condition.type), tuple(v.type for v in returned)
        ),
        ir.Region([ir.Block((x, y, condition), operations)]),
    )


def evaluate(
    module: ir.Module, name: str, arguments: tuple[int | bool, ...]
) -> tuple[tuple[int | bool, ...], list[tuple[str, tuple[int | bool, ...]]]]:
    """Interpret scalar values and effect traces independently of the rewrite API."""
    functions = {function.name: function for function in module.functions}
    effects: list[tuple[str, tuple[int | bool, ...]]] = []

    def call(name: str, args: tuple[int | bool, ...]) -> tuple[int | bool, ...]:
        block = functions[name].first_block
        assert block is not None
        return run(
            block, {v.id: arg for v, arg in zip(block.arguments, args, strict=True)}
        )

    def run(
        block: ir.Block, scope: dict[ir.ValueId, int | bool]
    ) -> tuple[int | bool, ...]:
        for op in block.operations:
            match op:
                case ir.Add():
                    # Scalar i32 addition wraps, including after repeated inlining.
                    total = int(scope[op.lhs.id]) + int(scope[op.rhs.id])
                    scope[op.result.id] = (total + 2**31) % 2**32 - 2**31
                case ir.Call():
                    results = call(op.callee, tuple(scope[v.id] for v in op.arguments))
                    scope.update(
                        (v.id, r) for v, r in zip(op.results, results, strict=True)
                    )
                case ir.If():
                    region = (
                        op.then_region if scope[op.condition.id] else op.else_region
                    )
                    results = run(region.blocks[0], dict(scope))
                    scope.update(
                        (v.id, r) for v, r in zip(op.results, results, strict=True)
                    )
                case ir.UnknownOp():
                    assert not op.results
                    effects.append((op.name, tuple(scope[v.id] for v in op.operands)))
                case ir.Return() | ir.Yield():
                    return tuple(scope[v.id] for v in op.operands)
                case _:
                    raise AssertionError(f"unexpected test operation: {op}")
        raise AssertionError("missing terminator")

    return call(name, arguments), effects


@hypothesis.given(
    inlining_cases(modules()),
    st.one_of(st.none(), st.integers(0, 20)),
    st.integers(-(2**31), 2**31 - 1),
    st.integers(-(2**31), 2**31 - 1),
)
def test_inline_functions_preserves_semantics(
    case: tuple[ir.VerifiedModule, frozenset[ir.SymbolName] | None],
    limit: int | None,
    x: int,
    y: int,
) -> None:
    module, callees = case
    operations = [
        op
        for function in module.functions
        if function.body is not None
        for op in ir.iter_ops(function.body)
    ]
    hypothesis.event(f"has_calls={any(isinstance(op, ir.Call) for op in operations)}")
    hypothesis.event(f"has_branches={any(isinstance(op, ir.If) for op in operations)}")
    snapshot = pickle.dumps(module)
    updated = optimizations.inline_functions(
        module, callees=callees, max_callee_ops=limit
    )
    verify.module(updated)
    hypothesis.event(f"rewritten={updated is not module}")
    assert [(f.name, f.type) for f in updated.functions] == [
        (f.name, f.type) for f in module.functions
    ]
    for function in module.functions:
        for condition in (False, True):
            args = (x, y, condition)
            assert evaluate(updated, function.name, args) == evaluate(
                module, function.name, args
            )
    assert pickle.dumps(module) == snapshot
    assert (
        optimizations.inline_functions(updated, callees=callees, max_callee_ops=limit)
        is updated
    )
    if limit == 0 or callees == frozenset():
        assert updated is module
    if limit is None:
        assert not any(
            isinstance(op, ir.Call) and (callees is None or op.callee in callees)
            for function in updated.functions
            if function.body is not None
            for op in ir.iter_ops(function.body)
        )


@hypothesis.given(
    inlining_cases(mixed_modules()), st.one_of(st.none(), st.integers(0, 20))
)
def test_inline_functions_preserves_validity(
    case: tuple[ir.VerifiedModule, frozenset[ir.SymbolName] | None], limit: int | None
) -> None:
    module, callees = case
    snapshot = pickle.dumps(module)
    updated = optimizations.inline_functions(
        module, callees=callees, max_callee_ops=limit
    )
    verify.module(updated)
    hypothesis.event(f"rewritten={updated is not module}")
    assert pickle.dumps(module) == snapshot
    assert updated.globals == module.globals
    assert updated.attributes == module.attributes
    assert [(f.name, f.type, f.attributes) for f in updated.functions] == [
        (f.name, f.type, f.attributes) for f in module.functions
    ]
    assert (
        optimizations.inline_functions(updated, callees=callees, max_callee_ops=limit)
        is updated
    )
