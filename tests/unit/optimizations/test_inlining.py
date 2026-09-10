import dataclasses
import itertools
import pickle
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

import pytest

from niro import ir, optimizations, verify


def value(index: int, type: ir.Type = ir.ScalarType.I32) -> ir.Value:
    return ir.Value(ir.ValueId(index), type)


def region(*operations: ir.Op) -> ir.Region:
    return ir.Region([ir.Block(operations=list(operations))])


def function(
    name: str, arguments: tuple[ir.Value, ...], operations: Sequence[ir.Op]
) -> ir.Function:
    ret = operations[-1]
    assert isinstance(ret, ir.Return)
    return ir.Function(
        name,
        ir.FunctionType(
            tuple(v.type for v in arguments), tuple(v.type for v in ret.operands)
        ),
        ir.Region([ir.Block(arguments, list(operations))]),
    )


def block(function: ir.Function) -> ir.Block:
    assert function.first_block is not None
    return function.first_block


def named(module: ir.Module, name: str) -> ir.Function:
    return next(function for function in module.functions if function.name == name)


def calls(function: ir.Function) -> list[ir.Call]:
    assert function.body is not None
    return [op for op in ir.iter_ops(function.body) if isinstance(op, ir.Call)]


def evaluate(
    module: ir.Module, name: str, arguments: tuple[int | bool, ...]
) -> tuple[tuple[int | bool, ...], list[tuple[str, tuple[int | bool, ...]]]]:
    """Execute scalar calls and branches, recording the effects in program order."""
    effects: list[tuple[str, tuple[int | bool, ...]]] = []

    def run(
        body: ir.Block, scope: dict[ir.ValueId, int | bool]
    ) -> tuple[int | bool, ...]:
        for op in body.operations:
            operands = tuple(scope[v.id] for v in ir.get_operands(op))
            match op:
                case ir.Const():
                    assert isinstance(op.literal, (int, bool))
                    results = (op.literal,)
                case ir.Add():
                    results = (operands[0] + operands[1],)
                case ir.Call():
                    callee = block(named(module, op.callee))
                    results = run(
                        callee,
                        {
                            arg.id: operand
                            for arg, operand in zip(
                                callee.arguments, operands, strict=True
                            )
                        },
                    )
                case ir.If():
                    selected = op.then_region if operands[0] else op.else_region
                    results = run(selected.blocks[0], dict(scope))
                case ir.GetGlobal():
                    global_ = next(g for g in module.globals if g.name == op.name)
                    assert isinstance(global_.initializer, (int, bool))
                    results = (global_.initializer,)
                case ir.UnknownOp():
                    assert not op.results
                    effects.append((op.name, operands))
                    results = ()
                case ir.Return() | ir.Yield():
                    return operands
                case _:
                    raise AssertionError(f"unexpected test operation: {op}")
            scope.update(
                (result.id, operand)
                for result, operand in zip(ir.get_results(op), results, strict=True)
            )
        raise AssertionError("missing terminator")

    entry = block(named(module, name))
    result = run(
        entry,
        {
            arg.id: operand
            for arg, operand in zip(entry.arguments, arguments, strict=True)
        },
    )
    return result, effects


def test_repeated_calls_get_fresh_values_and_preserve_input() -> None:
    arg, doubled = value(0), value(1)
    helper = function(
        "double", (arg,), [ir.Add(doubled, arg, arg), ir.Return((doubled,))]
    )
    x, first, second, unused = value(0), value(1), value(2), value(100)
    caller = function(
        "caller",
        (x,),
        [
            ir.Const(unused, 99),
            ir.Call("double", (x,), (first,)),
            ir.Call("double", (first,), (second,)),
            ir.Return((second,)),
        ],
    )
    original = verify.module(ir.Module(functions=[caller, helper]))
    snapshot = pickle.dumps(original)
    updated = optimizations.inline_functions(original)
    optimized = named(updated, "caller")
    assert calls(optimized) == []
    additions = [op for op in block(optimized).operations if isinstance(op, ir.Add)]
    assert len(additions) == 2
    assert additions[0].result.id > unused.id
    assert additions[1].result.id > additions[0].result.id
    assert additions[1].lhs == additions[0].result
    assert (
        evaluate(updated, "caller", (7,))
        == evaluate(original, "caller", (7,))
        == ((28,), [])
    )
    assert named(updated, "double") is helper
    assert pickle.dumps(original) == snapshot
    assert optimizations.inline_functions(updated) is updated


@pytest.mark.parametrize("result_count", [0, 1, 2])
def test_argument_only_returns_and_zero_result_calls(result_count: int) -> None:
    arg, x = value(0), value(4)
    helper = function("forward", (arg,), [ir.Return((arg,) * result_count)])
    results = tuple(value(5 + i) for i in range(result_count))
    caller = function(
        "caller", (x,), [ir.Call("forward", (x,), results), ir.Return(results)]
    )
    original = verify.module(ir.Module(functions=[helper, caller]))
    updated = optimizations.inline_functions(original, max_callee_ops=1)
    assert block(named(updated, "caller")).operations == [
        ir.Return((x,) * result_count)
    ]


def test_equal_zero_result_calls_preserve_each_effect() -> None:
    helper = function("effectful", (), [ir.UnknownOp("effect", (), ()), ir.Return()])
    first, second = ir.Call("effectful", (), ()), ir.Call("effectful", (), ())
    caller = function("caller", (), [first, second, ir.Return()])
    original = verify.module(ir.Module(functions=[caller, helper]))
    updated = optimizations.inline_functions(original)
    assert calls(named(updated, "caller")) == []
    assert (
        evaluate(updated, "caller", ())
        == evaluate(original, "caller", ())
        == ((), [("effect", ()), ("effect", ())])
    )
    assert block(caller).operations == [first, second, ir.Return()]


def test_multiple_results_mix_arguments_and_computed_values() -> None:
    a, b, total = value(0), value(1), value(2)
    helper = function("mixed", (a, b), [ir.Add(total, a, b), ir.Return((b, total, b))])
    x, y = value(10), value(11)
    outputs = (value(12), value(13), value(14))
    caller = function(
        "caller", (x, y), [ir.Call("mixed", (x, y), outputs), ir.Return(outputs)]
    )
    original = verify.module(ir.Module(functions=[helper, caller]))
    updated = optimizations.inline_functions(original)
    operations = block(named(updated, "caller")).operations
    assert len(operations) == 2
    assert isinstance(operations[0], ir.Add)
    assert operations[1] == ir.Return((y, operations[0].result, y))
    assert (
        evaluate(updated, "caller", (3, 7))
        == evaluate(original, "caller", (3, 7))
        == ((7, 10, 7), [])
    )


def test_nullary_calls_preserve_effect_order_and_unused_computation() -> None:
    constant, result = value(0), value(0)
    helper = function(
        "effectful",
        (),
        [
            ir.UnknownOp("inside", (), ()),
            ir.Const(constant, 42),
            ir.Return((constant,)),
        ],
    )
    caller = function(
        "caller",
        (),
        [
            ir.UnknownOp("before", (), ()),
            ir.Call("effectful", (), (result,)),
            ir.UnknownOp("after", (), ()),
            ir.Return(),
        ],
    )
    original = verify.module(ir.Module(functions=[helper, caller]))
    updated = optimizations.inline_functions(original)
    assert calls(named(updated, "caller")) == []
    assert any(
        isinstance(op, ir.Const) for op in block(named(updated, "caller")).operations
    )
    assert (
        evaluate(updated, "caller", ())
        == evaluate(original, "caller", ())
        == ((), [("before", ()), ("inside", ()), ("after", ())])
    )


@pytest.mark.parametrize("condition", [False, True])
def test_nested_calls_clone_nested_regions_and_captures(condition: bool) -> None:
    x, flag, doubled, result = (
        value(0),
        value(1, ir.ScalarType.BOOL),
        value(2),
        value(3),
    )
    helper = function(
        "choose",
        (x, flag),
        [
            ir.If(
                (result,),
                flag,
                region(
                    ir.Add(doubled, x, x),
                    ir.UnknownOp("then", (doubled,), ()),
                    ir.Yield((doubled,)),
                ),
                region(ir.UnknownOp("else", (x,), ()), ir.Yield((x,))),
            ),
            ir.Return((result,)),
        ],
    )
    argument, predicate = value(0), value(1, ir.ScalarType.BOOL)
    then_result, else_result, output = value(2), value(3), value(4)
    caller = function(
        "caller",
        (argument, predicate),
        [
            ir.UnknownOp("before", (argument,), ()),
            ir.If(
                (output,),
                predicate,
                region(
                    ir.Call("choose", (argument, predicate), (then_result,)),
                    ir.Yield((then_result,)),
                ),
                region(
                    ir.Call("choose", (argument, predicate), (else_result,)),
                    ir.Yield((else_result,)),
                ),
            ),
            ir.UnknownOp("after", (output,), ()),
            ir.Return((output,)),
        ],
    )
    original = verify.module(ir.Module(functions=[caller, helper]))
    snapshot = pickle.dumps(original)
    updated = optimizations.inline_functions(original)
    optimized = named(updated, "caller")
    assert calls(optimized) == []
    assert optimized.body is not None
    assert sum(isinstance(op, ir.If) for op in ir.iter_ops(optimized.body)) == 3
    expected = 6 if condition else 3
    assert (
        evaluate(updated, "caller", (3, condition))
        == evaluate(original, "caller", (3, condition))
        == (
            (expected,),
            [
                ("before", (3,)),
                ("then" if condition else "else", (expected,)),
                ("after", (expected,)),
            ],
        )
    )
    assert pickle.dumps(original) == snapshot


@pytest.mark.parametrize(
    "limit, inlined", [(0, False), (5, False), (6, True), (None, True)]
)
def test_size_limit_counts_nested_operations_and_terminators(
    limit: int | None, inlined: bool
) -> None:
    flag, one, two, output = value(0, ir.ScalarType.BOOL), value(1), value(2), value(3)
    # If + two Consts + two Yields + Return = six operations.
    helper = function(
        "choose",
        (flag,),
        [
            ir.If(
                (output,),
                flag,
                region(ir.Const(one, 1), ir.Yield((one,))),
                region(ir.Const(two, 2), ir.Yield((two,))),
            ),
            ir.Return((output,)),
        ],
    )
    predicate, result = value(0, ir.ScalarType.BOOL), value(1)
    caller = function(
        "caller",
        (predicate,),
        [ir.Call("choose", (predicate,), (result,)), ir.Return((result,))],
    )
    original = verify.module(ir.Module(functions=[caller, helper]))
    updated = optimizations.inline_functions(original, max_callee_ops=limit)
    assert bool(calls(named(updated, "caller"))) is not inlined
    if not inlined:
        assert updated is original


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("leaf", "middle", "root")))
)
@pytest.mark.parametrize("limit", [4, None])
def test_bottom_up_inlining_uses_expanded_size_independently_of_function_order(
    order: tuple[str, ...], limit: int | None
) -> None:
    x, a, b, c = (value(i) for i in range(4))
    leaf = function(
        "leaf",
        (x,),
        [ir.Add(a, x, x), ir.Add(b, a, a), ir.Add(c, b, b), ir.Return((c,))],
    )
    middle = function(
        "middle",
        (x,),
        [ir.Call("leaf", (x,), (a,)), ir.Call("leaf", (a,), (b,)), ir.Return((b,))],
    )
    root = function("root", (x,), [ir.Call("middle", (x,), (a,)), ir.Return((a,))])
    functions = {f.name: f for f in (leaf, middle, root)}
    original = verify.module(ir.Module(functions=[functions[name] for name in order]))
    updated = optimizations.inline_functions(original, max_callee_ops=limit)
    assert [f.name for f in updated.functions] == list(order)
    assert calls(named(updated, "middle")) == []
    assert len(block(named(updated, "middle")).operations) == 7
    root_calls = calls(named(updated, "root"))
    assert [call.callee for call in root_calls] == (["middle"] if limit == 4 else [])
    assert (
        evaluate(updated, "root", (3,))
        == evaluate(original, "root", (3,))
        == ((192,), [])
    )
    assert optimizations.inline_functions(updated, max_callee_ops=limit) is updated


@pytest.mark.parametrize("mutual", [False, True])
def test_recursive_callees_are_skipped_but_their_nonrecursive_helpers_are_inlined(
    mutual: bool,
) -> None:
    x, a, b = value(0), value(1), value(2)
    leaf = function("leaf", (x,), [ir.Add(a, x, x), ir.Return((a,))])
    recursive = function(
        "recursive",
        (x,),
        [
            ir.Call("leaf", (x,), (a,)),
            ir.Call("partner" if mutual else "recursive", (a,), (b,)),
            ir.Return((b,)),
        ],
    )
    wrapper = function(
        "wrapper", (x,), [ir.Call("recursive", (x,), (a,)), ir.Return((a,))]
    )
    root = function("root", (x,), [ir.Call("wrapper", (x,), (a,)), ir.Return((a,))])
    functions = [root, wrapper, recursive, leaf]
    if mutual:
        functions.append(
            function(
                "partner", (x,), [ir.Call("recursive", (x,), (a,)), ir.Return((a,))]
            )
        )
    original = verify.module(ir.Module(functions=functions))
    updated = optimizations.inline_functions(original)
    assert [call.callee for call in calls(named(updated, "root"))] == ["recursive"]
    assert [call.callee for call in calls(named(updated, "recursive"))] == [
        "partner" if mutual else "recursive"
    ]
    assert isinstance(block(named(updated, "recursive")).operations[0], ir.Add)
    if mutual:
        assert named(updated, "partner") is named(original, "partner")
    assert optimizations.inline_functions(updated) is updated


def test_call_cycles_inside_regions_are_detected() -> None:
    flag, x, local, output = value(0, ir.ScalarType.BOOL), value(1), value(2), value(3)
    recursive = function(
        "recursive",
        (flag, x),
        [
            ir.If(
                (output,),
                flag,
                region(ir.Call("recursive", (flag, x), (local,)), ir.Yield((local,))),
                region(ir.Yield((x,))),
            ),
            ir.Return((output,)),
        ],
    )
    caller = function(
        "caller",
        (flag, x),
        [ir.Call("recursive", (flag, x), (local,)), ir.Return((local,))],
    )
    original = verify.module(ir.Module(functions=[caller, recursive]))
    assert optimizations.inline_functions(original) is original


def test_external_calls_survive_inlined_wrappers() -> None:
    x, result = value(0), value(1)
    external = ir.Function("external", ir.FunctionType((x.type,), (x.type,)))
    wrapper = function(
        "wrapper", (x,), [ir.Call("external", (x,), (result,)), ir.Return((result,))]
    )
    caller = function(
        "caller", (x,), [ir.Call("wrapper", (x,), (result,)), ir.Return((result,))]
    )
    original = verify.module(ir.Module(functions=[caller, external, wrapper]))
    updated = optimizations.inline_functions(original)
    assert named(updated, "external") is external
    assert named(updated, "wrapper") is wrapper
    assert [call.callee for call in calls(named(updated, "caller"))] == ["external"]


def test_preserves_globals_signatures_names_metadata_and_definitions() -> None:
    x, offset, result = value(0), value(1), value(2)
    helper = function(
        "offset",
        (x,),
        [
            ir.GetGlobal("offset_value", offset),
            ir.Add(result, x, offset),
            ir.Return((result,)),
        ],
    )
    caller = dataclasses.replace(
        function(
            "caller", (x,), [ir.Call("offset", (x,), (result,)), ir.Return((result,))]
        ),
        input_names=("input",),
        output_names=("output",),
        attributes={"tag": "caller"},
    )
    global_ = ir.Global("offset_value", ir.ScalarType.I32, 7)
    original = verify.module(
        ir.Module(
            functions=[caller, helper], globals=[global_], attributes={"tag": "module"}
        )
    )
    snapshot = pickle.dumps(original)
    updated = optimizations.inline_functions(original)
    optimized = named(updated, "caller")
    assert optimized.type == caller.type
    assert optimized.input_names == caller.input_names
    assert optimized.output_names == caller.output_names
    assert optimized.attributes == caller.attributes
    assert updated.attributes == original.attributes
    assert updated.globals[0] is global_
    assert named(updated, "offset") is helper
    assert (
        evaluate(updated, "caller", (3,))
        == evaluate(original, "caller", (3,))
        == ((10,), [])
    )
    assert pickle.dumps(original) == snapshot


def test_inlining_exposes_transposes_to_simplification() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 3))
    swapped = ir.TensorType(ir.ScalarType.F32, (3, 2))
    arg, back = value(0, swapped), value(1, tensor)
    helper = function(
        "transpose_back", (arg,), [ir.Transpose(back, arg, (1, 0)), ir.Return((back,))]
    )
    x, transposed, result = value(0, tensor), value(1, swapped), value(2, tensor)
    caller = function(
        "caller",
        (x,),
        [
            ir.Transpose(transposed, x, (1, 0)),
            ir.Call("transpose_back", (transposed,), (result,)),
            ir.Return((result,)),
        ],
    )
    original = verify.module(ir.Module(functions=[caller, helper]))
    updated = optimizations.simplify_transposes(
        optimizations.inline_functions(original)
    )
    assert block(named(updated, "caller")).operations == [ir.Return((x,))]


def test_empty_and_external_only_modules_are_unchanged() -> None:
    for functions in ([], [ir.Function("external", ir.FunctionType((), ()))]):
        original = verify.module(ir.Module(functions=functions))
        assert optimizations.inline_functions(original) is original


def test_negative_limit_is_rejected_without_mutating_input() -> None:
    original = verify.module(ir.Module())
    snapshot = pickle.dumps(original)
    with pytest.raises(ValueError, match="nonnegative"):
        optimizations.inline_functions(original, max_callee_ops=-1)
    assert pickle.dumps(original) == snapshot


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("leaf", "middle", "root")))
)
@pytest.mark.parametrize(
    "callees, root_targets, middle_targets",
    [
        (None, [], []),
        (set(), ["middle"], ["leaf"]),
        ({"leaf"}, ["middle"], []),
        (frozenset({"middle"}), ["leaf"], ["leaf"]),
        ({"leaf", "middle"}, [], []),
        ({"root"}, ["middle"], ["leaf"]),
    ],
)
def test_selection_applies_to_targets_in_every_caller(
    order: tuple[str, ...],
    callees: AbstractSet[ir.SymbolName] | None,
    root_targets: list[str],
    middle_targets: list[str],
) -> None:
    x, result = value(0), value(1)
    leaf = function("leaf", (x,), [ir.Add(result, x, x), ir.Return((result,))])
    middle = function(
        "middle", (x,), [ir.Call("leaf", (x,), (result,)), ir.Return((result,))]
    )
    root = function(
        "root", (x,), [ir.Call("middle", (x,), (result,)), ir.Return((result,))]
    )
    functions = {f.name: f for f in (leaf, middle, root)}
    original = verify.module(ir.Module(functions=[functions[name] for name in order]))
    snapshot = pickle.dumps(original)
    selected_snapshot = None if callees is None else set(callees)
    updated = optimizations.inline_functions(original, callees=callees)
    assert [op.callee for op in calls(named(updated, "root"))] == root_targets
    assert [op.callee for op in calls(named(updated, "middle"))] == middle_targets
    assert evaluate(updated, "root", (7,)) == evaluate(original, "root", (7,))
    assert pickle.dumps(original) == snapshot
    assert callees == selected_snapshot
    assert optimizations.inline_functions(updated, callees=callees) is updated
    if callees in (set(), {"root"}):
        assert updated is original


@pytest.mark.parametrize(
    "callees, root_targets",
    [({"middle"}, ["leaf", "leaf"]), ({"middle", "leaf"}, ["middle"])],
)
def test_selection_controls_expansion_before_size_check(
    callees: AbstractSet[ir.SymbolName], root_targets: list[str]
) -> None:
    x, a, b, c = (value(i) for i in range(4))
    leaf = function(
        "leaf",
        (x,),
        [ir.Add(a, x, x), ir.Add(b, a, a), ir.Add(c, b, b), ir.Return((c,))],
    )
    middle = function(
        "middle",
        (x,),
        [ir.Call("leaf", (x,), (a,)), ir.Call("leaf", (a,), (b,)), ir.Return((b,))],
    )
    root = function("root", (x,), [ir.Call("middle", (x,), (a,)), ir.Return((a,))])
    original = verify.module(ir.Module(functions=[root, middle, leaf]))
    updated = optimizations.inline_functions(
        original, callees=callees, max_callee_ops=4
    )
    assert [op.callee for op in calls(named(updated, "root"))] == root_targets
    assert evaluate(updated, "root", (3,)) == evaluate(original, "root", (3,))
    assert (
        optimizations.inline_functions(updated, callees=callees, max_callee_ops=4)
        is updated
    )


def test_selection_does_not_hide_cycles_through_unselected_functions() -> None:
    x, a = value(0), value(1)
    first = function("first", (x,), [ir.Call("second", (x,), (a,)), ir.Return((a,))])
    second = function("second", (x,), [ir.Call("first", (x,), (a,)), ir.Return((a,))])
    root = function("root", (x,), [ir.Call("first", (x,), (a,)), ir.Return((a,))])
    external = ir.Function("external", root.type)
    original = verify.module(ir.Module(functions=[root, first, second, external]))
    assert (
        optimizations.inline_functions(original, callees={"first", "external"})
        is original
    )


@pytest.mark.parametrize("callees", [{"missing"}, {"global"}, {"helper", "missing"}])
def test_selection_rejects_unknown_functions_without_mutation(
    callees: AbstractSet[ir.SymbolName],
) -> None:
    x = value(0)
    helper = function("helper", (x,), [ir.Return((x,))])
    original = verify.module(
        ir.Module(functions=[helper], globals=[ir.Global("global", x.type, 0)])
    )
    snapshot = pickle.dumps(original)
    with pytest.raises(ValueError, match="Unknown callee functions"):
        optimizations.inline_functions(original, callees=callees)
    assert pickle.dumps(original) == snapshot


def test_inlines_single_block_helpers_in_cfg_callers_and_skips_cfg_callees() -> None:
    x, result = value(0), value(1)
    helper = function("helper", (x,), [ir.Add(result, x, x), ir.Return((result,))])
    exit_ = ir.Block(operations=[ir.Return((result,))])
    entry = ir.Block((x,), [ir.Call("helper", (x,), (result,)), ir.Branch(exit_)])
    cfg = ir.Function("cfg", helper.type, ir.Region([entry, exit_]))
    caller = function(
        "caller", (x,), [ir.Call("cfg", (x,), (result,)), ir.Return((result,))]
    )
    original = verify.module(ir.Module(functions=[helper, cfg, caller]))
    snapshot = pickle.dumps(original)
    updated = optimizations.inline_functions(original)
    assert calls(named(updated, "cfg")) == []
    assert [op.callee for op in calls(named(updated, "caller"))] == ["cfg"]
    updated_cfg = named(updated, "cfg")
    assert updated_cfg.body is not None
    assert ir.get_successors(block(updated_cfg).operations[-1]) == (
        updated_cfg.body.blocks[1],
    )
    assert pickle.dumps(original) == snapshot
    assert optimizations.inline_functions(updated) is updated
