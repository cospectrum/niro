from typing import assert_type

import pytest

from niro import ir, verify


def test_empty_module_can_be_verified_repeatedly() -> None:
    module = ir.Module()

    verified = verify.module(module)
    assert_type(verified, ir.VerifiedModule)
    assert verified is module
    assert verify.module(module) is module
    assert module.functions == []
    assert module.globals == []


@pytest.mark.parametrize(
    ("type_", "initializer"),
    [
        (ir.ScalarType.BOOL, True),
        (ir.ScalarType.I32, -1),
        (ir.ScalarType.I64, 42),
        (ir.ScalarType.F32, 1.5),
        (ir.ScalarType.F64, 2.5),
        (ir.TensorType(ir.ScalarType.F32, (2, 3)), bytes(24)),
        (ir.TensorType(ir.ScalarType.I64, ()), bytes(8)),
        (ir.TensorType(ir.ScalarType.BOOL, (2,)), bytes(2)),
        (ir.TensorType(ir.ScalarType.I32, (0, 3)), b""),
    ],
)
def test_valid_global_initializers(type_: ir.Type, initializer: ir.Literal) -> None:
    global_ = ir.Global("weight", type_, initializer)
    module = ir.Module(globals=[global_])

    assert verify.module(module) is module
    assert module.globals[0] is global_


@pytest.mark.parametrize("used", [False, True])
@pytest.mark.parametrize(
    ("type_", "initializer", "error", "message"),
    [
        (ir.ScalarType.BOOL, 1, TypeError, "declared type"),
        (ir.ScalarType.I32, True, TypeError, "declared type"),
        (ir.ScalarType.I64, 1.5, TypeError, "declared type"),
        (ir.ScalarType.F32, True, TypeError, "declared type"),
        (ir.ScalarType.F64, b"", TypeError, "declared type"),
        (
            ir.TensorType(ir.ScalarType.F32, (2,)),
            bytes(4),
            ValueError,
            "4 bytes, expected 8",
        ),
        (
            ir.TensorType(ir.ScalarType.F32, (2,)),
            bytes(12),
            ValueError,
            "12 bytes, expected 8",
        ),
        (ir.TensorType(ir.ScalarType.F32, (None,)), b"", TypeError, "static shape"),
        (ir.TensorType(ir.ScalarType.F32, None), b"", TypeError, "static shape"),
        (ir.TensorType(ir.ScalarType.F32, ()), 1.0, TypeError, "packed bytes"),
    ],
)
def test_invalid_global_initializers(
    type_: ir.Type,
    initializer: ir.Literal,
    error: type[Exception],
    message: str,
    used: bool,
) -> None:
    module = ir.Module(
        globals=[
            ir.Global("valid", ir.ScalarType.I32, 0),
            ir.Global("weight", type_, initializer),
        ]
    )
    if used:
        result = ir.Value(ir.ValueId(0), type_)
        module.functions.append(
            ir.Function(
                "main",
                ir.FunctionType((), (type_,)),
                ir.Region(
                    [
                        ir.Block(
                            operations=[
                                ir.GetGlobal("weight", result),
                                ir.Return((result,)),
                            ]
                        )
                    ]
                ),
            )
        )

    with pytest.raises(error, match=f"global 'weight' initializer.*{message}"):
        verify.module(module)


def test_declarations_need_no_body() -> None:
    declaration = ir.Function(
        name="external",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.F32,)),
    )
    module = ir.Module(functions=[declaration])

    assert verify.module(module) is module
    assert declaration.body is None


def test_value_ids_are_local_to_each_function() -> None:
    integer = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    floating = ir.Value(ir.ValueId(0), ir.ScalarType.F32)
    module = ir.Module(
        functions=[
            ir.Function(
                name="integer_identity",
                type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
                body=ir.Region([ir.Block((integer,), [ir.Return((integer,))])]),
            ),
            ir.Function(
                name="float_identity",
                type=ir.FunctionType((ir.ScalarType.F32,), (ir.ScalarType.F32,)),
                body=ir.Region([ir.Block((floating,), [ir.Return((floating,))])]),
            ),
        ]
    )

    assert verify.module(module) is module


def test_symbols_resolve_after_the_caller_is_constructed() -> None:
    loaded = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    result = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    load = ir.GetGlobal("weight", loaded)
    call = ir.Call("convert", (loaded,), (result,))
    block = ir.Block(operations=[load, call, ir.Return((result,))])
    caller = ir.Function(
        name="main",
        type=ir.FunctionType((), (ir.ScalarType.F32,)),
        body=ir.Region([block]),
    )
    module = ir.Module(functions=[caller])
    module.functions.append(
        ir.Function(
            name="convert",
            type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.F32,)),
        )
    )
    module.globals.append(ir.Global("weight", ir.ScalarType.I32, 42))

    assert verify.module(module) is module
    assert block.operations[0] is load
    assert block.operations[1] is call


def test_nested_branches_capture_outer_values_and_publish_only_if_results() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    argument = ir.Value(ir.ValueId(1), ir.ScalarType.I32)
    local = ir.Value(ir.ValueId(2), ir.ScalarType.I32)
    nested_result = ir.Value(ir.ValueId(3), ir.ScalarType.I32)
    outer_result = ir.Value(ir.ValueId(4), ir.ScalarType.I32)
    sum_ = ir.Value(ir.ValueId(5), ir.ScalarType.I32)
    nested = ir.If(
        results=(nested_result,),
        condition=condition,
        then_region=ir.Region([ir.Block(operations=[ir.Yield((local,))])]),
        else_region=ir.Region([ir.Block(operations=[ir.Yield((argument,))])]),
    )
    outer = ir.If(
        results=(outer_result,),
        condition=condition,
        then_region=ir.Region(
            [
                ir.Block(
                    operations=[ir.Const(local, 7), nested, ir.Yield((nested_result,))]
                )
            ]
        ),
        else_region=ir.Region([ir.Block(operations=[ir.Yield((argument,))])]),
    )
    block = ir.Block(
        arguments=(condition, argument),
        operations=[outer, ir.Add(sum_, outer_result, argument), ir.Return((sum_,))],
    )
    function = ir.Function(
        name="select",
        type=ir.FunctionType(
            (ir.ScalarType.BOOL, ir.ScalarType.I32), (ir.ScalarType.I32,)
        ),
        body=ir.Region([block]),
    )
    module = ir.Module(functions=[function])

    assert verify.module(module) is module
    assert block.operations[0] is outer
    assert outer.then_region.blocks[0].operations[1] is nested


@pytest.mark.parametrize("with_else", [False, True])
def test_resultless_if_requires_else_block(with_else: bool) -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    else_region = (
        ir.Region([ir.Block(operations=[ir.Yield()])]) if with_else else ir.Region()
    )
    conditional = ir.If(
        results=(),
        condition=condition,
        then_region=ir.Region([ir.Block(operations=[ir.Yield()])]),
        else_region=else_region,
    )
    module = ir.Module(
        functions=[
            ir.Function(
                name="main",
                type=ir.FunctionType((ir.ScalarType.BOOL,), ()),
                body=ir.Region([ir.Block((condition,), [conditional, ir.Return()])]),
            )
        ]
    )

    if with_else:
        assert verify.module(module) is module
        assert conditional.else_region is else_region
    else:
        with pytest.raises(ValueError, match="region must contain a block"):
            verify.module(module)


def test_unknown_operations_participate_in_value_scope() -> None:
    result = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    source = ir.UnknownOp("example.source", (), (result,))
    consumer = ir.UnknownOp("example.consume", (result,), ())
    module = ir.Module(
        functions=[
            ir.Function(
                name="main",
                type=ir.FunctionType((), (ir.ScalarType.I32,)),
                body=ir.Region(
                    [ir.Block(operations=[source, consumer, ir.Return((result,))])]
                ),
            )
        ]
    )

    assert verify.module(module) is module


def _module_with_body(
    blocks: list[ir.Block],
    inputs: tuple[ir.Type, ...] = (),
    outputs: tuple[ir.Type, ...] = (),
) -> ir.Module:
    return ir.Module(
        functions=[
            ir.Function("main", ir.FunctionType(inputs, outputs), ir.Region(blocks))
        ]
    )


@pytest.mark.parametrize(
    ("functions", "globals_", "message"),
    [
        ([ir.Function("", ir.FunctionType((), ()))], [], "cannot be empty"),
        ([], [ir.Global("", ir.ScalarType.I32, 0)], "cannot be empty"),
        (
            [ir.Function("f", ir.FunctionType((), ()))] * 2,
            [],
            "duplicate module symbol",
        ),
        ([], [ir.Global("g", ir.ScalarType.I32, 0)] * 2, "duplicate module symbol"),
        (
            [ir.Function("shared", ir.FunctionType((), ()))],
            [ir.Global("shared", ir.ScalarType.I32, 0)],
            "duplicate module symbol",
        ),
    ],
    ids=[
        "empty-function",
        "empty-global",
        "duplicate-functions",
        "duplicate-globals",
        "shared-namespace",
    ],
)
def test_invalid_symbol_names(
    functions: list[ir.Function], globals_: list[ir.Global], message: str
) -> None:
    module = ir.Module(functions=functions, globals=globals_)

    with pytest.raises(ValueError, match=message):
        verify.module(module)


@pytest.mark.parametrize(
    "definition", ["argument", "operation", "multi-result", "sibling-branch"]
)
def test_duplicate_value_definitions(definition: str) -> None:
    value = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    condition = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    match definition:
        case "argument":
            block = ir.Block((value, value), [ir.Return()])
            inputs = (ir.ScalarType.I32, ir.ScalarType.I32)
        case "operation":
            block = ir.Block((value,), [ir.Const(value, 1), ir.Return()])
            inputs = (ir.ScalarType.I32,)
        case "multi-result":
            block = ir.Block(
                operations=[ir.UnknownOp("source", (), (value, value)), ir.Return()]
            )
            inputs = ()
        case _:
            conditional = ir.If(
                (),
                condition,
                ir.Region([ir.Block(operations=[ir.Const(value, 1), ir.Yield()])]),
                ir.Region([ir.Block(operations=[ir.Const(value, 2), ir.Yield()])]),
            )
            block = ir.Block((condition,), [conditional, ir.Return()])
            inputs = (ir.ScalarType.BOOL,)
    module = _module_with_body([block], inputs)

    with pytest.raises(ValueError, match="duplicate value ID"):
        verify.module(module)


@pytest.mark.parametrize(
    ("blocks", "inputs", "error", "message"),
    [
        ([], (), ValueError, "region must contain a block"),
        (
            [ir.Block(operations=[ir.Return()]), ir.Block(operations=[ir.Return()])],
            (),
            ValueError,
            "region contains unreachable blocks",
        ),
        ([ir.Block()], (), ValueError, "must end with Return"),
        ([ir.Block(operations=[ir.Yield()])], (), ValueError, "must end with Return"),
        (
            [ir.Block(operations=[ir.Return(), ir.Return()])],
            (),
            ValueError,
            "unexpected block terminator",
        ),
        (
            [ir.Block(operations=[ir.Yield(), ir.Return()])],
            (),
            ValueError,
            "unexpected block terminator",
        ),
        (
            [ir.Block(operations=[ir.Return()])],
            (ir.ScalarType.I32,),
            TypeError,
            "region argument types",
        ),
        (
            [ir.Block((ir.Value(ir.ValueId(0), ir.ScalarType.F32),), [ir.Return()])],
            (ir.ScalarType.I32,),
            TypeError,
            "region argument types",
        ),
    ],
    ids=[
        "empty-body",
        "multiple-blocks",
        "empty-block",
        "wrong-terminator",
        "early-return",
        "early-yield",
        "entry-arity",
        "entry-type",
    ],
)
def test_invalid_function_structure(
    blocks: list[ir.Block],
    inputs: tuple[ir.Type, ...],
    error: type[Exception],
    message: str,
) -> None:
    module = _module_with_body(blocks, inputs)

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize(
    "scenario",
    [
        "undefined",
        "use-before-definition",
        "self-use",
        "wrong-type",
        "sibling-use",
        "branch-escape",
        "if-result-in-branch",
    ],
)
def test_invalid_value_visibility(scenario: str) -> None:
    value = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    condition = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(2), ir.ScalarType.I32)
    inputs: tuple[ir.Type, ...] = ()
    error: type[Exception] = ValueError
    message = "not defined in this scope"
    match scenario:
        case "undefined":
            block = ir.Block(
                operations=[ir.UnknownOp("consume", (value,), ()), ir.Return()]
            )
        case "use-before-definition":
            block = ir.Block(
                operations=[
                    ir.UnknownOp("consume", (value,), ()),
                    ir.Const(value, 1),
                    ir.Return(),
                ]
            )
        case "self-use":
            block = ir.Block(operations=[ir.Add(value, value, value), ir.Return()])
        case "wrong-type":
            wrong = ir.Value(value.id, ir.ScalarType.F32)
            block = ir.Block(
                (value,), [ir.UnknownOp("consume", (wrong,), ()), ir.Return()]
            )
            inputs = (ir.ScalarType.I32,)
            error = TypeError
            message = "type differs from its definition"
        case _:
            then_block = ir.Block(operations=[ir.Const(value, 1), ir.Yield()])
            else_block = ir.Block(operations=[ir.Yield()])
            conditional = ir.If(
                (), condition, ir.Region([then_block]), ir.Region([else_block])
            )
            block = ir.Block((condition,), [conditional, ir.Return()])
            inputs = (ir.ScalarType.BOOL,)
            if scenario == "sibling-use":
                else_block.operations.insert(0, ir.UnknownOp("consume", (value,), ()))
            elif scenario == "branch-escape":
                block.operations.insert(1, ir.UnknownOp("consume", (value,), ()))
            else:
                conditional = ir.If(
                    (result,),
                    condition,
                    ir.Region([ir.Block(operations=[ir.Yield((result,))])]),
                    ir.Region([ir.Block(operations=[ir.Yield((result,))])]),
                )
                block.operations[0] = conditional
    module = _module_with_body([block], inputs)

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize(
    "scenario",
    [
        "unknown-callee",
        "call-arity",
        "call-input-type",
        "call-results",
        "unknown-global",
        "global-type",
        "return-type",
        "return-arity",
    ],
)
def test_invalid_references_and_result_types(scenario: str) -> None:
    argument = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    result = ir.Value(ir.ValueId(1), ir.ScalarType.F32)
    operations: list[ir.Op] = []
    outputs: tuple[ir.Type, ...] = ()
    error: type[Exception] = TypeError
    match scenario:
        case "unknown-callee":
            operations = [ir.Call("missing", (), ())]
            error, message = ValueError, "unknown function"
        case "call-arity":
            operations = [ir.Call("callee", (), (result,))]
            message = "call argument types"
        case "call-input-type":
            operations = [ir.Call("callee", (argument,), (result,))]
            message = "call argument types"
        case "call-results":
            operations = [ir.Call("callee", (), (result,))]
            message = "call result types"
        case "unknown-global":
            operations = [ir.GetGlobal("missing", result)]
            error, message = ValueError, "unknown global"
        case "global-type":
            operations = [ir.GetGlobal("weight", result)]
            message = "global load type"
        case "return-type":
            operations = [ir.Return((argument,))]
            outputs = (ir.ScalarType.F32,)
            message = "terminator operand types"
        case _:
            outputs = (ir.ScalarType.I32,)
            message = "terminator operand types"
    if not operations or not isinstance(operations[-1], ir.Return):
        operations.append(ir.Return())
    module = _module_with_body(
        [ir.Block((argument,), operations)], (ir.ScalarType.I32,), outputs
    )
    callee_inputs = () if scenario == "call-results" else (ir.ScalarType.F32,)
    module.functions.append(
        ir.Function("callee", ir.FunctionType(callee_inputs, (ir.ScalarType.I32,)))
    )
    module.globals.append(ir.Global("weight", ir.ScalarType.I32, 42))

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize(
    "scenario",
    [
        "condition-type",
        "empty-then",
        "multiple-blocks",
        "branch-arguments",
        "missing-else",
        "yield-type",
        "yield-arity",
        "return-in-branch",
    ],
)
def test_invalid_if_regions(scenario: str) -> None:
    condition_type = (
        ir.ScalarType.I32 if scenario == "condition-type" else ir.ScalarType.BOOL
    )
    condition = ir.Value(ir.ValueId(0), condition_type)
    value = ir.Value(ir.ValueId(1), ir.ScalarType.I32)
    result = ir.Value(ir.ValueId(2), ir.ScalarType.I32)
    then = ir.Region([ir.Block(operations=[ir.Yield((value,))])])
    else_ = ir.Region([ir.Block(operations=[ir.Yield((value,))])])
    error: type[Exception] = ValueError
    match scenario:
        case "condition-type":
            error, message = TypeError, "if condition must be boolean"
        case "empty-then":
            then.blocks.clear()
            message = "region must contain a block"
        case "multiple-blocks":
            then.blocks.append(ir.Block(operations=[ir.Yield((value,))]))
            message = "region contains unreachable blocks"
        case "branch-arguments":
            then.blocks[0].arguments = (ir.Value(ir.ValueId(3), ir.ScalarType.I32),)
            error, message = TypeError, "region argument types"
        case "missing-else":
            else_.blocks.clear()
            message = "region must contain a block"
        case "yield-type":
            then.blocks[0].operations = [ir.Yield((condition,))]
            error, message = TypeError, "terminator operand types"
        case "yield-arity":
            then.blocks[0].operations = [ir.Yield()]
            error, message = TypeError, "terminator operand types"
        case _:
            then.blocks[0].operations = [ir.Return((value,))]
            message = "must end with Yield"
    conditional = ir.If((result,), condition, then, else_)
    block = ir.Block((condition, value), [conditional, ir.Return((result,))])
    module = _module_with_body(
        [block], (condition_type, ir.ScalarType.I32), (ir.ScalarType.I32,)
    )

    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize("with_body", [False, True])
@pytest.mark.parametrize("names", [None, (None,), ("value",)])
def test_valid_interface_names(
    with_body: bool, names: tuple[str | None, ...] | None
) -> None:
    argument = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    body = (
        ir.Region([ir.Block((argument,), [ir.Return((argument,))])])
        if with_body
        else None
    )
    function = ir.Function(
        "identity",
        ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
        body,
        input_names=names,
        output_names=names,
    )
    module = ir.Module(functions=[function])

    assert verify.module(module) is module


@pytest.mark.parametrize("with_body", [False, True])
@pytest.mark.parametrize("kind", ["input", "output"])
@pytest.mark.parametrize(
    ("names", "message"), [((), "arity"), (("",), "cannot be empty")]
)
def test_invalid_interface_names(
    with_body: bool, kind: str, names: tuple[str, ...], message: str
) -> None:
    argument = ir.Value(ir.ValueId(0), ir.ScalarType.I32)
    body = (
        ir.Region([ir.Block((argument,), [ir.Return((argument,))])])
        if with_body
        else None
    )
    function = ir.Function(
        "identity",
        ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
        body,
        input_names=names if kind == "input" else None,
        output_names=names if kind == "output" else None,
    )

    with pytest.raises(ValueError, match=f"{kind} names.*{message}"):
        verify.module(ir.Module(functions=[function]))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("invalid_use", [False, True])
def test_diamond_dominance_and_block_arguments(
    reverse: bool, invalid_use: bool
) -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    x, left_value, merged = (
        ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(1, 4)
    )
    join = ir.Block((merged,), [ir.Return((left_value if invalid_use else merged,))])
    left = ir.Block(
        operations=[ir.Add(left_value, x, x), ir.Branch(join, (left_value,))]
    )
    right = ir.Block(operations=[ir.Branch(join, (x,))])
    entry = ir.Block((flag, x), [ir.CondBranch(flag, left, right)])
    rest = [left, right, join]
    module = _module_with_body(
        [entry, *(reversed(rest) if reverse else rest)],
        (flag.type, x.type),
        (x.type,),
    )
    if invalid_use:
        with pytest.raises(ValueError, match="not defined in this scope"):
            verify.module(module)
    else:
        assert verify.module(module) is module


def test_loop_carried_arguments_and_nonterminating_loop() -> None:
    x, carried, updated = (ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(3))
    loop = ir.Block((carried,))
    loop.operations = [ir.Add(updated, carried, x), ir.Branch(loop, (updated,))]
    entry = ir.Block((x,), [ir.Branch(loop, (x,))])
    module = _module_with_body([entry, loop], (x.type,))
    assert verify.module(module) is module


@pytest.mark.parametrize(
    "scenario",
    [
        "foreign",
        "entry",
        "arity",
        "type",
        "condition",
        "early",
        "forward",
        "reference-type",
    ],
)
def test_invalid_control_flow(scenario: str) -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    x, arg, later = (ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(1, 4))
    target = ir.Block((arg,), [ir.Return()])
    entry = ir.Block((flag, x), [ir.CondBranch(flag, target, target, (x,), (x,))])
    error: type[Exception] = ValueError
    match scenario:
        case "foreign":
            entry.operations = [ir.Branch(ir.Block((arg,), [ir.Return()]), (x,))]
            message = "outside the current region"
        case "entry":
            target.operations = [ir.Branch(entry, (flag, x))]
            message = "entry block"
        case "arity":
            entry.operations = [ir.CondBranch(flag, target, target, (x,), ())]
            error, message = TypeError, "branch argument types"
        case "type":
            entry.operations = [ir.CondBranch(flag, target, target, (x,), (flag,))]
            error, message = TypeError, "branch argument types"
        case "condition":
            entry.operations = [ir.CondBranch(x, target, target, (x,), (x,))]
            error, message = TypeError, "condition must be boolean"
        case "early":
            entry.operations.append(ir.Return())
            message = "unexpected block terminator"
        case "forward":
            target.operations = [ir.Add(arg, later, x), ir.Const(later, 1), ir.Return()]
            target.arguments = ()
            entry.operations = [ir.Branch(target)]
            message = "not defined in this scope"
        case _:
            target.operations = [ir.Return((ir.Value(x.id, ir.ScalarType.BOOL),))]
            error, message = TypeError, "type differs from its definition"
    module = _module_with_body([entry, target], (flag.type, x.type))
    with pytest.raises(error, match=message):
        verify.module(module)


@pytest.mark.parametrize("escape", [False, True])
def test_cfg_in_if_region_captures_dominating_values(escape: bool) -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    x, doubled, result = (
        ir.Value(ir.ValueId(i), ir.ScalarType.I32) for i in range(1, 4)
    )
    then_exit = ir.Block(operations=[ir.Yield((doubled,))])
    then_entry = ir.Block(operations=[ir.Branch(then_exit)])
    nested = ir.If(
        (result,),
        flag,
        ir.Region([then_entry, then_exit]),
        ir.Region([ir.Block(operations=[ir.Yield((x,))])]),
    )
    exit_ = ir.Block(operations=[nested, ir.Return((result,))])
    entry = ir.Block((flag, x), [ir.Add(doubled, x, x), ir.Branch(exit_)])
    if escape:
        then_entry.operations = [ir.Branch(exit_)]
    module = _module_with_body([entry, exit_], (flag.type, x.type), (x.type,))
    if escape:
        with pytest.raises(ValueError, match="outside the current region"):
            verify.module(module)
    else:
        assert verify.module(module) is module


@pytest.mark.parametrize(
    "scenario", ["duplicate-block", "shared-region", "cyclic-ownership"]
)
def test_control_flow_ownership(scenario: str) -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    body = ir.Region([ir.Block((flag,), [ir.Return()])])
    if scenario == "duplicate-block":
        body.blocks.append(body.blocks[0])
    else:
        nested = ir.Region([ir.Block(operations=[ir.Yield()])])
        body.blocks[0].operations.insert(
            0,
            ir.If((), flag, nested, body if scenario == "cyclic-ownership" else nested),
        )
    module = _module_with_body(body.blocks, (flag.type,))
    with pytest.raises(ValueError, match="unique owner"):
        verify.module(module)
