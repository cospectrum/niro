from copy import deepcopy
from dataclasses import replace
from typing import cast

import pytest

from niro import ir
from niro.builder import ModuleBuilder


def value(id: int, type: ir.Type = ir.ScalarType.I32) -> ir.Value:
    return ir.Value(ir.ValueId(id), type)


def module_with(
    operations: list[ir.Op],
    arguments: tuple[ir.Value, ...] = (),
    outputs: tuple[ir.Type, ...] = (),
) -> ir.Module:
    return ir.Module(
        functions=[
            ir.Function(
                "main",
                ir.FunctionType(
                    tuple(argument.type for argument in arguments), outputs
                ),
                ir.Region([ir.Block(arguments, operations)]),
            )
        ]
    )


def conditional_module() -> tuple[ir.Module, ir.If]:
    condition, outer, result, local = (
        value(0, ir.ScalarType.BOOL),
        value(1),
        value(2),
        value(3),
    )
    conditional = ir.If(
        (result,),
        condition,
        ir.Region([ir.Block(operations=[ir.Const(local, 1), ir.Yield((local,))])]),
        ir.Region([ir.Block(operations=[ir.Yield((outer,))])]),
    )
    module = module_with(
        [conditional, ir.Return((result,))], (condition, outer), (ir.ScalarType.I32,)
    )
    return module, conditional


def test_verifies_built_module_without_mutation() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (2, 2))
    builder = ModuleBuilder()
    main = builder.function(
        name="main", type=ir.FunctionType((ir.ScalarType.BOOL, tensor), (tensor,))
    )
    helper = builder.function(name="helper", type=ir.FunctionType((tensor,), (tensor,)))
    helper_block = helper.region().first_block()
    helper_block.return_(*helper_block.inner.arguments)
    external = builder.function(name="external", type=ir.FunctionType((), ()))
    weight = builder.global_("weight", tensor, bytes(16))
    block = main.region().first_block()
    condition, operand = block.inner.arguments
    result = block.matmul(operand, block.get_global(weight))
    result = block.add(result, block.transpose(result, (1, 0)))
    result = block.mul(result, block.tensor(bytes(16), tensor))
    conditional = block.if_(condition, (tensor,))
    then = conditional.then_region.first_block()
    nested = then.if_(condition, (tensor,))
    nested.then_region.first_block().yield_(result)
    other = nested.else_region.first_block()
    other.yield_(*other.unknown_op("custom.identity", (result,), (tensor,)))
    then.yield_(*nested.inner.results)
    conditional.else_region.first_block().yield_(result)
    block.call(external)
    block.return_(*block.call(helper, conditional.inner.results))
    original = deepcopy(builder.inner)

    assert ir.verify(builder.inner) is None
    assert builder.inner == original


def test_accepts_empty_modules_and_recursive_calls() -> None:
    ir.verify(ir.Module())
    ir.verify(module_with([ir.Call("main", (), ()), ir.Return()]))


@pytest.mark.parametrize(
    "value_type",
    [
        ir.TensorType(ir.ScalarType.F32, (-1,)),
        ir.TensorType(ir.ScalarType.F32, (True,)),
        ir.FunctionType((), (ir.TensorType(ir.ScalarType.I32, (-1,)),)),
    ],
)
def test_checks_types_explicitly(value_type: ir.Type | ir.FunctionType) -> None:
    with pytest.raises(ir.VerificationError, match="non-negative integer"):
        ir.verify_type(value_type)


@pytest.mark.parametrize(
    "bad", [value(-1), value(True), value(0, ir.TensorType(ir.ScalarType.F32, (-1,)))]
)
def test_checks_value_ids_and_types(bad: ir.Value) -> None:
    with pytest.raises(ir.VerificationError, match="non-negative integer"):
        ir.verify_value(bad)


@pytest.mark.parametrize(
    "type, literal",
    [
        (ir.ScalarType.BOOL, True),
        (ir.ScalarType.I32, -(1 << 31)),
        (ir.ScalarType.I32, (1 << 31) - 1),
        (ir.ScalarType.I64, -(1 << 63)),
        (ir.ScalarType.I64, (1 << 63) - 1),
        (ir.ScalarType.F32, float("nan")),
        (ir.ScalarType.F32, float("inf")),
        (ir.ScalarType.F64, 1e300),
        (ir.TensorType(ir.ScalarType.F32, ()), bytes(4)),
        (ir.TensorType(ir.ScalarType.F32, (0, 2)), b""),
    ],
)
def test_literal_boundaries(type: ir.Type, literal: ir.Literal) -> None:
    ir.verify_op(ir.Const(value(0, type), literal))
    ir.verify(ir.Module(globals=[ir.Global("constant", type, literal)]))


@pytest.mark.parametrize(
    "type, literal, error",
    [
        (ir.ScalarType.I32, True, "does not match"),
        (ir.ScalarType.BOOL, 1, "does not match"),
        (ir.ScalarType.F32, 1, "does not match"),
        (ir.ScalarType.I32, 1 << 31, "out of range"),
        (ir.ScalarType.I64, -(1 << 63) - 1, "out of range"),
        (ir.ScalarType.F32, 1e300, "out of range"),
        (ir.TensorType(ir.ScalarType.F32, (2,)), bytes(4), "4 bytes, expected 8"),
        (ir.TensorType(ir.ScalarType.F32, (None,)), b"", "static shape"),
        (ir.TensorType(ir.ScalarType.F32, None), b"", "static shape"),
    ],
)
def test_rejects_invalid_literals(
    type: ir.Type, literal: ir.Literal, error: str
) -> None:
    constant = ir.Const(value(0, type), literal)
    global_ = ir.Global("constant", type, literal)
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify_op(constant)
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify_global(global_)


@pytest.mark.parametrize("operation", [ir.Add, ir.Mul, ir.MatMul])
def test_rejects_boolean_arithmetic(
    operation: type[ir.Add | ir.Mul | ir.MatMul],
) -> None:
    tensor = ir.TensorType(ir.ScalarType.BOOL, (2, 2))
    op = operation(value(2, tensor), value(0, tensor), value(1, tensor))
    with pytest.raises(ir.VerificationError, match="does not support boolean"):
        ir.verify_op(op)


def test_checks_arithmetic_shapes_and_inference() -> None:
    lhs = value(0, ir.TensorType(ir.ScalarType.F32, (2, 3)))
    rhs = value(1, ir.TensorType(ir.ScalarType.F32, (3, 4)))
    expected = ir.TensorType(ir.ScalarType.F32, (2, 4))
    assert ir.matmul_result_type(lhs.type, rhs.type) == expected
    ir.verify_op(ir.MatMul(value(2, expected), lhs, rhs))
    with pytest.raises(ir.VerificationError, match="result type"):
        ir.verify_op(ir.MatMul(value(2, lhs.type), lhs, rhs))
    with pytest.raises(ir.VerificationError, match="contracting dimensions"):
        ir.matmul_result_type(rhs.type, lhs.type)
    with pytest.raises(ir.VerificationError, match="same type"):
        ir.verify_op(ir.Add(value(2, lhs.type), lhs, rhs))
    assert ir.matmul_result_type(
        ir.TensorType(ir.ScalarType.F32, (None, 3)), rhs.type
    ) == ir.TensorType(ir.ScalarType.F32, (None, 4))


@pytest.mark.parametrize("shape", [None, (2, 3)])
@pytest.mark.parametrize("permutation", [(0, 0), (-1, 0), (1, 2), (True, 0)])
def test_checks_transpose_permutation_even_with_unknown_rank(
    shape: ir.Shape | None, permutation: tuple[int, ...]
) -> None:
    with pytest.raises(ir.VerificationError, match="permutation"):
        ir.transpose_result_type(ir.TensorType(ir.ScalarType.F32, shape), permutation)


@pytest.mark.parametrize(
    "function, error",
    [
        (ir.Function("", ir.FunctionType((), ())), "function name"),
        (
            ir.Function("f", ir.FunctionType((), ()), input_names=("x",)),
            "input names must match",
        ),
        (
            ir.Function(
                "f", ir.FunctionType((), (ir.ScalarType.I32,)), output_names=("",)
            ),
            "output name",
        ),
        (
            ir.Function(
                "f", ir.FunctionType((ir.TensorType(ir.ScalarType.I32, (-1,)),), ())
            ),
            "non-negative integer",
        ),
    ],
)
def test_checks_function_declarations(function: ir.Function, error: str) -> None:
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify_function_signature(function)
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify(ir.Module(functions=[function]))


def test_checks_symbol_collisions_after_mutation() -> None:
    module = module_with([ir.Return()])
    module.globals.append(ir.Global("main", ir.ScalarType.I32, 1))
    with pytest.raises(ir.VerificationError, match="symbol names must be unique"):
        ir.verify(module)


@pytest.mark.parametrize(
    "operations, error",
    [
        ([], "must end with Return"),
        ([ir.Const(value(0), 1)], "must end with Return"),
        ([ir.Return(), ir.Const(value(0), 1)], "terminator must be the last"),
        ([ir.Yield()], "expected Return"),
    ],
)
def test_checks_function_terminators(operations: list[ir.Op], error: str) -> None:
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify(module_with(operations))


@pytest.mark.parametrize("count", [0, 2])
def test_requires_one_block_per_region(count: int) -> None:
    module, conditional = conditional_module()
    conditional.then_region.blocks = [
        ir.Block(operations=[ir.Yield((value(1),))]) for _ in range(count)
    ]
    with pytest.raises(
        ir.VerificationError, match="then region: region must contain exactly one block"
    ):
        ir.verify(module)


def test_checks_function_entry_and_return_types() -> None:
    module = module_with([ir.Return()], (value(0),))
    module.functions[0].type = ir.FunctionType((), ())
    with pytest.raises(ir.VerificationError, match="block argument types"):
        ir.verify(module)
    module = module_with([ir.Return((value(0),))], (value(0),), (ir.ScalarType.F32,))
    with pytest.raises(ir.VerificationError, match="Return operand types"):
        ir.verify(module)


@pytest.mark.parametrize(
    "operations, arguments",
    [
        ([ir.Return()], (value(0), value(0))),
        ([ir.Const(value(0), 1), ir.Return()], (value(0),)),
        ([ir.Const(value(0), 1), ir.Const(value(0), 2), ir.Return()], ()),
    ],
)
def test_rejects_duplicate_definitions(
    operations: list[ir.Op], arguments: tuple[ir.Value, ...]
) -> None:
    with pytest.raises(
        ir.VerificationError, match="value %0 is defined more than once"
    ):
        ir.verify(module_with(operations, arguments))


@pytest.mark.parametrize(
    "operations",
    [
        [ir.Return((value(0),))],
        [ir.UnknownOp("use", (value(0),), ()), ir.Const(value(0), 1), ir.Return()],
        [ir.UnknownOp("self", (value(0),), (value(0),)), ir.Return()],
    ],
)
def test_rejects_undefined_and_forward_uses(operations: list[ir.Op]) -> None:
    with pytest.raises(ir.VerificationError, match="value %0 is not visible"):
        ir.verify(module_with(operations))


def test_checks_use_type_against_definition() -> None:
    module = module_with(
        [ir.Return((value(0, ir.ScalarType.F32),))], (value(0),), (ir.ScalarType.F32,)
    )
    with pytest.raises(
        ir.VerificationError, match="type does not match its definition"
    ):
        ir.verify(module)


@pytest.mark.parametrize("source", ["sibling", "if_result", "future", "outside"])
def test_nested_values_obey_scope(source: str) -> None:
    module, conditional = conditional_module()
    block = module.functions[0].first_block
    assert block is not None
    if source == "outside":
        block.operations[-1] = ir.Return((value(3),))
    else:
        captured = {"sibling": value(3), "if_result": value(2), "future": value(4)}[
            source
        ]
        conditional.else_region.blocks[0].operations = [ir.Yield((captured,))]
        if source == "future":
            block.operations.insert(1, ir.Const(captured, 1))
    with pytest.raises(ir.VerificationError, match="is not visible"):
        ir.verify(module)


@pytest.mark.parametrize("id", [1, 2, 3])
def test_value_ids_are_unique_across_nested_regions(id: int) -> None:
    module, conditional = conditional_module()
    conditional.else_region.blocks[0].operations = [
        ir.Const(value(id), 2),
        ir.Yield((value(id),)),
    ]
    with pytest.raises(
        ir.VerificationError, match=f"value %{id} is defined more than once"
    ):
        ir.verify(module)


def test_reports_location_of_invalid_yield() -> None:
    module, conditional = conditional_module()
    conditional.else_region.blocks[0].operations = [ir.Yield()]
    with pytest.raises(
        ir.VerificationError,
        match=r"function 'main': operation 0 \(If\): else region: operation 0 \(Yield\): Yield operand types",
    ):
        ir.verify(module)


@pytest.mark.parametrize("kind", ["region", "block", "cycle"])
def test_rejects_shared_or_cyclic_structure(kind: str) -> None:
    module, conditional = conditional_module()
    if kind == "region":
        replacement = replace(conditional, else_region=conditional.then_region)
        block = module.functions[0].first_block
        assert block is not None
        block.operations[0] = replacement
    elif kind == "block":
        conditional.else_region.blocks = conditional.then_region.blocks
    else:
        conditional.then_region.blocks[0].operations.insert(0, conditional)
    with pytest.raises(ir.VerificationError, match="multiple owners|cycle"):
        ir.verify(module)


@pytest.mark.parametrize(
    "operation, error",
    [
        (ir.Call("missing", (), ()), "must name a function"),
        (ir.Call("weight", (), ()), "must name a function"),
        (ir.Call("callee", (), (value(1),)), "call argument types"),
        (
            ir.Call("callee", (value(0),), (value(1, ir.ScalarType.F32),)),
            "call result types",
        ),
        (ir.GetGlobal("missing", value(1)), "must name a global"),
        (ir.GetGlobal("callee", value(1)), "must name a global"),
        (ir.GetGlobal("weight", value(1, ir.ScalarType.F32)), "result type"),
    ],
)
def test_resolves_symbols_and_checks_reference_types(
    operation: ir.Op, error: str
) -> None:
    module = module_with([operation, ir.Return()], (value(0),))
    module.functions.append(
        ir.Function(
            "callee", ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,))
        )
    )
    module.globals.append(ir.Global("weight", ir.ScalarType.I32, 1))
    with pytest.raises(ir.VerificationError, match=error):
        ir.verify(module)


def test_checks_unknown_operation_attributes() -> None:
    operation = ir.UnknownOp(
        "custom.op", (), (), {"nested": (1, "value", (None, True))}
    )
    ir.verify_op(operation)
    operation.attributes["nested"] = cast(ir.AttributeValue, [1])
    with pytest.raises(ir.VerificationError, match="attribute 'nested'"):
        ir.verify(module_with([operation, ir.Return()]))


def test_local_checks_allow_unfinished_regions_but_module_check_rejects_them() -> None:
    conditional = ir.If((), value(0, ir.ScalarType.BOOL), ir.Region(), ir.Region())
    ir.verify_op(conditional)
    with pytest.raises(ir.VerificationError, match="exactly one block"):
        ir.verify(module_with([conditional, ir.Return()], (conditional.condition,)))
