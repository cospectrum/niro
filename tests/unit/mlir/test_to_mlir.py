import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, cf, func, ml_program, scf
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.parser import Parser

import niro
from niro import builder, ir, verify


def test_lowers_tensor_add() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    module = builder.ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type, tensor_type), (tensor_type,)),
    )
    block = function.region().block((tensor_type, tensor_type))
    result = block.add(*block.raw.arguments)
    block.return_(result)

    text = niro.format_mlir(niro.to_mlir(module.verify()))

    assert "func.func @model" in text
    assert "%2 = arith.addf %0, %1 : tensor<2x2xf32>" in text
    assert "func.return %2 : tensor<2x2xf32>" in text


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_lowers_tensor_extract(shape: tuple[int, ...]) -> None:
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

    text = niro.format_mlir(niro.to_mlir(module.verify()))

    assert "tensor.extract" in text
    assert text.count("arith.index_cast") == len(shape)


def test_lowers_tensor_weight_to_arithmetic_constant() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    data = bytes(range(16))
    module = builder.ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type,), (tensor_type,)),
    )
    block = function.region().block((tensor_type,))
    weight = block.tensor(data, tensor_type)
    result = block.matmul(block.raw.arguments[0], weight)
    block.return_(result)

    lowered = niro.to_mlir(module.verify())

    assert all(isinstance(op, func.FuncOp) for op in lowered.body.block.ops)
    constants = [op for op in lowered.walk() if isinstance(op, arith.ConstantOp)]
    assert isinstance(constants[0].value, builtin.DenseResourceAttr)
    text = niro.format_mlir(lowered)
    assert "arith.constant dense_resource<__niro_model_1>" in text
    assert f'__niro_model_1: "0x{data.hex().upper()}"' in text
    assert "ml_program.global" not in text
    assert text.index("linalg.fill") < text.index("linalg.matmul")


@pytest.mark.parametrize("nested", [False, True])
def test_preserves_constants_and_explicit_globals(nested: bool) -> None:
    tensor_type = ir.TensorType(ir.ScalarType.I32, (2,))
    data = bytes(range(8))
    module = builder.ModuleBuilder()
    module.raw.globals.append(ir.Global("weight", tensor_type, data))
    block = (
        module.function(
            name="constants",
            type=ir.FunctionType((), (tensor_type, tensor_type)),
        )
        .region()
        .first_block()
    )
    global_value = block.get_global("weight")
    if nested:
        conditional = block.if_(block.bool(True), (tensor_type,))
        for region in (conditional.then_region, conditional.else_region):
            arm = region.block()
            arm.yield_(arm.tensor(data, tensor_type))
        constant = conditional.raw.results[0]
    else:
        constant = block.tensor(data, tensor_type)
    block.return_(constant, global_value)

    lowered = niro.to_mlir(module.verify())
    ops = list(lowered.walk())
    assert sum(isinstance(op, ml_program.GlobalOp) for op in ops) == 1
    assert sum(isinstance(op, ml_program.GlobalLoadConstantOp) for op in ops) == 1
    assert sum(isinstance(op, arith.ConstantOp) for op in ops) == (3 if nested else 1)
    parse_mlir(niro.format_mlir(lowered)).verify()


def test_lowers_private_helper_and_call() -> None:
    module = builder.ModuleBuilder()
    helper = module.function(
        name="helper",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
    )
    helper_block = helper.region().block((ir.ScalarType.I32,))
    helper_block.return_(helper_block.raw.arguments[0])
    main = module.function(
        name="model",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
    )
    main_block = main.region().block((ir.ScalarType.I32,))
    (result,) = main_block.call(helper, main_block.raw.arguments)
    main_block.return_(result)

    text = niro.format_mlir(niro.to_mlir(module.verify()))

    assert "func.func @helper" in text
    assert "func.func @model" in text
    assert "func.call @helper(%0) : (i32) -> i32" in text


def test_lowers_static_transpose() -> None:
    input_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 3))
    output_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(3, 2))
    module = builder.ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((input_type,), (output_type,)),
    )
    block = function.region().block((input_type,))
    result = block.transpose(block.raw.arguments[0], [1, 0])
    block.return_(result)

    text = niro.format_mlir(niro.to_mlir(module.verify()))

    assert "%1 = tensor.empty() : tensor<3x2xf32>" in text
    assert "linalg.transpose" in text
    assert "permutation = [1, 0]" in text


def test_lowers_if_and_yield() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    branch = ir.Region([ir.Block(operations=[ir.Yield(operands=(condition,))])])
    function = ir.Function(
        name="model",
        type=ir.FunctionType(
            inputs=(ir.ScalarType.BOOL,),
            outputs=(ir.ScalarType.BOOL,),
        ),
        body=ir.Region(
            [
                ir.Block(
                    arguments=(condition,),
                    operations=[
                        ir.If(
                            results=(result,),
                            condition=condition,
                            then_region=branch,
                            else_region=ir.Region(
                                [ir.Block(operations=[ir.Yield((condition,))])]
                            ),
                        ),
                        ir.Return(operands=(result,)),
                    ],
                )
            ]
        ),
    )
    module = ir.Module(functions=[function])

    text = niro.format_mlir(niro.to_mlir(verify.module(module)))

    assert "scf.if %0 -> (i1)" in text
    assert text.count("scf.yield %0 : i1") == 2
    assert "func.return %1 : i1" in text


def test_preserves_metadata_with_niro_namespace() -> None:
    module = builder.ModuleBuilder()
    function = module.function(name="model", type=ir.FunctionType((), ()))
    function.raw.attributes["note"] = "function"
    function.region().block().return_()
    module.raw.attributes["version"] = 1

    text = niro.format_mlir(niro.to_mlir(module.verify()))

    assert "niro.version = 1 : i64" in text
    assert 'niro.note = "function"' in text


def test_rejects_unknown_operation() -> None:
    module = builder.ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((ir.ScalarType.F32,), (ir.ScalarType.F32,)),
    )
    block = function.region().block((ir.ScalarType.F32,))
    (result,) = block.unknown_op(
        name="onnx.Relu",
        operands=block.raw.arguments,
        result_types=[ir.ScalarType.F32],
    )
    block.return_(result)

    with pytest.raises(
        NotImplementedError,
        match="cannot lower unknown operation to MLIR: onnx.Relu",
    ):
        niro.to_mlir(module.verify())


def test_rejects_dynamic_matmul() -> None:
    tensor_type = ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(None, 2),
    )
    module = builder.ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type, tensor_type), (tensor_type,)),
    )
    block = function.region().block((tensor_type, tensor_type))
    result = block.matmul(
        block.raw.arguments[0],
        block.raw.arguments[1],
    )
    block.return_(result)

    with pytest.raises(
        NotImplementedError,
        match="matmul requires a static ranked tensor",
    ):
        niro.to_mlir(module.verify())


def test_lowers_nonreturning_function_cfg() -> None:
    flag = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    loop = ir.Block()
    loop.operations = [ir.CondBranch(flag, loop, loop)]
    body = ir.Region([ir.Block((flag,), [ir.Branch(loop)]), loop])
    fn = ir.Function("f", ir.FunctionType((flag.type,), ()), body)
    module = verify.module(ir.Module(functions=[fn]))
    text = niro.format_mlir(niro.to_mlir(module))
    assert "cf.br" in text and "cf.cond_br" in text
    parse_mlir(text).verify()


def parse_mlir(text: str) -> builtin.ModuleOp:
    context = Context()
    for dialect in (
        builtin.Builtin,
        arith.Arith,
        cf.Cf,
        func.Func,
        scf.Scf,
        ml_program.MLProgram,
    ):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("flag", [False, True])
def test_builder_lowering_pipeline_preserves_loop_results(
    nested: bool, flag: bool
) -> None:
    module = builder.ModuleBuilder()
    signature = ir.FunctionType(
        (ir.ScalarType.BOOL, ir.ScalarType.I32), (ir.ScalarType.I32,)
    )
    helper = module.function(name="helper", type=signature)
    region = helper.region()
    entry = region.first_block()
    condition, x = entry.raw.arguments
    start = entry
    # Layout deliberately puts the exit and loop before the base's definition.
    exit_ = region.block((x.type,))
    loop = region.block((condition.type, x.type))
    work = region.block()
    start.branch(work)
    base = work.add(x, x)
    work.branch(loop, condition, x)
    repeat, carried = loop.raw.arguments
    total = loop.add(carried, base)
    stop = loop.bool(False)
    loop.cond_branch(repeat, loop, exit_, (stop, total), (total,))
    exit_.return_(*exit_.raw.arguments)
    caller = module.function(name="caller", type=signature).region().first_block()
    if nested:
        flag_value, operand = caller.raw.arguments
        conditional = caller.if_(flag_value, (operand.type,))
        then = conditional.then_region.block()
        then.yield_(*then.call(helper, caller.raw.arguments))
        conditional.else_region.block().yield_(operand)
        caller.return_(*conditional.raw.results)
    else:
        caller.return_(*caller.call(helper, caller.raw.arguments))
    original = module.verify()
    expected = 7 if nested and not flag else (35 if flag else 21)
    lowered = niro.to_mlir(original)
    assert not any(isinstance(op, scf.ExecuteRegionOp) for op in lowered.walk())
    parsed = parse_mlir(niro.format_mlir(lowered))
    parsed.verify()
    interpreter = Interpreter(parsed)
    for implementations in (
        ArithFunctions(),
        CfFunctions(),
        FuncFunctions(),
        ScfFunctions(),
    ):
        interpreter.register_implementations(implementations)
    assert interpreter.call_op("caller", (flag, 7)) == (expected,)
