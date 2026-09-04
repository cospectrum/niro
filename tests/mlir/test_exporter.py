import pytest
from xdsl.dialects import builtin, ml_program

from niro import ir
from niro.builder import ModuleBuilder
from niro.mlir import export_mlir, format_mlir


def test_lowers_tensor_add() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    module = ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type, tensor_type), (tensor_type,)),
    )
    block = function.region().block((tensor_type, tensor_type))
    result = block.add(*block.inner.arguments)
    block.return_(result)

    text = format_mlir(export_mlir(module.inner))

    assert "func.func @model" in text
    assert "%2 = arith.addf %0, %1 : tensor<2x2xf32>" in text
    assert "func.return %2 : tensor<2x2xf32>" in text


def test_lowers_tensor_weight_to_private_immutable_global() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    data = bytes(range(16))
    module = ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type,), (tensor_type,)),
    )
    block = function.region().block((tensor_type,))
    weight = block.tensor(data, tensor_type)
    result = block.matmul(block.inner.arguments[0], weight)
    block.return_(result)

    lowered = export_mlir(module.inner)

    operations = list(lowered.body.block.ops)
    global_ = operations[0]
    assert isinstance(global_, ml_program.GlobalOp)
    assert isinstance(global_.value, builtin.DenseResourceAttr)
    text = format_mlir(lowered)
    assert "ml_program.global private @__niro_model_1" in text
    assert "dense_resource<__niro_model_1>" in text
    assert f'__niro_model_1: "0x{data.hex().upper()}"' in text
    assert "ml_program.global_load_const @__niro_model_1" in text
    assert text.index("linalg.fill") < text.index("linalg.matmul")


def test_lowers_private_helper_and_call() -> None:
    module = ModuleBuilder()
    helper = module.function(
        name="helper",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
    )
    helper_block = helper.region().block((ir.ScalarType.I32,))
    helper_block.return_(helper_block.inner.arguments[0])
    main = module.function(
        name="model",
        type=ir.FunctionType((ir.ScalarType.I32,), (ir.ScalarType.I32,)),
    )
    main_block = main.region().block((ir.ScalarType.I32,))
    (result,) = main_block.call(helper, main_block.inner.arguments)
    main_block.return_(result)

    text = format_mlir(export_mlir(module.inner))

    assert "func.func @helper" in text
    assert "func.func @model" in text
    assert "func.call @helper(%0) : (i32) -> i32" in text


def test_lowers_static_transpose() -> None:
    input_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 3))
    output_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(3, 2))
    module = ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((input_type,), (output_type,)),
    )
    block = function.region().block((input_type,))
    result = block.transpose(block.inner.arguments[0], [1, 0])
    block.return_(result)

    text = format_mlir(export_mlir(module.inner))

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
                            else_region=branch,
                        ),
                        ir.Return(operands=(result,)),
                    ],
                )
            ]
        ),
    )
    module = ir.Module(functions=[function])

    text = format_mlir(export_mlir(module))

    assert "scf.if %0 -> (i1)" in text
    assert text.count("scf.yield %0 : i1") == 2
    assert "func.return %1 : i1" in text


def test_preserves_metadata_with_niro_namespace() -> None:
    module = ModuleBuilder()
    function = module.function(name="model", type=ir.FunctionType((), ()))
    function.inner.attributes["note"] = "function"
    function.region().block().return_()
    module.inner.attributes["version"] = 1

    text = format_mlir(export_mlir(module.inner))

    assert "niro.version = 1 : i64" in text
    assert 'niro.note = "function"' in text


def test_rejects_unknown_operation() -> None:
    module = ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((ir.ScalarType.F32,), (ir.ScalarType.F32,)),
    )
    block = function.region().block((ir.ScalarType.F32,))
    (result,) = block.unknown_op(
        name="onnx.Relu",
        operands=block.inner.arguments,
        result_types=[ir.ScalarType.F32],
    )
    block.return_(result)

    with pytest.raises(
        NotImplementedError,
        match="cannot lower unknown operation to MLIR: onnx.Relu",
    ):
        export_mlir(module.inner)


def test_rejects_dynamic_matmul() -> None:
    tensor_type = ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(None, 2),
    )
    module = ModuleBuilder()
    function = module.function(
        name="model",
        type=ir.FunctionType((tensor_type, tensor_type), (tensor_type,)),
    )
    block = function.region().block((tensor_type, tensor_type))
    result = block.matmul(
        block.inner.arguments[0],
        block.inner.arguments[1],
    )
    block.return_(result)

    with pytest.raises(
        NotImplementedError,
        match="matmul requires a static ranked tensor",
    ):
        export_mlir(module.inner)
