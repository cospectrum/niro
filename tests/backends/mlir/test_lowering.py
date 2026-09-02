import pytest
from xdsl.dialects import builtin, ml_program

from niro import ir
from niro.backends.mlir import format_mlir, lower_to_mlir
from niro.builder import ModuleBuilder


def test_lowers_public_entry_point_and_tensor_add() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    module = ModuleBuilder()
    function = module.func(
        name="model",
        arg_types=[tensor_type, tensor_type],
        ret_types=[tensor_type],
    )
    result = function.entry.add(*function.args)
    function.entry.return_(result)
    module.set_entry_point(function)

    text = format_mlir(lower_to_mlir(module.ir))

    assert "func.func public @model" in text
    assert "attributes {niro.entry_point}" in text
    assert "%2 = arith.addf %0, %1 : tensor<2x2xf32>" in text
    assert "func.return %2 : tensor<2x2xf32>" in text


def test_lowers_tensor_weight_to_private_immutable_global() -> None:
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 2))
    data = bytes(range(16))
    module = ModuleBuilder()
    function = module.func(
        name="model",
        arg_types=[tensor_type],
        ret_types=[tensor_type],
    )
    weight = function.entry.tensor(data=data, result_type=tensor_type)
    result = function.entry.matmul(function.args[0], weight)
    function.entry.return_(result)
    module.set_entry_point(function)

    lowered = lower_to_mlir(module.ir)

    operations = list(lowered.body.block.ops)
    global_ = operations[0]
    assert isinstance(global_, ml_program.GlobalOp)
    assert isinstance(global_.value, builtin.DenseIntOrFPElementsAttr)
    assert global_.value.data.data == data
    text = format_mlir(lowered)
    assert "ml_program.global private @__niro_model_1" in text
    assert "ml_program.global_load_const @__niro_model_1" in text
    assert text.index("linalg.fill") < text.index("linalg.matmul")


def test_lowers_private_helper_and_call() -> None:
    module = ModuleBuilder()
    helper = module.func(
        name="helper",
        arg_types=[ir.ScalarType.I32],
        ret_types=[ir.ScalarType.I32],
    )
    helper.entry.return_(helper.args[0])
    main = module.func(
        name="model",
        arg_types=[ir.ScalarType.I32],
        ret_types=[ir.ScalarType.I32],
    )
    result = main.entry.call(helper, main.args[0])
    assert isinstance(result, ir.Value)
    main.entry.return_(result)
    module.set_entry_point(main)

    text = format_mlir(lower_to_mlir(module.ir))

    assert "func.func private @helper" in text
    assert "func.func public @model" in text
    assert "func.call @helper(%0) : (i32) -> i32" in text


def test_lowers_static_transpose() -> None:
    input_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, 3))
    output_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(3, 2))
    module = ModuleBuilder()
    function = module.func(
        name="model",
        arg_types=[input_type],
        ret_types=[output_type],
    )
    result = function.entry.transpose(function.args[0], [1, 0])
    function.entry.return_(result)
    module.set_entry_point(function)

    text = format_mlir(lower_to_mlir(module.ir))

    assert "%1 = tensor.empty() : tensor<3x2xf32>" in text
    assert "linalg.transpose" in text
    assert "permutation = [1, 0]" in text


def test_lowers_if_and_yield() -> None:
    condition = ir.Value(ir.ValueId(0), ir.ScalarType.BOOL)
    result = ir.Value(ir.ValueId(1), ir.ScalarType.BOOL)
    branch = ir.Region(
        [ir.Block(operations=[ir.Yield(id=ir.OpId(0), operands=(condition,))])]
    )
    function = ir.Function(
        id=ir.FuncId(0),
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
                            id=ir.OpId(1),
                            results=(result,),
                            condition=condition,
                            then_region=branch,
                            else_region=branch,
                        ),
                        ir.Return(id=ir.OpId(2), operands=(result,)),
                    ],
                )
            ]
        ),
    )
    module = ir.Module(
        functions=[function],
        attributes={"entry_point": "model"},
    )

    text = format_mlir(lower_to_mlir(module))

    assert "scf.if %0 -> (i1)" in text
    assert text.count("scf.yield %0 : i1") == 2
    assert "func.return %1 : i1" in text


def test_preserves_metadata_with_niro_namespace() -> None:
    module = ModuleBuilder()
    function = module.func(name="model")
    function.function.attributes["note"] = "function"
    function.entry.return_()
    module.ir.attributes["version"] = 1
    module.set_entry_point(function)

    text = format_mlir(lower_to_mlir(module.ir))

    assert "niro.version = 1 : i64" in text
    assert 'niro.note = "function"' in text


def test_rejects_unknown_operation() -> None:
    module = ModuleBuilder()
    function = module.func(
        name="model",
        arg_types=[ir.ScalarType.F32],
        ret_types=[ir.ScalarType.F32],
    )
    (result,) = function.entry.unknown(
        name="onnx.Relu",
        operands=function.args,
        result_types=[ir.ScalarType.F32],
    )
    function.entry.return_(result)
    module.set_entry_point(function)

    with pytest.raises(
        NotImplementedError,
        match="cannot lower unknown operation to MLIR: onnx.Relu",
    ):
        lower_to_mlir(module.ir)


def test_rejects_dynamic_matmul() -> None:
    tensor_type = ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(None, 2),
    )
    module = ModuleBuilder()
    function = module.func(
        name="model",
        arg_types=[tensor_type, tensor_type],
        ret_types=[tensor_type],
    )
    result = function.entry.matmul(
        function.args[0],
        function.args[1],
    )
    function.entry.return_(result)
    module.set_entry_point(function)

    with pytest.raises(
        NotImplementedError,
        match="matmul requires a static ranked tensor",
    ):
        lower_to_mlir(module.ir)
