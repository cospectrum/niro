from dataclasses import dataclass

import onnx
from onnx import TensorProto, helper


@dataclass(frozen=True)
class ModelCase:
    name: str
    model: onnx.ModelProto
    signature: str
    mlir_fragments: tuple[str, ...]


def onnx_model_cases() -> tuple[ModelCase, ...]:
    return _add_case(), _linear_case()


def _add_case() -> ModelCase:
    lhs = _tensor("lhs", [2, 2])
    rhs = _tensor("rhs", [2, 2])
    result = _tensor("result", [2, 2])
    node = helper.make_node("Add", ["lhs", "rhs"], ["result"])
    graph = helper.make_graph([node], "add", [lhs, rhs], [result])
    return ModelCase(
        name="add",
        model=helper.make_model(graph),
        signature=(
            "add(lhs: tensor<2x2xf32>, rhs: tensor<2x2xf32>) "
            "-> (result: tensor<2x2xf32>)\n"
        ),
        mlir_fragments=("func.func @add", "arith.addf", "func.return"),
    )


def _linear_case() -> ModelCase:
    lhs = _tensor("lhs", [2, 3])
    weight_transposed = _tensor("weight_transposed", [3, 4])
    result = _tensor("result", [2, 4])
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [4, 3],
        [float(value) for value in range(12)],
    )
    transpose = helper.make_node(
        "Transpose",
        ["weight"],
        ["weight_transposed"],
        perm=[1, 0],
    )
    matmul = helper.make_node(
        "MatMul",
        ["lhs", "weight_transposed"],
        ["result"],
    )
    graph = helper.make_graph(
        [transpose, matmul],
        "linear",
        [lhs],
        [result],
        initializer=[weight],
        value_info=[weight_transposed],
    )
    return ModelCase(
        name="linear",
        model=helper.make_model(graph),
        signature="linear(lhs: tensor<2x3xf32>) -> (result: tensor<2x4xf32>)\n",
        mlir_fragments=(
            "ml_program.global private @weight",
            "dense_resource<weight>",
            "func.func @linear",
            "linalg.transpose",
            "linalg.matmul",
        ),
    )


def _tensor(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
