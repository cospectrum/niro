"""Write an example linear ONNX model to standard output."""

import sys

from onnx import TensorProto, helper


def main() -> None:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2, 3])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 4])
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
    )
    model = helper.make_model(graph)
    sys.stdout.buffer.write(model.SerializeToString())


if __name__ == "__main__":
    main()
