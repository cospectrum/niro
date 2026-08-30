import struct

import onnx
import pytest
from onnx import TensorProto, helper

from niro import ir
from niro.frontends.onnx import import_onnx


def tensor(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name=name,
        elem_type=TensorProto.FLOAT,
        shape=shape,
    )


def test_imports_add_graph() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="Add",
                inputs=["lhs", "rhs"],
                outputs=["sum"],
            )
        ],
        name="add",
        inputs=[tensor("lhs", [2]), tensor("rhs", [2])],
        outputs=[tensor("sum", [2])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.name == "add"
    assert function.type == ir.FunctionType(
        inputs=(
            ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,)),
            ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,)),
        ),
        outputs=(ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,)),),
    )
    assert function.body is not None
    assert isinstance(function.body.blocks[0].operations[0], ir.Add)
    assert isinstance(function.body.blocks[0].operations[1], ir.Return)
    assert module.attributes == {"entry_point": "add"}


def test_imports_initializer_as_tensor_constant() -> None:
    weight = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[2],
        vals=[2.0, 3.0],
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="Mul",
                inputs=["x", "weight"],
                outputs=["result"],
            )
        ],
        name="scale",
        inputs=[tensor("x", [2])],
        outputs=[tensor("result", [2])],
        initializer=[weight],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    constant, multiply, _ = function.body.blocks[0].operations
    assert constant == ir.Const(
        result=ir.Value(
            id=ir.ValueId(1),
            type=ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,)),
        ),
        value=struct.pack("<2f", 2.0, 3.0),
    )
    assert isinstance(multiply, ir.Mul)


def test_imports_matmul_and_transpose() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="Transpose",
                inputs=["rhs"],
                outputs=["rhs_t"],
                perm=[1, 0],
            ),
            helper.make_node(
                op_type="MatMul",
                inputs=["lhs", "rhs_t"],
                outputs=["result"],
            ),
        ],
        name="linear",
        inputs=[tensor("lhs", [2, 3]), tensor("rhs", [4, 3])],
        outputs=[tensor("result", [2, 4])],
        value_info=[tensor("rhs_t", [3, 4])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    transpose, matmul, _ = function.body.blocks[0].operations
    assert isinstance(transpose, ir.Transpose)
    assert transpose.permutation == (1, 0)
    assert isinstance(matmul, ir.MatMul)
    assert matmul.result.type == ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(2, 4),
    )


def test_rejects_unsupported_operation() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="Relu",
                inputs=["x"],
                outputs=["result"],
            )
        ],
        name="relu",
        inputs=[tensor("x", [2])],
        outputs=[tensor("result", [2])],
    )

    with pytest.raises(
        NotImplementedError,
        match="unsupported ONNX operation: Relu",
    ):
        import_onnx(helper.make_model(graph=graph))
