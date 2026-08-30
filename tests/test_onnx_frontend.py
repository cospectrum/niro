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
    add, return_ = function.body.blocks[0].operations
    assert isinstance(add, ir.Add)
    assert isinstance(return_, ir.Return)
    assert [value.id for value in function.arguments] == [
        ir.ValueId(0),
        ir.ValueId(1),
    ]
    assert add.lhs is function.arguments[0]
    assert add.rhs is function.arguments[1]
    assert add.result.id == ir.ValueId(2)
    assert return_.operands == (add.result,)
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
    constant, multiply, return_ = function.body.blocks[0].operations
    assert isinstance(constant, ir.Const)
    assert isinstance(multiply, ir.Mul)
    assert isinstance(return_, ir.Return)
    assert constant == ir.Const(
        result=ir.Value(
            id=ir.ValueId(1),
            type=ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,)),
        ),
        value=struct.pack("<2f", 2.0, 3.0),
    )
    assert multiply.lhs is function.arguments[0]
    assert multiply.rhs is constant.result
    assert multiply.result.id == ir.ValueId(2)
    assert return_.operands == (multiply.result,)


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
    transpose, matmul, return_ = function.body.blocks[0].operations
    assert isinstance(transpose, ir.Transpose)
    assert isinstance(matmul, ir.MatMul)
    assert isinstance(return_, ir.Return)
    assert [value.id for value in function.arguments] == [
        ir.ValueId(0),
        ir.ValueId(1),
    ]
    assert transpose.operand is function.arguments[1]
    assert transpose.permutation == (1, 0)
    assert transpose.result.id == ir.ValueId(2)
    assert matmul.lhs is function.arguments[0]
    assert matmul.rhs is transpose.result
    assert matmul.result.id == ir.ValueId(3)
    assert matmul.result.type == ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(2, 4),
    )
    assert return_.operands == (matmul.result,)


def test_imports_unsupported_operation_as_unknown() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="LeakyRelu",
                inputs=["x"],
                outputs=["result"],
                alpha=0.2,
            )
        ],
        name="relu",
        inputs=[tensor("x", [2])],
        outputs=[tensor("result", [2])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    operation = function.body.blocks[0].operations[0]
    assert isinstance(operation, ir.UnknownOp)
    assert operation.name == "onnx.LeakyRelu"
    assert operation.operands == function.arguments
    assert operation.attributes == {"alpha": pytest.approx(0.2)}
    assert operation.results[0].type == ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(2,),
    )
    assert operation.results[0].id == ir.ValueId(1)
    return_ = function.body.blocks[0].operations[1]
    assert isinstance(return_, ir.Return)
    assert return_.operands == operation.results


def test_preserves_node_and_graph_output_order() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type="Add",
                inputs=["lhs", "rhs"],
                outputs=["sum"],
            ),
            helper.make_node(
                op_type="Mul",
                inputs=["lhs", "rhs"],
                outputs=["product"],
            ),
        ],
        name="sum_and_product",
        inputs=[tensor("lhs", [2]), tensor("rhs", [2])],
        outputs=[tensor("product", [2]), tensor("sum", [2])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    add, multiply, return_ = function.body.blocks[0].operations
    assert isinstance(add, ir.Add)
    assert isinstance(multiply, ir.Mul)
    assert isinstance(return_, ir.Return)
    assert add.result.id == ir.ValueId(2)
    assert multiply.result.id == ir.ValueId(3)
    assert return_.operands == (multiply.result, add.result)
