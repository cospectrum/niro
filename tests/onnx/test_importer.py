import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from niro import ir
from niro.onnx import OnnxOpType, import_onnx
from niro.onnx.importer import _ONNX_DOMAINS, node_name


def onnx_tensor(
    name: str,
    shape: list[int],
    element_type: int = TensorProto.FLOAT,
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name=name,
        elem_type=element_type,
        shape=shape,
    )


@dataclass(frozen=True)
class OneNodeCase:
    node: onnx.NodeProto
    inputs: tuple[onnx.ValueInfoProto, ...]
    outputs: tuple[onnx.ValueInfoProto, ...]
    expected_op: type[ir.Op]


def one_node_case(
    *,
    onnx_op_type: OnnxOpType | str,
    input_shapes: Sequence[Sequence[int]],
    output_shapes: Sequence[Sequence[int]],
    expected_op: type[ir.Op],
    domain: str = "",
    input_types: Sequence[int] | None = None,
    output_types: Sequence[int] | None = None,
    **attributes: Any,
) -> OneNodeCase:
    input_names = tuple(f"x{index}" for index in range(len(input_shapes)))
    output_names = (
        ("result",)
        if len(output_shapes) == 1
        else tuple(f"result{index}" for index in range(len(output_shapes)))
    )
    if input_types is None:
        input_types = (TensorProto.FLOAT,) * len(input_shapes)
    if output_types is None:
        output_types = (TensorProto.FLOAT,) * len(output_shapes)
    return OneNodeCase(
        node=helper.make_node(
            onnx_op_type,
            inputs=input_names,
            outputs=output_names,
            domain=domain,
            **attributes,
        ),
        inputs=tuple(
            onnx_tensor(name, list(shape), element_type)
            for name, shape, element_type in zip(
                input_names, input_shapes, input_types, strict=True
            )
        ),
        outputs=tuple(
            onnx_tensor(name, list(shape), element_type)
            for name, shape, element_type in zip(
                output_names, output_shapes, output_types, strict=True
            )
        ),
        expected_op=expected_op,
    )


@pytest.mark.parametrize(
    "case",
    [
        one_node_case(
            onnx_op_type=OnnxOpType.Add,
            input_shapes=[(2,), (2,)],
            output_shapes=[(2,)],
            expected_op=ir.Add,
        ),
        one_node_case(
            onnx_op_type=OnnxOpType.Mul,
            input_shapes=[(2,), (2,)],
            output_shapes=[(2,)],
            expected_op=ir.Mul,
        ),
        one_node_case(
            onnx_op_type=OnnxOpType.MatMul,
            input_shapes=[(2, 3), (3, 4)],
            output_shapes=[(2, 4)],
            expected_op=ir.MatMul,
        ),
        one_node_case(
            onnx_op_type=OnnxOpType.Transpose,
            input_shapes=[(2, 3)],
            output_shapes=[(3, 2)],
            expected_op=ir.Transpose,
            perm=[1, 0],
        ),
        one_node_case(
            onnx_op_type=OnnxOpType.LeakyRelu,
            input_shapes=[(2,)],
            output_shapes=[(2,)],
            expected_op=ir.UnknownOp,
            alpha=0.2,
        ),
        one_node_case(
            onnx_op_type="Custom",
            input_shapes=[(2,)],
            output_shapes=[(2,)],
            expected_op=ir.UnknownOp,
            domain="example",
        ),
        one_node_case(
            onnx_op_type=OnnxOpType.TopK,
            input_shapes=[(5,), ()],
            output_shapes=[(3,), (3,)],
            expected_op=ir.UnknownOp,
            input_types=(TensorProto.FLOAT, TensorProto.INT64),
            output_types=(TensorProto.FLOAT, TensorProto.INT64),
            axis=0,
        ),
    ],
)
def test_imports_single_node_graph(case: OneNodeCase) -> None:
    graph = helper.make_graph(
        nodes=[case.node],
        name="model",
        inputs=list(case.inputs),
        outputs=list(case.outputs),
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    (block,) = function.body.blocks
    operation, return_ = block.operations
    assert isinstance(operation, case.expected_op)
    assert isinstance(return_, ir.Return)
    assert function.name == "model"
    assert function.input_names == tuple(value.name for value in case.inputs)
    assert function.output_names == tuple(value.name for value in case.outputs)
    assert [value.id for value in block.arguments] == [
        ir.ValueId(index) for index in range(len(case.inputs))
    ]
    values = dict(
        zip(
            (value.name for value in case.inputs),
            block.arguments,
            strict=True,
        )
    )
    results = assert_imported_node(
        case.node,
        operation,
        values,
    )
    assert [result.id for result in results] == [
        ir.ValueId(len(case.inputs) + index) for index in range(len(case.outputs))
    ]
    assert tuple(result.type for result in results) == function.type.outputs
    assert return_.operands == results
    assert module.attributes == {}


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
                op_type=OnnxOpType.Mul,
                inputs=["x", "weight"],
                outputs=["result"],
            )
        ],
        name="scale",
        inputs=[onnx_tensor("x", [2])],
        outputs=[onnx_tensor("result", [2])],
        initializer=[weight],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    (global_,) = module.globals
    (block,) = function.body.blocks
    get_global, multiply, return_ = block.operations
    assert isinstance(get_global, ir.GetGlobal)
    assert isinstance(multiply, ir.Mul)
    assert isinstance(return_, ir.Return)
    tensor_type = ir.TensorType(element_type=ir.ScalarType.F32, shape=(2,))
    assert global_ == ir.Global(
        name="weight",
        type=tensor_type,
        initializer=struct.pack("<2f", 2.0, 3.0),
    )
    assert get_global == ir.GetGlobal(
        name="weight",
        result=ir.Value(id=ir.ValueId(1), type=tensor_type),
    )
    (result,) = assert_imported_node(
        graph.node[0],
        multiply,
        {"x": block.arguments[0], "weight": get_global.result},
    )
    assert result.id == ir.ValueId(2)
    assert return_.operands == (result,)


def test_imports_matmul_and_transpose() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type=OnnxOpType.Transpose,
                inputs=["rhs"],
                outputs=["rhs_t"],
                perm=[1, 0],
            ),
            helper.make_node(
                op_type=OnnxOpType.MatMul,
                inputs=["lhs", "rhs_t"],
                outputs=["result"],
            ),
        ],
        name="linear",
        inputs=[onnx_tensor("lhs", [2, 3]), onnx_tensor("rhs", [4, 3])],
        outputs=[onnx_tensor("result", [2, 4])],
        value_info=[onnx_tensor("rhs_t", [3, 4])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    (block,) = function.body.blocks
    transpose, matmul, return_ = block.operations
    assert isinstance(transpose, ir.Transpose)
    assert isinstance(matmul, ir.MatMul)
    assert isinstance(return_, ir.Return)
    assert [value.id for value in block.arguments] == [
        ir.ValueId(0),
        ir.ValueId(1),
    ]
    values = dict(zip(("lhs", "rhs"), block.arguments, strict=True))
    (transpose_result,) = assert_imported_node(graph.node[0], transpose, values)
    (matmul_result,) = assert_imported_node(graph.node[1], matmul, values)
    assert transpose_result.id == ir.ValueId(2)
    assert matmul_result.id == ir.ValueId(3)
    assert matmul_result.type == ir.TensorType(
        element_type=ir.ScalarType.F32,
        shape=(2, 4),
    )
    assert return_.operands == (matmul_result,)


def test_preserves_node_and_graph_output_order() -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                op_type=OnnxOpType.Add,
                inputs=["lhs", "rhs"],
                outputs=["sum"],
            ),
            helper.make_node(
                op_type=OnnxOpType.Mul,
                inputs=["lhs", "rhs"],
                outputs=["product"],
            ),
        ],
        name="sum_and_product",
        inputs=[onnx_tensor("lhs", [2]), onnx_tensor("rhs", [2])],
        outputs=[onnx_tensor("product", [2]), onnx_tensor("sum", [2])],
    )

    module = import_onnx(helper.make_model(graph=graph))

    function = module.functions[0]
    assert function.body is not None
    (block,) = function.body.blocks
    add, multiply, return_ = block.operations
    assert isinstance(add, ir.Add)
    assert isinstance(multiply, ir.Mul)
    assert isinstance(return_, ir.Return)
    values = dict(zip(("lhs", "rhs"), block.arguments, strict=True))
    (sum_,) = assert_imported_node(graph.node[0], add, values)
    (product,) = assert_imported_node(graph.node[1], multiply, values)
    assert sum_.id == ir.ValueId(2)
    assert product.id == ir.ValueId(3)
    assert return_.operands == (product, sum_)


def assert_imported_node(
    node: onnx.NodeProto,
    operation: ir.Op,
    values: dict[str, ir.Value],
) -> tuple[ir.Value, ...]:
    """Compare one imported operation with its source node and bind outputs."""
    normalized = as_onnx_unknown_op(operation, node_name(node))
    attributes = {
        attribute.name: _normalize_attribute(onnx.helper.get_attribute_value(attribute))
        for attribute in node.attribute
    }
    assert normalized.name == node_name(node)
    assert normalized.operands == tuple(values[name] for name in node.input)
    assert normalized.attributes == attributes
    assert len(normalized.results) == len(node.output)
    values.update(zip(node.output, normalized.results, strict=True))
    return normalized.results


def as_onnx_unknown_op(operation: ir.Op, name: str) -> ir.UnknownOp:
    """Project a node-backed Niro operation into its generic ONNX form."""
    attributes: dict[str, ir.AttributeValue]
    match operation:
        case ir.Add() | ir.Mul() | ir.MatMul():
            attributes = {}
        case ir.Transpose(permutation=permutation):
            attributes = {"perm": permutation}
        case ir.UnknownOp():
            return operation
        case _:
            raise TypeError(f"operation has no ONNX node representation: {operation!r}")
    return ir.UnknownOp(
        name=name,
        operands=operation.get_operands(),
        results=operation.get_results(),
        attributes=attributes,
    )


def _normalize_attribute(value: object) -> ir.AttributeValue:
    if isinstance(value, (bool, int, float, str, bytes)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_attribute(element) for element in value)
    raise TypeError(f"unsupported test attribute: {value!r}")


def test_domains_match_latest_onnx_schema_registry() -> None:
    registered_domains = {
        schema.domain for schema in onnx.defs.get_all_schemas()
    }

    assert set(_ONNX_DOMAINS) == registered_domains
