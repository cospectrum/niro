"""Import valid ONNX graphs and preserve their typed dataflow and attributes."""

import dataclasses

import hypothesis
import numpy as np
import onnx
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

import niro
from niro import ir, verify

from ..strategies import onnx as onnx_st

# ONNX op name -> Niro op type, used only for comparison in tests.
_SUPPORTED_OP_TYPES: dict[str, type[ir.Op]] = {
    "Add": ir.Add,
    "Mul": ir.Mul,
    "MatMul": ir.MatMul,
    "Transpose": ir.Transpose,
    "If": ir.If,
}


@pytest.mark.parametrize("unknown_only", [False, True], ids=["mixed", "unknown"])
@hypothesis.given(data=st.data())
def test_import_preserves_graph(unknown_only: bool, data: st.DataObject) -> None:
    operators = _unknown_operators()
    if not unknown_only:
        operators = _native_operators() + operators
    model = data.draw(onnx_st.models(operators=operators, min_nodes=1))
    onnx.checker.check_model(model, full_check=True)
    original = model.SerializeToString()

    module = niro.from_onnx(model)

    assert model.SerializeToString() == original
    assert verify.module(module) is module
    _assert_import_matches_graph(module, model.graph)
    for node in model.graph.node:
        hypothesis.event(f"operator={node.op_type}")


@hypothesis.given(data=st.data())
def test_import_preserves_if_graph(data: st.DataObject) -> None:
    model = data.draw(
        onnx_st.models(
            operators=(onnx_st.OPERATORS.if_,),
            min_nodes=1,
            max_initializers=3,
            max_seed_initializers=0,
        )
    )
    onnx.checker.check_model(model, full_check=True)
    original = model.SerializeToString()
    module = niro.from_onnx(model)
    assert model.SerializeToString() == original
    assert verify.module(module) is module
    _assert_import_matches_graph(module, model.graph)


def _assert_import_matches_graph(module: ir.Module, graph: onnx.GraphProto) -> None:
    """Check signatures, initializer contents, typed dataflow, attributes, and returns.

    The graph must use concrete tensor types, with value_info for node outputs,
    and the native or opaque operators supported by the importer property tests.
    """
    (function,) = module.functions
    assert function.name == graph.name
    assert function.input_names == tuple(value.name for value in graph.input)
    assert function.output_names == tuple(value.name for value in graph.output)
    assert function.type.inputs == tuple(
        _tensor_type(value.type) for value in graph.input
    )
    assert function.type.outputs == tuple(
        _tensor_type(value.type) for value in graph.output
    )
    assert function.body is not None
    (block,) = function.body.blocks
    values = dict(
        zip((value.name for value in graph.input), block.arguments, strict=True)
    )
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    assert {global_.name for global_ in module.globals} == set(initializers)
    for global_ in module.globals:
        tensor = initializers[global_.name]
        assert global_.type == _tensor_type(
            onnx.helper.make_tensor_type_proto(tensor.data_type, tensor.dims)
        )
        assert isinstance(global_.initializer, bytes)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).newbyteorder("<")
        np.testing.assert_array_equal(
            np.frombuffer(global_.initializer, dtype=dtype),
            onnx.numpy_helper.to_array(tensor).reshape(-1),
        )

    outputs = _assert_graph_body(block, graph, values, initializers)
    return_ = block.operations[-1]
    assert isinstance(return_, ir.Return)
    assert return_.operands == outputs


def _assert_graph_body(
    block: ir.Block,
    graph: onnx.GraphProto,
    values: dict[str, ir.Value],
    initializers: dict[str, onnx.TensorProto],
) -> tuple[ir.Value, ...]:
    """Compare nodes recursively and return the expected terminator operands."""
    types = {
        value.name: _tensor_type(value.type)
        for value in (*graph.value_info, *graph.output)
    }
    conditions: dict[ir.ValueId, ir.Value] = {}
    nodes = iter(graph.node)
    for operation in block.operations[:-1]:
        if isinstance(operation, ir.TensorExtract):
            assert not operation.indices
            assert operation.result.type is ir.ScalarType.BOOL
            conditions[operation.operand.id] = operation.result
            continue
        if isinstance(operation, ir.GetGlobal):
            assert operation.name in initializers
            assert operation.name not in values
            values[operation.name] = operation.result
            continue
        node = next(nodes)
        assert isinstance(
            operation, _SUPPORTED_OP_TYPES.get(node.op_type, ir.UnknownOp)
        )
        operands = tuple(values[name] for name in node.input)
        if isinstance(operation, ir.If):
            operands = (conditions[operands[0].id],)
        assert ir.get_operands(operation) == operands
        results = ir.get_results(operation)
        assert tuple(result.type for result in results) == tuple(
            types[name] for name in node.output
        )
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if isinstance(operation, ir.UnknownOp):
            assert operation.name == f"{node.domain or 'onnx'}.{node.op_type}"
            assert operation.attributes == attributes
        elif isinstance(operation, ir.If):
            for name, region in (
                ("then_branch", operation.then_region),
                ("else_branch", operation.else_region),
            ):
                (branch,) = region.blocks
                assert not branch.arguments
                outputs = _assert_graph_body(
                    branch, attributes[name], dict(values), initializers
                )
                yield_ = branch.operations[-1]
                assert isinstance(yield_, ir.Yield)
                assert yield_.operands == outputs
        elif isinstance(operation, ir.Transpose):
            rank = len(types[node.output[0]].shape or ())
            assert operation.permutation == tuple(
                attributes.get("perm", reversed(range(rank)))
            )
        values.update(zip(node.output, results, strict=True))
    assert next(nodes, None) is None
    return tuple(values[output.name] for output in graph.output)


def _unknown_operators() -> tuple[onnx_st.Operator, ...]:
    """Return generation rules for operators imported as UnknownOp."""
    return (
        onnx_st.OPERATORS.identity,
        onnx_st.OPERATORS.relu,
        onnx_st.unary("Neg"),
        onnx_st.broadcast_binary("Sub"),
        onnx_st.broadcast_binary("Less", output_element_type=onnx.TensorProto.BOOL),
        onnx_st.Operator("TopK", _top_k),
    )


def _native_operators() -> tuple[onnx_st.Operator, ...]:
    """Return generation rules within Niro's supported native operator subset."""
    return (
        onnx_st.Operator("Add", _same_type_binary),
        onnx_st.Operator("Mul", _same_type_binary),
        onnx_st.Operator("MatMul", _matrix_multiply),
        onnx_st.OPERATORS.transpose,
        onnx_st.OPERATORS.if_,
    )


def _same_type_binary(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    """Niro's native Add and Mul require identical numeric tensor types."""
    pairs = [
        (lhs, rhs)
        for lhs in context.values
        for rhs in context.values
        if lhs.type == rhs.type
        and lhs.type.tensor_type.elem_type != onnx.TensorProto.BOOL
    ]
    if not pairs:
        return None
    return st.sampled_from(pairs).map(
        lambda pair: onnx_st.NodeSpec((pair[0].name, pair[1].name), (pair[0].type,))
    )


def _matrix_multiply(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    """Niro's native MatMul currently accepts rank-two tensors only."""
    matrices = tuple(
        value for value in context.values if len(onnx_st.tensor_shape(value)) == 2
    )
    return onnx_st.OPERATORS.matmul.strategy(
        dataclasses.replace(context, values=matrices)
    )


@st.composite
def _top_k_nodes(draw: DrawFn, values: tuple[onnx_st.Value, ...]) -> onnx_st.NodeSpec:
    value = draw(st.sampled_from(values))
    shape = onnx_st.tensor_shape(value)
    k = draw(st.integers(1, shape[-1]))
    output_shape = (*shape[:-1], k)
    return onnx_st.NodeSpec(
        inputs=(
            value.name,
            onnx.helper.make_tensor("", onnx.TensorProto.INT64, (1,), (k,)),
        ),
        outputs=(
            onnx.helper.make_tensor_type_proto(
                value.type.tensor_type.elem_type, output_shape
            ),
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, output_shape),
        ),
        attributes=(
            onnx.helper.make_attribute("axis", -1),
            onnx.helper.make_attribute("largest", int(draw(st.booleans()))),
            onnx.helper.make_attribute("sorted", int(draw(st.booleans()))),
        ),
    )


def _top_k(context: onnx_st.Context) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if (
        not context.initializer_slots
        or context.limits.max_rank < 1
        or context.limits.max_dim < 1
    ):
        return None
    values = tuple(
        value
        for value in context.values
        if value.type.tensor_type.elem_type != onnx.TensorProto.BOOL
        and onnx_st.tensor_shape(value)
        and onnx_st.tensor_shape(value)[-1] > 0
    )
    return _top_k_nodes(values) if values else None


def _tensor_type(type_: onnx.TypeProto) -> ir.TensorType:
    element_types: dict[int, ir.ScalarType] = {
        onnx.TensorProto.FLOAT: ir.ScalarType.F32,
        onnx.TensorProto.DOUBLE: ir.ScalarType.F64,
        onnx.TensorProto.INT32: ir.ScalarType.I32,
        onnx.TensorProto.INT64: ir.ScalarType.I64,
        onnx.TensorProto.BOOL: ir.ScalarType.BOOL,
    }
    tensor = type_.tensor_type
    return ir.TensorType(
        element_types[tensor.elem_type],
        tuple(dim.dim_value for dim in tensor.shape.dim),
    )
