"""Assertions comparing imported Niro IR with its source ONNX graph."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import onnx

from niro import ir
from niro.onnx.op_type import OnnxOpType

DEFAULT_OP_TYPE_MAPPING: dict[OnnxOpType, type[ir.Op]] = {
    OnnxOpType.Add: ir.Add,
    OnnxOpType.Mul: ir.Mul,
    OnnxOpType.MatMul: ir.MatMul,
    OnnxOpType.Transpose: ir.Transpose,
    OnnxOpType.If: ir.If,
}


def assert_niro_matches_onnx(
    module: ir.Module,
    graph: onnx.GraphProto,
    op_type_mapping: Mapping[OnnxOpType, type[ir.Op]] = DEFAULT_OP_TYPE_MAPPING,
) -> None:
    """Check signatures, initializer contents, typed dataflow, attributes, and returns.

    The graph must use concrete tensor types, with value_info for node outputs,
    and the native or opaque operators supported by the importer property tests.
    op_type_mapping selects expected Niro classes; unmapped operators use UnknownOp.
    """
    (function,) = module.functions
    assert function.name == graph.name, (
        f"Expected function name {graph.name!r}, got {function.name!r}"
    )
    assert function.input_names == tuple(value.name for value in graph.input), (
        f"Graph {graph.name!r}: input names or order differ"
    )
    assert function.output_names == tuple(value.name for value in graph.output), (
        f"Graph {graph.name!r}: output names or order differ"
    )
    assert function.type.inputs == tuple(
        _tensor_type(value.type) for value in graph.input
    ), f"Graph {graph.name!r}: input types differ"
    assert function.type.outputs == tuple(
        _tensor_type(value.type) for value in graph.output
    ), f"Graph {graph.name!r}: output types differ"
    assert function.body is not None, f"Graph {graph.name!r}: function has no body"
    (block,) = function.body.blocks
    values = dict(
        zip((value.name for value in graph.input), block.arguments, strict=True)
    )
    initializers = {tensor.name: tensor for tensor in _iter_initializers(graph)}
    assert {global_.name for global_ in module.globals} == set(initializers), (
        f"Graph {graph.name!r}: globals do not match ONNX initializers"
    )
    for global_ in module.globals:
        tensor = initializers[global_.name]
        assert global_.type == _tensor_type(
            onnx.helper.make_tensor_type_proto(tensor.data_type, tensor.dims)
        ), f"Initializer {global_.name!r}: tensor type differs"
        assert isinstance(global_.initializer, bytes), (
            f"Initializer {global_.name!r}: expected bytes, "
            f"got {type(global_.initializer).__name__}"
        )
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).newbyteorder("<")
        np.testing.assert_array_equal(
            np.frombuffer(global_.initializer, dtype=dtype),
            onnx.numpy_helper.to_array(tensor).reshape(-1),
            err_msg=f"Initializer {global_.name!r}: tensor contents differ",
        )

    outputs = _assert_graph_body(block, graph, values, {}, op_type_mapping)
    return_ = block.operations[-1]
    assert isinstance(return_, ir.Return), (
        f"Graph {graph.name!r}: expected Return terminator, got {type(return_).__name__}"
    )
    assert return_.operands == outputs, (
        f"Graph {graph.name!r}: returned values do not match ONNX outputs"
    )


@dataclass(frozen=True)
class _GraphContext:
    """Track remaining ONNX nodes and value bindings within one graph scope."""

    graph_name: str
    nodes: Iterator[onnx.NodeProto]
    values: dict[str, ir.Value]
    types: Mapping[str, ir.TensorType]
    initializers: Mapping[str, onnx.TensorProto]
    conditions: dict[ir.ValueId, ir.Value]
    op_type_mapping: Mapping[OnnxOpType, type[ir.Op]]


def _assert_graph_body(
    block: ir.Block,
    graph: onnx.GraphProto,
    values: dict[str, ir.Value],
    initializers: Mapping[str, onnx.TensorProto],
    op_type_mapping: Mapping[OnnxOpType, type[ir.Op]],
) -> tuple[ir.Value, ...]:
    """Compare nodes recursively and return the expected terminator operands."""
    ctx = _GraphContext(
        graph_name=graph.name,
        nodes=iter(graph.node),
        values=values,
        types={
            value.name: _tensor_type(value.type)
            for value in (*graph.value_info, *graph.output)
        },
        initializers={
            **initializers,
            **{tensor.name: tensor for tensor in graph.initializer},
        },
        conditions={},
        op_type_mapping=op_type_mapping,
    )
    for operation in block.operations[:-1]:
        _assert_operation(ctx, operation)
    remaining = next(ctx.nodes, None)
    assert remaining is None, (
        f"Graph {graph.name!r}: ONNX node {remaining.name!r} "
        f"({remaining.op_type}, outputs={tuple(remaining.output)!r}) has no Niro operation"
    )
    return tuple(ctx.values[output.name] for output in graph.output)


def _assert_operation(ctx: _GraphContext, operation: ir.Op) -> None:
    """Check an operation and update its graph's node cursor and value bindings."""
    location = f"Graph {ctx.graph_name!r}, {type(operation).__name__}"
    if isinstance(operation, ir.TensorExtract):
        assert not operation.indices, (
            f"{location}: expected scalar condition extraction without indices"
        )
        assert operation.result.type is ir.ScalarType.BOOL, (
            f"{location}: expected Boolean condition, got {operation.result.type}"
        )
        ctx.conditions[operation.operand.id] = operation.result
        return
    if isinstance(operation, ir.GetGlobal):
        assert operation.name in ctx.initializers, (
            f"{location}: no ONNX initializer named {operation.name!r}"
        )
        assert operation.name not in ctx.values, (
            f"{location}: initializer {operation.name!r} is already bound"
        )
        ctx.values[operation.name] = operation.result
        return

    node = next(ctx.nodes, None)
    assert node is not None, f"{location}: no remaining ONNX node for this operation"
    location = (
        f"Graph {ctx.graph_name!r}, node {node.name!r} "
        f"({node.op_type}, outputs={tuple(node.output)!r})"
    )
    expected_type = ir.UnknownOp
    if node.op_type in OnnxOpType:
        expected_type = ctx.op_type_mapping.get(OnnxOpType(node.op_type), ir.UnknownOp)
    assert isinstance(operation, expected_type), (
        f"{location}: expected {expected_type.__name__}, got {type(operation).__name__}"
    )
    operands = tuple(ctx.values[name] for name in node.input)
    if isinstance(operation, ir.If):
        operands = (ctx.conditions[operands[0].id],)
    assert ir.get_operands(operation) == operands, (
        f"{location}: operands do not match ONNX inputs {tuple(node.input)!r}"
    )
    results = ir.get_results(operation)
    assert tuple(result.type for result in results) == tuple(
        ctx.types[name] for name in node.output
    ), f"{location}: result types differ from ONNX output types"
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    if isinstance(operation, ir.UnknownOp):
        attributes = {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in attributes.items()
        }
        assert operation.name == f"{node.domain or 'onnx'}.{node.op_type}", (
            f"{location}: unexpected opaque operation name {operation.name!r}"
        )
        assert operation.attributes == attributes, f"{location}: attributes differ"
    elif isinstance(operation, ir.If):
        for name, region in (
            ("then_branch", operation.then_region),
            ("else_branch", operation.else_region),
        ):
            (branch,) = region.blocks
            assert not branch.arguments, (
                f"{location}, {name}: expected captures, not explicit block arguments"
            )
            outputs = _assert_graph_body(
                branch,
                attributes[name],
                dict(ctx.values),
                ctx.initializers,
                ctx.op_type_mapping,
            )
            yield_ = branch.operations[-1]
            assert isinstance(yield_, ir.Yield), (
                f"{location}, {name}: expected Yield terminator, "
                f"got {type(yield_).__name__}"
            )
            assert yield_.operands == outputs, (
                f"{location}, {name}: yielded values do not match ONNX branch outputs"
            )
    elif isinstance(operation, ir.Transpose):
        rank = len(ctx.types[node.output[0]].shape or ())
        assert operation.permutation == tuple(
            attributes.get("perm", reversed(range(rank)))
        ), f"{location}: transpose permutation differs"
    ctx.values.update(zip(node.output, results, strict=True))


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


def _iter_initializers(graph: onnx.GraphProto) -> Iterator[onnx.TensorProto]:
    """Visit initializers in every scope; generated names are globally unique."""
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_initializers(attribute.g)
