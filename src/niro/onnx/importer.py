from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import onnx

from niro import ir
from niro.builder import BlockBuilder, FunctionBuilder, ModuleBuilder

from .op_type import OnnxOpType
from .value_table import OnnxValueName, OnnxValueTable


_ONNX_DOMAINS = (
    "",
    "ai.onnx.ml",
    "ai.onnx.preview",
    "ai.onnx.preview.training",
)


@dataclass(frozen=True)
class Ctx:
    graph: onnx.GraphProto
    weights: Mapping[OnnxValueName, ir.Global]
    types: Mapping[OnnxValueName, ir.Type]


def import_onnx(onnx_model: onnx.ModelProto) -> ir.Module:
    graph = onnx_model.graph
    module = ModuleBuilder()
    weights = _import_initializers(graph, module)
    ctx = Ctx(
        graph=graph,
        weights=weights,
        types=_collect_types(graph),
    )
    _import_forward(ctx, module)
    return module.ir


def node_name(node: onnx.NodeProto) -> str:
    """Return the normalized node name for an ONNX operation."""
    domain = node.domain or "onnx"
    return f"{domain}.{node.op_type}"


def _import_forward(ctx: Ctx, module: ModuleBuilder) -> ir.Function:
    fn = _declare_entry_point(ctx.graph, module)
    input_names = fn.ir.input_names
    output_names = fn.ir.output_names
    assert input_names is not None
    assert all(input_names)
    assert output_names is not None

    block = fn.region().block(fn.ir.type.inputs)

    value_table = OnnxValueTable()
    value_table.define_many(
        (cast(str, name) for name in input_names),
        block.ir.arguments,
    )
    for node in ctx.graph.node:
        operands = []
        for name in node.input:
            assert name
            if name in value_table:
                pass
            elif name in ctx.weights:
                value = block.get_global(ctx.weights[name])
                value_table.define(name, value)
            else:
                raise ValueError(
                    f"could not resolve ONNX value {name!r} "
                    f"used as an operand by node {node_name(node)!r}"
                )
            operands.append(value_table.lookup(name))

        result = _import_node(ctx, block, node, operands)
        results = (result,) if isinstance(result, ir.Value) else result
        value_table.define_many(node.output, results)

    outputs = [value_table.lookup(cast(str, name)) for name in output_names]
    for output, ty in zip(outputs, fn.ir.type.outputs, strict=True):
        assert output.type == ty

    block.return_(*outputs)
    return fn.ir


def _declare_entry_point(
    graph: onnx.GraphProto,
    module: ModuleBuilder,
) -> FunctionBuilder:
    initializer_names = {t.name for t in graph.initializer}
    pb_inputs = [val for val in graph.input if val.name not in initializer_names]
    pb_outputs = [val for val in graph.output]

    input_types = tuple(_value_type(val) for val in pb_inputs)
    output_types = tuple(_value_type(val) for val in pb_outputs)
    input_names = [val.name for val in pb_inputs]
    output_names = [val.name for val in pb_outputs]
    return module.function(
        name=graph.name,
        type=ir.FunctionType(inputs=input_types, outputs=output_types),
        input_names=input_names,
        output_names=output_names,
    )


def _import_node(
    ctx: Ctx,
    block: BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
) -> ir.Value | Sequence[ir.Value]:
    if node.domain not in _ONNX_DOMAINS:
        return _import_unknown_node(ctx, block, node, operands)

    match node.op_type:
        case OnnxOpType.Add:
            lhs, rhs = operands
            return block.add(lhs, rhs)
        case OnnxOpType.Mul:
            lhs, rhs = operands
            return block.mul(lhs, rhs)
        case OnnxOpType.MatMul:
            lhs, rhs = operands
            return block.matmul(lhs, rhs)
        case OnnxOpType.Transpose:
            return _import_transpose(block, node, operands)
        case _:
            return _import_unknown_node(ctx, block, node, operands)


def _import_transpose(
    block: BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
) -> ir.Value:
    (operand,) = operands
    attributes = {attribute.name: attribute for attribute in node.attribute}
    if "perm" in attributes:
        raw_permutation = onnx.helper.get_attribute_value(attributes["perm"])
    else:
        assert isinstance(operand.type, ir.TensorType)
        rank = operand.type.rank
        if rank is None:
            raise ValueError("cannot infer the default Transpose permutation")
        raw_permutation = reversed(range(rank))
    permutation = tuple(int(index) for index in raw_permutation)
    return block.transpose(operand, permutation)


def _import_unknown_node(
    ctx: Ctx,
    block: BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    return block.unknown_op(
        name=node_name(node),
        operands=operands,
        result_types=[ctx.types[name] for name in node.output],
        attributes={attr.name: _attribute_value(attr) for attr in node.attribute},
    )


def _import_initializers(
    graph: onnx.GraphProto,
    module: ModuleBuilder,
) -> dict[OnnxValueName, ir.Global]:
    for t in graph.initializer:
        ty = _tensor_type(t)
        val = _tensor_data(t)
        module.global_(t.name, ty, val)
    globals = module.ir.globals
    assert len(globals) >= len(graph.initializer)
    sym_table = {global_.name: global_ for global_ in globals}
    return sym_table


def _attribute_value(attr: onnx.AttributeProto) -> ir.AttributeValue:
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, (bool, int, float, str, bytes)) or value is None:
        return value
    if not isinstance(value, Iterable):
        raise NotImplementedError(
            f"unsupported ONNX attribute {attr.name!r} on an unknown operation"
        )
    vals = []
    for el in value:
        assert isinstance(el, (bool, int, float, str, bytes)) or el is None
        vals.append(el)
    return tuple(vals)


def _value_type(value_info: onnx.ValueInfoProto) -> ir.TensorType:
    if not value_info.type.HasField("tensor_type"):
        raise NotImplementedError(f"ONNX value {value_info.name!r} is not a tensor")
    tensor_type = value_info.type.tensor_type
    if tensor_type.HasField("shape"):
        shape = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        )
    else:
        shape = None
    return ir.TensorType(_scalar_type(tensor_type.elem_type), shape)


def _tensor_data(proto: onnx.TensorProto) -> bytes:
    array = onnx.numpy_helper.to_array(proto)
    little_endian_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_endian_dtype, copy=False).tobytes(order="C")


def _tensor_type(proto: onnx.TensorProto) -> ir.TensorType:
    scalar_type = _scalar_type(proto.data_type)
    return ir.TensorType(
        element_type=scalar_type,
        shape=tuple(proto.dims),
    )


def _scalar_type(proto: onnx.TensorProto.DataType | int) -> ir.ScalarType:
    scalar_types: dict[int, ir.ScalarType] = {
        onnx.TensorProto.BOOL: ir.ScalarType.BOOL,
        onnx.TensorProto.INT32: ir.ScalarType.I32,
        onnx.TensorProto.INT64: ir.ScalarType.I64,
        onnx.TensorProto.FLOAT: ir.ScalarType.F32,
        onnx.TensorProto.DOUBLE: ir.ScalarType.F64,
    }
    try:
        return scalar_types[proto]
    except KeyError:
        raise NotImplementedError(f"unsupported ONNX data type: {proto}") from None


def _collect_types(graph: onnx.GraphProto) -> dict[OnnxValueName, ir.Type]:
    onnx_values = (
        *graph.input,
        *graph.value_info,
        *graph.output,
    )
    return {value.name: _value_type(value) for value in onnx_values}
