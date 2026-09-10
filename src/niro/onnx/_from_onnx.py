from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import onnx

from niro import builder, ir
from niro.ir import VerifiedModule
from niro.onnx.op_type import OnnxOpType

from .value_table import OnnxValueName, OnnxValueTable

_ONNX_DOMAINS = (
    "",
    "ai.onnx.ml",
    "ai.onnx.preview",
    "ai.onnx.preview.training",
)


@dataclass(frozen=True)
class Ctx:
    """Import context holding a graph and its initializer and value-type mappings."""

    module: builder.ModuleBuilder
    graph: onnx.GraphProto
    weights: Mapping[OnnxValueName, ir.Global]
    types: Mapping[OnnxValueName, ir.Type]


def from_onnx(onnx_model: onnx.ModelProto) -> VerifiedModule:
    """Convert an ONNX model to verified Niro IR.

    Import If branches recursively, preserving captures and local initializers.
    Conditions need a statically known single-element Boolean tensor; branches
    must yield the declared result types. Unrecognized operations remain opaque
    when their attributes are supported scalar values or flat sequences.
    """
    graph = onnx_model.graph
    module = builder.ModuleBuilder()
    weights = _import_initializers(graph, module)
    ctx = Ctx(
        module=module,
        graph=graph,
        weights=weights,
        types=_collect_types(graph),
    )
    _import_forward(ctx, module)
    return module.verify()


def node_name(node: onnx.NodeProto) -> str:
    """Return the normalized node name for an ONNX operation."""
    domain = node.domain or "onnx"
    return f"{domain}.{node.op_type}"


def _import_forward(ctx: Ctx, module: builder.ModuleBuilder) -> ir.Function:
    """Build and return the graph entry point, adding it to the module.

    Resolve operands in graph order and load initializers on first use, including
    those returned directly. Inputs must be named and declared graph outputs
    must match the resulting types.
    """
    fn = _declare_entry_point(ctx.graph, module)
    input_names = fn.raw.input_names
    output_names = fn.raw.output_names
    assert input_names is not None
    assert all(input_names)
    assert output_names is not None

    block = fn.region().first_block()

    value_table = OnnxValueTable()
    value_table.define_many(
        (cast(str, name) for name in input_names),
        block.raw.arguments,
    )
    outputs = _import_graph(ctx, block, value_table)
    for output, ty in zip(outputs, fn.raw.type.outputs, strict=True):
        assert output.type == ty
    block.return_(*outputs)
    return fn.raw


def _import_graph(
    ctx: Ctx, block: builder.BlockBuilder, values: OnnxValueTable
) -> tuple[ir.Value, ...]:
    """Import nodes into a scope and resolve graph outputs for its terminator."""
    for node in ctx.graph.node:
        operands = [_resolve_value(ctx, block, values, name) for name in node.input]
        result = _import_node(ctx, block, node, operands, values)
        results = (result,) if isinstance(result, ir.Value) else result
        values.define_many(node.output, results)
    return tuple(
        _resolve_value(ctx, block, values, output.name) for output in ctx.graph.output
    )


def _resolve_value(
    ctx: Ctx, block: builder.BlockBuilder, values: OnnxValueTable, name: str
) -> ir.Value:
    """Resolve a visible value, loading an initializer into this scope on first use."""
    if name not in values and name in ctx.weights:
        values.define(name, block.get_global(ctx.weights[name]))
    return values.lookup(name)


def _declare_entry_point(
    graph: onnx.GraphProto,
    module: builder.ModuleBuilder,
) -> builder.FunctionBuilder:
    """Add and return a function declaration, excluding initializers from inputs."""
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
    block: builder.BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
    values: OnnxValueTable,
) -> ir.Value | Sequence[ir.Value]:
    """Append a supported or opaque ONNX operation and return its SSA results."""
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
        case OnnxOpType.If:
            return _import_if(ctx, block, node, operands, values)
        case OnnxOpType.Transpose:
            return _import_transpose(block, node, operands)
        case _:
            return _import_unknown_node(ctx, block, node, operands)


def _import_if(
    ctx: Ctx,
    block: builder.BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
    values: OnnxValueTable,
) -> tuple[ir.Value, ...]:
    """Import branches into isolated regions that capture the enclosing scope.

    Conditions must be single-element Boolean tensors; extract their scalar
    value before constructing If. Branch outputs must match the declared
    result types, as required by Niro If. Branch-local initializers become globals
    with unique names and are bound only within their branch.
    """
    (condition,) = operands
    type_ = condition.type
    if (
        not isinstance(type_, ir.TensorType)
        or type_.element_type is not ir.ScalarType.BOOL
    ):
        raise TypeError("ONNX If condition must be a Boolean tensor")
    if type_.shape is None or any(dimension != 1 for dimension in type_.shape):
        raise NotImplementedError(
            "ONNX If condition must have a statically known single element"
        )
    indices = tuple(block.const(0, ir.ScalarType.I64) for _ in type_.shape)
    condition = block.tensor_extract(condition, indices)
    conditional = block.if_(condition, [ctx.types[name] for name in node.output])
    attributes = {attribute.name: attribute for attribute in node.attribute}
    for name, region in (
        ("then_branch", conditional.then_region),
        ("else_branch", conditional.else_region),
    ):
        attribute = attributes[name]
        if attribute.type != onnx.AttributeProto.GRAPH:
            raise ValueError(f"ONNX If {name!r} must be a graph")
        graph = attribute.g
        if graph.input:
            raise NotImplementedError("ONNX If branches must have no explicit inputs")
        local_weights = _import_initializers(graph, ctx.module)
        branch_ctx = Ctx(
            module=ctx.module,
            graph=graph,
            weights={**ctx.weights, **local_weights},
            types={**ctx.types, **_collect_types(graph)},
        )
        branch = region.block()
        branch_values = OnnxValueTable(parent=values)
        for weight_name, weight in local_weights.items():
            branch_values.define(weight_name, branch.get_global(weight))
        branch.yield_(*_import_graph(branch_ctx, branch, branch_values))
    return conditional.raw.results


def _import_transpose(
    block: builder.BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
) -> ir.Value:
    """Append a transpose and return its result.

    An omitted permutation reverses all axes and requires a known operand rank.
    """
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
    block: builder.BlockBuilder,
    node: onnx.NodeProto,
    operands: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    """Append an opaque operation and return results with declared ONNX types.

    Every output must have a type in the context; attributes must be supported
    scalar values or flat sequences of those values.
    """
    return block.unknown_op(
        name=node_name(node),
        operands=operands,
        result_types=[ctx.types[name] for name in node.output],
        attributes={attr.name: _attribute_value(attr) for attr in node.attribute},
    )


def _import_initializers(
    graph: onnx.GraphProto,
    module: builder.ModuleBuilder,
) -> dict[OnnxValueName, ir.Global]:
    """Add uniquely named globals and return their graph-local ONNX-name mapping."""
    names = {symbol.name for symbol in (*module.raw.globals, *module.raw.functions)}
    sym_table: dict[OnnxValueName, ir.Global] = {}
    for t in graph.initializer:
        ty = _tensor_type(t)
        val = _tensor_data(t)
        name = t.name
        suffix = 0
        while name in names:
            suffix += 1
            name = f"{t.name}.{suffix}"
        names.add(name)
        sym_table[t.name] = module.global_(name, ty, val)
    return sym_table


def _attribute_value(attr: onnx.AttributeProto) -> ir.AttributeValue:
    """Return a scalar or flat tuple attribute, rejecting unsupported ONNX values."""
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
    """Return a tensor type, preserving unknown ranks and dynamic dimensions."""
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
    """Return tensor contents as contiguous, little-endian, row-major bytes."""
    array = onnx.numpy_helper.to_array(proto)
    little_endian_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_endian_dtype, copy=False).tobytes(order="C")


def _tensor_type(proto: onnx.TensorProto) -> ir.TensorType:
    """Return the tensor initializer type with its concrete dimensions."""
    scalar_type = _scalar_type(proto.data_type)
    return ir.TensorType(
        element_type=scalar_type,
        shape=tuple(proto.dims),
    )


def _scalar_type(proto: onnx.TensorProto.DataType | int) -> ir.ScalarType:
    """Return the corresponding Niro scalar type, rejecting unsupported types."""
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
    """Return declared input, intermediate, and output tensor types by ONNX name."""
    onnx_values = (
        *graph.input,
        *graph.value_info,
        *graph.output,
    )
    return {value.name: _value_type(value) for value in onnx_values}
