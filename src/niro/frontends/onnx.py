"""ONNX frontend for Niro IR."""

from __future__ import annotations

from dataclasses import dataclass

import onnx
from onnx import numpy_helper

from niro import ir
from niro.builder import FunctionBuilder, ModuleBuilder

type OnnxValueName = str


@dataclass(slots=True)
class Ctx:
    graph: onnx.GraphProto
    module: ModuleBuilder
    function: FunctionBuilder
    types: dict[OnnxValueName, ir.Type]
    values: dict[OnnxValueName, ir.Value]


def import_onnx(onnx_model: onnx.ModelProto) -> ir.Module:
    """Import a single ONNX graph as a Niro entry-point function."""
    graph = onnx_model.graph
    types = _collect_types(graph)
    module = ModuleBuilder()
    initializer_names = {
        initializer.name for initializer in graph.initializer
    }
    inputs = [
        value
        for value in graph.input
        if value.name not in initializer_names
    ]
    function = module.func(
        name=graph.name or "main",
        arg_types=[
            _lookup_type(types, value.name) for value in inputs
        ],
        ret_types=[
            _lookup_type(types, value.name) for value in graph.output
        ],
    )
    ctx = Ctx(
        graph=graph,
        module=module,
        function=function,
        types=types,
        values={},
    )
    ctx.values.update(
        (value_info.name, argument)
        for value_info, argument in zip(
            inputs, function.args, strict=True
        )
    )

    for initializer in graph.initializer:
        initializer_type = _tensor_type(
            initializer.data_type, tuple(initializer.dims)
        )
        ctx.values[initializer.name] = function.tensor(
            data=_initializer_data(initializer),
            result_type=initializer_type,
        )
    for node in graph.node:
        _import_node(ctx, node)

    function.return_(
        *(_lookup_value(ctx.values, value.name) for value in graph.output)
    )
    module.entry_point(function)
    return module.ir


def _collect_types(graph: onnx.GraphProto) -> dict[OnnxValueName, ir.Type]:
    onnx_values = (
        *graph.input,
        *graph.value_info,
        *graph.output,
    )
    return {
        value.name: _value_info_type(value) for value in onnx_values
    }


def _import_node(
    ctx: Ctx,
    node: onnx.NodeProto,
) -> None:
    operands = [
        _lookup_value(ctx.values, name) for name in node.input
    ]
    if node.domain not in ("", "ai.onnx"):
        _import_unknown(ctx, node, operands)
        return

    match node.op_type:
        case "Add":
            lhs, rhs = _require_two_inputs(node, operands)
            _record_output(
                ctx,
                node,
                ctx.function.add(lhs, rhs),
            )
        case "Mul":
            lhs, rhs = _require_two_inputs(node, operands)
            _record_output(
                ctx,
                node,
                ctx.function.mul(lhs, rhs),
            )
        case "MatMul":
            lhs, rhs = _require_two_inputs(node, operands)
            _record_output(
                ctx,
                node,
                ctx.function.matmul(
                    lhs,
                    rhs,
                    _lookup_type(ctx.types, node.output[0]),
                ),
            )
        case "Transpose":
            if len(operands) != 1:
                raise ValueError("Transpose must have exactly one input")
            attributes = {
                attribute.name: attribute for attribute in node.attribute
            }
            raw_permutation = (
                onnx.helper.get_attribute_value(attributes["perm"])
                if "perm" in attributes
                else range(
                    _require_known_rank(operands[0]) - 1,
                    -1,
                    -1,
                )
            )
            permutation = tuple(int(index) for index in raw_permutation)
            _record_output(
                ctx,
                node,
                ctx.function.transpose(operands[0], permutation),
            )
        case _:
            _import_unknown(ctx, node, operands)


def _import_unknown(
    ctx: Ctx,
    node: onnx.NodeProto,
    operands: list[ir.Value],
) -> None:
    if any(not name for name in node.output):
        raise NotImplementedError(
            f"optional outputs are not supported for {node.op_type}"
        )
    results = ctx.function.unknown(
        name=_operation_name(node),
        operands=operands,
        result_types=[
            _lookup_type(ctx.types, name) for name in node.output
        ],
        attributes={
            attribute.name: _attribute(attribute)
            for attribute in node.attribute
        },
    )
    ctx.values.update(
        zip(node.output, results, strict=True)
    )


def _record_output(
    ctx: Ctx,
    node: onnx.NodeProto,
    niro_value: ir.Value,
) -> None:
    if len(node.output) != 1 or not node.output[0]:
        raise ValueError(f"{node.op_type} must have exactly one output")
    ctx.values[node.output[0]] = niro_value


def _lookup_value(
    values: dict[OnnxValueName, ir.Value],
    onnx_name: OnnxValueName,
) -> ir.Value:
    try:
        return values[onnx_name]
    except KeyError:
        raise ValueError(f"unknown ONNX value: {onnx_name!r}") from None


def _lookup_type(
    types: dict[OnnxValueName, ir.Type],
    onnx_name: OnnxValueName,
) -> ir.Type:
    try:
        return types[onnx_name]
    except KeyError:
        raise ValueError(
            f"ONNX value has no type information: {onnx_name!r}"
        ) from None


def _value_info_type(
    value_info: onnx.ValueInfoProto,
) -> ir.TensorType:
    if not value_info.type.HasField("tensor_type"):
        raise NotImplementedError(
            f"ONNX value {value_info.name!r} is not a tensor"
        )
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        shape = None
    else:
        shape = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        )
    return _tensor_type(tensor_type.elem_type, shape)


def _tensor_type(
    element_type: int,
    shape: ir.Shape | None,
) -> ir.TensorType:
    scalar_types = {
        onnx.TensorProto.BOOL: ir.ScalarType.BOOL,
        onnx.TensorProto.INT32: ir.ScalarType.I32,
        onnx.TensorProto.INT64: ir.ScalarType.I64,
        onnx.TensorProto.FLOAT: ir.ScalarType.F32,
        onnx.TensorProto.DOUBLE: ir.ScalarType.F64,
    }
    try:
        scalar_type = scalar_types[element_type]
    except KeyError:
        raise NotImplementedError(
            f"unsupported ONNX tensor element type: {element_type}"
        ) from None
    return ir.TensorType(element_type=scalar_type, shape=shape)


def _initializer_data(initializer: onnx.TensorProto) -> bytes:
    array = numpy_helper.to_array(initializer)
    little_endian_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_endian_dtype, copy=False).tobytes(order="C")


def _operation_name(node: onnx.NodeProto) -> str:
    domain = (
        "onnx" if node.domain in ("", "ai.onnx") else node.domain
    )
    return f"{domain}.{node.op_type}"


def _attribute(
    attribute: onnx.AttributeProto,
) -> ir.Attribute:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, (bool, int, float, str, bytes)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _attribute_element(element) for element in value
        )
    raise NotImplementedError(
        f"unsupported ONNX attribute {attribute.name!r} on an unknown operation"
    )


def _attribute_element(value: object) -> ir.Attribute:
    if isinstance(value, (bool, int, float, str, bytes)) or value is None:
        return value
    raise NotImplementedError("unsupported value in an ONNX attribute sequence")


def _require_two_inputs(
    node: onnx.NodeProto,
    operands: list[ir.Value],
) -> tuple[ir.Value, ir.Value]:
    if len(operands) != 2:
        raise ValueError(f"{node.op_type} must have exactly two inputs")
    return operands[0], operands[1]


def _require_known_rank(value: ir.Value) -> int:
    match value.type:
        case ir.TensorType(shape=shape) if shape is not None:
            return len(shape)
        case ir.TensorType(shape=None):
            raise ValueError("cannot infer the default Transpose permutation")
        case _:
            raise TypeError("Transpose input must be a tensor")
