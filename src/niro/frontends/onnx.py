"""ONNX frontend for Niro IR."""

from __future__ import annotations

import onnx
from onnx import numpy_helper

from niro import ir
from niro.builder import FunctionBuilder, ModuleBuilder


def import_onnx(model: onnx.ModelProto) -> ir.Module:
    """Import a single ONNX graph as a Niro entry-point function."""
    return _OnnxImporter(model).import_graph()


class _OnnxImporter:
    def __init__(self, model: onnx.ModelProto) -> None:
        self._graph = model.graph
        self._types = self._collect_value_types()
        self._values: dict[str, ir.Value] = {}

    def import_graph(self) -> ir.Module:
        module = ModuleBuilder()
        initializer_names = {initializer.name for initializer in self._graph.initializer}
        inputs = [
            value for value in self._graph.input if value.name not in initializer_names
        ]
        function = module.func(
            name=self._graph.name or "main",
            arg_types=[self._lookup_value_type(value.name) for value in inputs],
            ret_types=[
                self._lookup_value_type(value.name) for value in self._graph.output
            ],
        )
        self._values.update(
            (value_info.name, argument)
            for value_info, argument in zip(inputs, function.args, strict=True)
        )

        for initializer in self._graph.initializer:
            initializer_type = _convert_tensor_type(
                initializer.data_type, tuple(initializer.dims)
            )
            self._values[initializer.name] = function.tensor(
                data=_decode_initializer_data(initializer),
                result_type=initializer_type,
            )
        for node in self._graph.node:
            self._import_node(function, node)

        function.return_(
            *(self._lookup_imported_value(value.name) for value in self._graph.output)
        )
        module.entry_point(function)
        return module.ir

    def _collect_value_types(self) -> dict[str, ir.Type]:
        values = (*self._graph.input, *self._graph.value_info, *self._graph.output)
        return {value.name: _convert_value_info_type(value) for value in values}

    def _import_node(
        self,
        function: FunctionBuilder,
        node: onnx.NodeProto,
    ) -> None:
        operands = [self._lookup_imported_value(name) for name in node.input]
        if node.domain not in ("", "ai.onnx"):
            self._import_unknown_node(function, node, operands)
            return

        match node.op_type:
            case "Add":
                lhs, rhs = _require_two_inputs(node, operands)
                self._record_single_node_output(node, function.add(lhs, rhs))
            case "Mul":
                lhs, rhs = _require_two_inputs(node, operands)
                self._record_single_node_output(node, function.mul(lhs, rhs))
            case "MatMul":
                lhs, rhs = _require_two_inputs(node, operands)
                self._record_single_node_output(
                    node,
                    function.matmul(
                        lhs, rhs, self._lookup_value_type(node.output[0])
                    ),
                )
            case "Transpose":
                if len(operands) != 1:
                    raise ValueError("Transpose must have exactly one input")
                attributes = {attribute.name: attribute for attribute in node.attribute}
                raw_permutation = (
                    onnx.helper.get_attribute_value(attributes["perm"])
                    if "perm" in attributes
                    else range(_require_known_tensor_rank(operands[0]) - 1, -1, -1)
                )
                permutation = tuple(int(index) for index in raw_permutation)
                self._record_single_node_output(
                    node, function.transpose(operands[0], permutation)
                )
            case _:
                self._import_unknown_node(function, node, operands)

    def _import_unknown_node(
        self,
        function: FunctionBuilder,
        node: onnx.NodeProto,
        operands: list[ir.Value],
    ) -> None:
        if any(not name for name in node.output):
            raise NotImplementedError(
                f"optional outputs are not supported for {node.op_type}"
            )
        results = function.unknown(
            name=_qualified_operation_name(node),
            operands=operands,
            result_types=[self._lookup_value_type(name) for name in node.output],
            attributes={
                attribute.name: _convert_attribute(attribute)
                for attribute in node.attribute
            },
        )
        self._values.update(zip(node.output, results, strict=True))

    def _record_single_node_output(
        self, node: onnx.NodeProto, value: ir.Value
    ) -> None:
        if len(node.output) != 1 or not node.output[0]:
            raise ValueError(f"{node.op_type} must have exactly one output")
        self._values[node.output[0]] = value

    def _lookup_imported_value(self, name: str) -> ir.Value:
        try:
            return self._values[name]
        except KeyError:
            raise ValueError(f"unknown ONNX value: {name!r}") from None

    def _lookup_value_type(self, name: str) -> ir.Type:
        try:
            return self._types[name]
        except KeyError:
            raise ValueError(f"ONNX value has no type information: {name!r}") from None


def _convert_value_info_type(value: onnx.ValueInfoProto) -> ir.TensorType:
    if not value.type.HasField("tensor_type"):
        raise NotImplementedError(f"ONNX value {value.name!r} is not a tensor")
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        shape = None
    else:
        shape = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in tensor.shape.dim
        )
    return _convert_tensor_type(tensor.elem_type, shape)


def _convert_tensor_type(
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


def _decode_initializer_data(initializer: onnx.TensorProto) -> bytes:
    array = numpy_helper.to_array(initializer)
    little_endian_dtype = array.dtype.newbyteorder("<")
    return array.astype(little_endian_dtype, copy=False).tobytes(order="C")


def _qualified_operation_name(node: onnx.NodeProto) -> str:
    domain = "onnx" if node.domain in ("", "ai.onnx") else node.domain
    return f"{domain}.{node.op_type}"


def _convert_attribute(attribute: onnx.AttributeProto) -> ir.Attribute:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, (bool, int, float, str, bytes)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_convert_attribute_element(element) for element in value)
    raise NotImplementedError(
        f"unsupported ONNX attribute {attribute.name!r} on an unknown operation"
    )


def _convert_attribute_element(value: object) -> ir.Attribute:
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


def _require_known_tensor_rank(value: ir.Value) -> int:
    match value.type:
        case ir.TensorType(shape=shape) if shape is not None:
            return len(shape)
        case ir.TensorType(shape=None):
            raise ValueError("cannot infer the default Transpose permutation")
        case _:
            raise TypeError("Transpose input must be a tensor")
