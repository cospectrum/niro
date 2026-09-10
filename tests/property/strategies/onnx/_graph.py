"""Assemble typed ONNX graphs from independent operator strategies."""

import math
from collections.abc import Mapping, Sequence

import onnx
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from ._core import (
    ELEMENT_TYPES,
    OPSET_VERSION,
    Context,
    Limits,
    NodeSpec,
    Operator,
    Value,
    tensor_shapes,
)
from ._operators import OPERATORS


def _literals(element_type: int) -> SearchStrategy[bool | int | float]:
    """Generate small, finite initializer elements representable in their type."""
    if element_type == onnx.TensorProto.BOOL:
        return st.booleans()
    if element_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
        return st.integers(-2, 2)
    width = 32 if element_type == onnx.TensorProto.FLOAT else 64
    return st.floats(-2, 2, allow_nan=False, allow_infinity=False, width=width)


def _assert_type_bounds(type_: onnx.TypeProto, limits: Limits) -> None:
    """Check concrete dimensions, including tensors nested in container types."""
    kind = type_.WhichOneof("value")
    assert kind is not None, "operator outputs must declare an ONNX type"
    if kind in ("tensor_type", "sparse_tensor_type"):
        tensor = (
            type_.tensor_type if kind == "tensor_type" else type_.sparse_tensor_type
        )
        if tensor.HasField("shape"):
            assert len(tensor.shape.dim) <= limits.max_rank, "tensor exceeds max_rank"
            assert all(
                0 <= dimension.dim_value <= limits.max_dim
                for dimension in tensor.shape.dim
                if dimension.HasField("dim_value")
            ), "tensor exceeds max_dim"
    elif kind == "sequence_type":
        _assert_type_bounds(type_.sequence_type.elem_type, limits)
    elif kind == "optional_type":
        _assert_type_bounds(type_.optional_type.elem_type, limits)
    elif kind == "map_type":
        _assert_type_bounds(type_.map_type.value_type, limits)


def _materialize(
    operator: Operator, spec: NodeSpec, context: Context, index: int
) -> tuple[onnx.NodeProto, list[Value], list[onnx.TensorProto]]:
    """Allocate node names and copy a rule's outputs, attributes, and constants."""
    names = {value.name for value in context.values}
    inputs: list[str] = []
    added_values: list[Value] = []
    initializers: list[onnx.TensorProto] = []
    for position, operand in enumerate(spec.inputs):
        if operand is None:
            inputs.append("")
        elif isinstance(operand, str):
            assert operand in names, f"operator references undefined value {operand!r}"
            inputs.append(operand)
        else:
            assert len(initializers) < context.initializer_slots, (
                "initializer budget exceeded"
            )
            initializer = onnx.TensorProto()
            initializer.CopyFrom(operand)
            initializer.name = f"node{index}_input{position}"
            type_ = onnx.helper.make_tensor_type_proto(
                initializer.data_type, initializer.dims
            )
            _assert_type_bounds(type_, context.limits)
            initializers.append(initializer)
            added_values.append(Value(initializer.name, type_))
            inputs.append(initializer.name)

    outputs: list[str] = []
    for position, type_ in enumerate(spec.outputs):
        if type_ is None:
            outputs.append("")
            continue
        _assert_type_bounds(type_, context.limits)
        copied_type = onnx.TypeProto()
        copied_type.CopyFrom(type_)
        value = Value(f"node{index}_output{position}", copied_type)
        added_values.append(value)
        outputs.append(value.name)
    node = onnx.helper.make_node(
        operator.op_type, inputs, outputs, name=f"node{index}", domain=operator.domain
    )
    node.attribute.extend(spec.attributes)
    return node, added_values, initializers


def _resolve_operators(
    operators: Sequence[str | Operator] | None,
) -> tuple[Operator, ...]:
    """Resolve registered names while allowing independent caller-supplied rules."""
    if operators is None:
        return tuple(OPERATORS)
    registry = {operator.op_type: operator for operator in OPERATORS}
    resolved = []
    for operator in operators:
        if isinstance(operator, str):
            if operator not in registry:
                raise ValueError(
                    f"unsupported operators: {operator!r}; supply an Operator rule"
                )
            operator = registry[operator]
        resolved.append(operator)
    return tuple(resolved)


@st.composite
def models(
    draw: DrawFn,
    *,
    min_nodes: int = 0,
    max_nodes: int = 12,
    max_inputs: int = 4,
    max_initializers: int = 3,
    max_seed_initializers: int | None = None,
    max_outputs: int = 4,
    max_rank: int = 3,
    max_dim: int = 4,
    operators: Sequence[str | Operator] | None = None,
    element_types: Sequence[int] = ELEMENT_TYPES,
    opsets: Mapping[str, int] | None = None,
    ir_version: int | None = None,
) -> onnx.ModelProto:
    """Generate a fresh ONNX model by composing compatible operator rules.

    Registered names and independent Operator objects can be mixed in
    ``operators``. The defaults retain Add, Mul, MatMul, Transpose, Identity,
    and Relu. A rule returns a strategy for a valid NodeSpec or None when it
    cannot apply. There is no filtering, shape repair, or checker invocation.

    Bounds are nonnegative, max_outputs is positive, and min_nodes cannot
    exceed max_nodes. Node bounds count top-level nodes. Generation stops when
    no rule applies; failing to reach min_nodes raises ValueError. Initial
    values can be absent when an applicable zero-input rule supplies them.

    max_initializers includes auxiliary tensors supplied by rules. Set
    max_seed_initializers=0 to reserve that budget for auxiliary inputs; None
    allows random initializers to use the whole budget. element_types restricts
    random sources only: operators may require other auxiliary or output types.

    Tensor ranks and concrete dimensions, including those in auxiliary inputs
    and container types, must respect max_rank and max_dim. Rules are responsible
    for data constraints, symbolic dimensions, subgraphs and their bounds. Every
    introduced value has value_info; outputs may reference any available value.

    opsets updates the default {"": 14} domain imports. Rules must support the
    configured version, and custom domains need an explicit import. IR version
    is inferred from known opsets unless overridden; custom domains may require
    an explicit ir_version. This API enables extensions, not automatic coverage
    of every ONNX operator or validation of arbitrary caller-supplied rules.
    """
    if min(min_nodes, max_nodes, max_inputs, max_initializers, max_rank, max_dim) < 0:
        raise ValueError("model generation bounds must be nonnegative")
    if min_nodes > max_nodes:
        raise ValueError("min_nodes cannot exceed max_nodes")
    if max_outputs < 1:
        raise ValueError("max_outputs must be positive")
    seed_limit = (
        max_initializers if max_seed_initializers is None else max_seed_initializers
    )
    if not 0 <= seed_limit <= max_initializers:
        raise ValueError("max_seed_initializers must be within the initializer budget")
    if not element_types or set(element_types) - set(ELEMENT_TYPES):
        raise ValueError("element_types must be a nonempty supported subset")
    rules = _resolve_operators(operators)
    versions = {"": OPSET_VERSION, **(opsets or {})}
    if any(version < 1 for version in versions.values()):
        raise ValueError("opset versions must be positive")
    for rule in rules:
        if rule.domain not in versions:
            raise ValueError(f"missing opset import for domain {rule.domain!r}")
    imports = [
        onnx.helper.make_opsetid(domain, version)
        for domain, version in versions.items()
    ]
    known_domains = {domain for domain, _ in onnx.helper.OP_SET_ID_VERSION_MAP} | {""}
    minimum_ir = max(
        onnx.helper.find_min_ir_version_for(
            [import_], ignore_unknown=import_.domain not in known_domains
        )
        for import_ in imports
    )
    if ir_version is None:
        ir_version = minimum_ir
    elif ir_version < minimum_ir:
        raise ValueError(f"configured opsets require IR version at least {minimum_ir}")

    limits = Limits(max_rank, max_dim, tuple(element_types))
    input_count = draw(
        st.integers(1 if max_inputs and not seed_limit else 0, max_inputs)
    )
    initializer_count = draw(
        st.integers(1 if seed_limit and not input_count else 0, seed_limit)
    )
    shapes = tensor_shapes(max_rank=max_rank, max_dim=max_dim)
    values: list[Value] = []
    for index in range(input_count + initializer_count):
        type_ = onnx.helper.make_tensor_type_proto(
            draw(st.sampled_from(element_types)), draw(shapes)
        )
        values.append(Value(f"v{index}", type_))
    inputs = [
        onnx.helper.make_value_info(value.name, value.type)
        for value in values[:input_count]
    ]
    initializers: list[onnx.TensorProto] = []
    for value in values[input_count:]:
        tensor = value.type.tensor_type
        shape = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        size = math.prod(shape)
        elements = draw(
            st.lists(_literals(tensor.elem_type), min_size=size, max_size=size)
        )
        initializers.append(
            onnx.helper.make_tensor(value.name, tensor.elem_type, shape, elements)
        )

    required = max(min_nodes, 0 if values else 1)
    if required > max_nodes:
        raise ValueError(
            "at least one input or initializer, or a source operator, is required"
        )
    nodes: list[onnx.NodeProto] = []
    intermediates: list[onnx.ValueInfoProto] = []
    for index in range(draw(st.integers(required, max_nodes))):
        context = Context(
            tuple(values),
            limits,
            max_initializers - len(initializers),
            versions,
            f"node{index}_nested",
        )
        available = [
            (rule, strategy)
            for rule in rules
            if (strategy := rule.strategy(context)) is not None
        ]
        if not available:
            break
        rule, strategy = draw(st.sampled_from(available))
        spec = draw(strategy)
        node, added_values, added_initializers = _materialize(
            rule, spec, context, index
        )
        nodes.append(node)
        values.extend(added_values)
        initializers.extend(added_initializers)
        intermediates.extend(
            onnx.helper.make_value_info(value.name, value.type)
            for value in added_values
        )
    if not values:
        raise ValueError(
            "at least one input or initializer, or an applicable source operator, is required"
        )
    if len(nodes) < min_nodes:
        raise ValueError("available operator rules cannot satisfy min_nodes")

    outputs = draw(
        st.lists(
            st.sampled_from(values),
            min_size=1,
            max_size=min(max_outputs, len(values)),
            unique_by=lambda value: value.name,
        )
    )
    graph = onnx.helper.make_graph(
        nodes,
        "main",
        inputs,
        [onnx.helper.make_value_info(value.name, value.type) for value in outputs],
        initializer=initializers,
        value_info=intermediates,
    )
    return onnx.helper.make_model(
        graph,
        ir_version=ir_version,
        opset_imports=imports,
        producer_name="niro.property",
    )
