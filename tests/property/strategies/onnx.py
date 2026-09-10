"""Construct bounded, valid ONNX graphs directly with Hypothesis and ONNX.

Like NNSmith, generation tracks available tensor types and shapes and chooses
compatible operands. It needs no solver, exporter, runtime, or Niro helpers.
Validity is established during construction, without filtering or asking ONNX
to repair graphs. Callers should treat checker failures as test failures.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import onnx
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

OPERATORS = ("Add", "Mul", "MatMul", "Transpose", "Identity", "Relu")
ELEMENT_TYPES = (
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.INT32,
    onnx.TensorProto.INT64,
    onnx.TensorProto.BOOL,
)
OPSET_VERSION = 14


@dataclass(frozen=True)
class _Value:
    """An available tensor with a unique name and concrete type and shape."""

    name: str
    element_type: int
    shape: tuple[int, ...]


def tensor_shapes(
    *, max_rank: int = 3, max_dim: int = 4
) -> SearchStrategy[tuple[int, ...]]:
    """Generate concrete shapes, including scalars and zero-length dimensions."""
    if min(max_rank, max_dim) < 0:
        raise ValueError("shape bounds must be nonnegative")
    return st.lists(st.integers(0, max_dim), max_size=max_rank).map(tuple)


def _value_info(value: _Value) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(
        value.name, value.element_type, value.shape
    )


def _literals(element_type: int) -> SearchStrategy[bool | int | float]:
    """Generate small, finite initializer elements representable in their type."""
    if element_type == onnx.TensorProto.BOOL:
        return st.booleans()
    if element_type in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
        return st.integers(-2, 2)
    width = 32 if element_type == onnx.TensorProto.FLOAT else 64
    return st.floats(-2, 2, allow_nan=False, allow_infinity=False, width=width)


def _broadcast_shape(
    lhs: tuple[int, ...], rhs: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Return the broadcast shape, or None for incompatible dimensions."""
    reversed_shape = []
    for left, right in itertools.zip_longest(reversed(lhs), reversed(rhs), fillvalue=1):
        if left == right or right == 1:
            reversed_shape.append(left)
        elif left == 1:
            reversed_shape.append(right)
        else:
            return None
    return tuple(reversed(reversed_shape))


def _matmul_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return a MatMul shape, including vector promotion and batch broadcasting."""
    if not lhs or not rhs:
        return None
    if lhs[-1] != (rhs[0] if len(rhs) == 1 else rhs[-2]):
        return None
    batch = _broadcast_shape(lhs[:-2], rhs[:-2])
    if batch is None:
        return None
    rows = () if len(lhs) == 1 else (lhs[-2],)
    columns = () if len(rhs) == 1 else (rhs[-1],)
    return batch + rows + columns


@st.composite
def models(
    draw: DrawFn,
    *,
    max_nodes: int = 12,
    max_inputs: int = 4,
    max_initializers: int = 3,
    max_outputs: int = 4,
    max_rank: int = 3,
    max_dim: int = 4,
    operators: Sequence[str] = OPERATORS,
    element_types: Sequence[int] = ELEMENT_TYPES,
) -> onnx.ModelProto:
    """Generate a fresh, typed, acyclic model using standard ONNX opset 14.

    For example, ``@given(onnx_strategies.models(max_nodes=5))`` supplies a
    ModelProto to a property test. Restrict ``operators`` and ``element_types``
    to subsets of OPERATORS and ELEMENT_TYPES to target particular features.

    Graphs may contain broadcasting, vector or batched MatMul, explicit or
    default Transpose permutations, shared operands, unused values, and several
    distinct outputs. Every intermediate has value_info. Outputs can directly
    reference inputs or initializers, and models may contain no nodes or inputs.

    Bounds are nonnegative, max_outputs must be positive, and at least one
    input or initializer must be allowed. Tensor ranks and dimensions stay
    within their bounds throughout the graph (at most max(1, max_dim ** max_rank)
    elements per tensor). Initializers are small and finite; arithmetic can
    still overflow. Generation stops early if no requested operator applies.

    This is a bounded tensor subset, not arbitrary ONNX: dynamic shapes,
    control flow, custom domains, functions, and external data are excluded.
    """
    if min(max_nodes, max_inputs, max_initializers, max_rank, max_dim) < 0:
        raise ValueError("model generation bounds must be nonnegative")
    if max_outputs < 1:
        raise ValueError("max_outputs must be positive")
    if max_inputs + max_initializers == 0:
        raise ValueError("at least one input or initializer must be allowed")
    if set(operators) - set(OPERATORS):
        raise ValueError("unsupported operators")
    if not element_types or set(element_types) - set(ELEMENT_TYPES):
        raise ValueError("element_types must be a nonempty supported subset")

    input_count = draw(st.integers(0 if max_initializers else 1, max_inputs))
    initializer_count = draw(st.integers(0 if input_count else 1, max_initializers))
    shapes = tensor_shapes(max_rank=max_rank, max_dim=max_dim)
    values = [
        _Value(f"v{index}", draw(st.sampled_from(element_types)), draw(shapes))
        for index in range(input_count + initializer_count)
    ]
    inputs = [_value_info(value) for value in values[:input_count]]
    initializers = []
    for value in values[input_count:]:
        size = math.prod(value.shape)
        elements = draw(
            st.lists(_literals(value.element_type), min_size=size, max_size=size)
        )
        initializers.append(
            onnx.helper.make_tensor(
                value.name, value.element_type, value.shape, elements
            )
        )

    nodes: list[onnx.NodeProto] = []
    intermediates: list[onnx.ValueInfoProto] = []
    for index in range(draw(st.integers(0, max_nodes))):
        numeric = [
            value for value in values if value.element_type != onnx.TensorProto.BOOL
        ]
        matmuls = [
            (lhs, rhs, shape)
            for lhs in numeric
            for rhs in numeric
            if "MatMul" in operators
            and lhs.element_type == rhs.element_type
            and (shape := _matmul_shape(lhs.shape, rhs.shape)) is not None
        ]
        available = [
            operator
            for operator in operators
            if operator in ("Identity", "Transpose")
            or (operator in ("Add", "Mul", "Relu") and numeric)
            or (operator == "MatMul" and matmuls)
        ]
        if not available:
            break
        operator = draw(st.sampled_from(available))
        attributes: list[onnx.AttributeProto] = []
        if operator == "MatMul":
            lhs, rhs, shape = draw(st.sampled_from(matmuls))
            operands = (lhs, rhs)
        elif operator in ("Add", "Mul"):
            lhs = draw(st.sampled_from(numeric))
            compatible = [
                (rhs, shape)
                for rhs in numeric
                if lhs.element_type == rhs.element_type
                and (shape := _broadcast_shape(lhs.shape, rhs.shape)) is not None
            ]
            rhs, shape = draw(st.sampled_from(compatible))
            operands = (lhs, rhs)
        else:
            lhs = draw(st.sampled_from(numeric if operator == "Relu" else values))
            operands = (lhs,)
            shape = lhs.shape
            if operator == "Transpose":
                permutation = tuple(reversed(range(len(shape))))
                if draw(st.booleans()):
                    permutation = tuple(draw(st.permutations(tuple(range(len(shape))))))
                    attributes.append(
                        onnx.helper.make_attribute(
                            "perm", permutation, attr_type=onnx.AttributeProto.INTS
                        )
                    )
                shape = tuple(shape[axis] for axis in permutation)
        result = _Value(f"v{len(values)}", lhs.element_type, shape)
        node = onnx.helper.make_node(
            operator,
            [value.name for value in operands],
            [result.name],
            name=f"node{index}",
        )
        node.attribute.extend(attributes)
        nodes.append(node)
        intermediates.append(_value_info(result))
        values.append(result)

    outputs = draw(
        st.lists(
            st.sampled_from(values),
            min_size=1,
            max_size=min(max_outputs, len(values)),
            unique=True,
        )
    )
    graph = onnx.helper.make_graph(
        nodes,
        "main",
        inputs,
        [_value_info(value) for value in outputs],
        initializer=initializers,
        value_info=intermediates,
    )
    return onnx.helper.make_model(
        graph,
        ir_version=7,
        opset_imports=[onnx.helper.make_opsetid("", OPSET_VERSION)],
        producer_name="niro.property",
    )
