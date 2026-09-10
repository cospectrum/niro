"""Contracts shared by ONNX operator rules and graph generation."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import onnx
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

ELEMENT_TYPES = (
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.INT32,
    onnx.TensorProto.INT64,
    onnx.TensorProto.BOOL,
)
OPSET_VERSION = 14


@dataclass(frozen=True)
class Value:
    """An available named value with its complete ONNX type.

    Rules must treat the protobuf type as read-only. Values need not be tensors;
    sequence, optional, and other ONNX types use the same interface.
    """

    name: str
    type: onnx.TypeProto


@dataclass(frozen=True)
class Limits:
    """Tensor shape bounds and the element types used for random sources."""

    max_rank: int
    max_dim: int
    element_types: tuple[int, ...]


@dataclass(frozen=True)
class Context:
    """Read-only graph state available to an operator strategy factory.

    initializer_slots bounds additional tensor inputs. name_prefix is unique
    to the current node and can prefix names inside graph attributes. Rules
    must not mutate available values or the opset mapping. max_depth bounds
    remaining control-flow nesting. subgraphs optionally builds bounded child
    graphs with requested output types and names, using the active operator rules.
    """

    values: tuple[Value, ...]
    limits: Limits
    initializer_slots: int
    opsets: Mapping[str, int]
    name_prefix: str
    max_depth: int = 1
    subgraphs: (
        Callable[[str, tuple[onnx.TypeProto, ...]], SearchStrategy[onnx.GraphProto]]
        | None
    ) = None


@dataclass(frozen=True)
class NodeSpec:
    """One valid operator application before the graph assigns result names.

    String inputs reference available values, tensor inputs request auxiliary
    initializers, and None inputs omit optional positions. Tensor input names
    are assigned by the graph generator. Each non-None output declares its
    complete ONNX type; None outputs omit optional positions. Attributes may
    contain any ONNX attribute type, including tensors and nested graphs.

    Rules establish operator-specific shape, type, attribute, and input-value
    constraints during construction. The graph generator does not infer or
    repair them. Treat every protobuf passed here as read-only.
    """

    inputs: tuple[str | onnx.TensorProto | None, ...]
    outputs: tuple[onnx.TypeProto | None, ...]
    attributes: tuple[onnx.AttributeProto, ...] = ()


@dataclass(frozen=True)
class Operator:
    """An operator identity and a factory for compatible node strategies.

    The factory returns None when the current graph has no valid application.
    Otherwise its strategy must construct valid nodes directly, without
    filtering, rejection, or checker calls. The factory can inspect active
    opsets and should reject versions whose semantics it does not implement.
    """

    op_type: str
    strategy: Callable[[Context], SearchStrategy[NodeSpec] | None]
    domain: str = ""


def tensor_shapes(
    *, max_rank: int = 3, max_dim: int = 4
) -> SearchStrategy[tuple[int, ...]]:
    """Generate concrete shapes, including scalars and zero-length dimensions."""
    if min(max_rank, max_dim) < 0:
        raise ValueError("shape bounds must be nonnegative")
    return st.lists(st.integers(0, max_dim), max_size=max_rank).map(tuple)


def tensor_shape(value: Value) -> tuple[int, ...]:
    """Return the shape of a value known to be a concrete, ranked tensor."""
    assert value.type.HasField("tensor_type")
    tensor = value.type.tensor_type
    assert tensor.HasField("shape")
    assert all(dimension.HasField("dim_value") for dimension in tensor.shape.dim)
    return tuple(dimension.dim_value for dimension in tensor.shape.dim)
