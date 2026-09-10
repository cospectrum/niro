"""Operator-specific rules and factories for common tensor semantics.

Add a default rule to OPERATORS, or pass an Operator directly to models().
Unrelated signatures need only implement the Context-to-NodeSpec contract;
graph assembly has no operator-specific branches.
"""

import itertools
from collections.abc import Sequence

import onnx
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from ._core import (
    ELEMENT_TYPES,
    OPSET_VERSION,
    Context,
    NodeSpec,
    Operator,
    Value,
    tensor_shape,
)

NUMERIC_TYPES = tuple(
    element_type
    for element_type in ELEMENT_TYPES
    if element_type != onnx.TensorProto.BOOL
)


def _check_opset(context: Context) -> None:
    """Reject standard-domain versions older than these rules implement."""
    if context.opsets.get("", 0) < OPSET_VERSION:
        raise ValueError(
            f"built-in operator rules require ONNX opset >= {OPSET_VERSION}"
        )


def _tensors(context: Context, element_types: Sequence[int]) -> tuple[Value, ...]:
    """Select concrete tensors of the requested types, ignoring other values."""
    return tuple(
        value
        for value in context.values
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type in element_types
        and value.type.tensor_type.HasField("shape")
        and all(
            dimension.HasField("dim_value")
            for dimension in value.type.tensor_type.shape.dim
        )
    )


def unary(op_type: str, *, element_types: Sequence[int] = NUMERIC_TYPES) -> Operator:
    """Build a rule for one tensor input and an unchanged output type and shape.

    The operator must need no attributes and accept every value of the supplied
    element types, including scalar and empty tensors. Callers are responsible
    for these semantic assumptions; ONNX schemas are not generation rules.
    These helpers target standard ONNX opsets 14 and newer.
    """
    types = tuple(element_types)

    def strategy(context: Context) -> SearchStrategy[NodeSpec] | None:
        """Choose an available tensor accepted by this unary operator."""
        _check_opset(context)
        values = _tensors(context, types)
        if not values:
            return None
        return st.sampled_from(values).map(
            lambda value: NodeSpec((value.name,), (value.type,))
        )

    return Operator(op_type, strategy)


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


def broadcast_binary(
    op_type: str,
    *,
    element_types: Sequence[int] = NUMERIC_TYPES,
    output_element_type: int | None = None,
) -> Operator:
    """Build a rule for two same-type tensors with multidirectional broadcasting.

    The output has the broadcast shape and either the operands' element type
    or output_element_type, such as BOOL for comparisons. The operator must
    require no attributes or further value constraints, and accept all supplied
    element types. For example, Div needs a specialized rule to avoid integer
    division by zero; it does not satisfy this helper's assumptions.
    """
    types = tuple(element_types)

    def strategy(context: Context) -> SearchStrategy[NodeSpec] | None:
        """Choose compatible operands and declare the exact broadcast result."""
        _check_opset(context)
        values = _tensors(context, types)
        candidates = [
            (lhs, rhs, shape)
            for lhs in values
            for rhs in values
            if lhs.type.tensor_type.elem_type == rhs.type.tensor_type.elem_type
            and (shape := _broadcast_shape(tensor_shape(lhs), tensor_shape(rhs)))
            is not None
        ]
        if not candidates:
            return None

        def specification(candidate: tuple[Value, Value, tuple[int, ...]]) -> NodeSpec:
            """Materialize the known result type for a compatible operand pair."""
            lhs, rhs, shape = candidate
            element_type = (
                lhs.type.tensor_type.elem_type
                if output_element_type is None
                else output_element_type
            )
            return NodeSpec(
                (lhs.name, rhs.name),
                (onnx.helper.make_tensor_type_proto(element_type, shape),),
            )

        return st.sampled_from(candidates).map(specification)

    return Operator(op_type, strategy)


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


def _matmul(context: Context) -> SearchStrategy[NodeSpec] | None:
    """Choose concrete operands with compatible contraction and batch axes."""
    _check_opset(context)
    values = _tensors(context, NUMERIC_TYPES)
    candidates = [
        (lhs, rhs, shape)
        for lhs in values
        for rhs in values
        if lhs.type.tensor_type.elem_type == rhs.type.tensor_type.elem_type
        and (shape := _matmul_shape(tensor_shape(lhs), tensor_shape(rhs))) is not None
    ]
    if not candidates:
        return None

    def specification(candidate: tuple[Value, Value, tuple[int, ...]]) -> NodeSpec:
        """Materialize a MatMul with its exact tensor result signature."""
        lhs, rhs, shape = candidate
        return NodeSpec(
            (lhs.name, rhs.name),
            (
                onnx.helper.make_tensor_type_proto(
                    lhs.type.tensor_type.elem_type, shape
                ),
            ),
        )

    return st.sampled_from(candidates).map(specification)


@st.composite
def _transposes(draw: DrawFn, values: tuple[Value, ...]) -> NodeSpec:
    """Draw a transpose with an explicit permutation or default reversed axes."""
    value = draw(st.sampled_from(values))
    shape = tensor_shape(value)
    permutation = tuple(reversed(range(len(shape))))
    attributes: tuple[onnx.AttributeProto, ...] = ()
    if draw(st.booleans()):
        permutation = tuple(draw(st.permutations(tuple(range(len(shape))))))
        attributes = (
            onnx.helper.make_attribute(
                "perm", permutation, attr_type=onnx.AttributeProto.INTS
            ),
        )
    result_shape = tuple(shape[axis] for axis in permutation)
    return NodeSpec(
        (value.name,),
        (
            onnx.helper.make_tensor_type_proto(
                value.type.tensor_type.elem_type, result_shape
            ),
        ),
        attributes,
    )


def _transpose(context: Context) -> SearchStrategy[NodeSpec] | None:
    """Select concrete tensor operands for axis permutations."""
    _check_opset(context)
    values = _tensors(context, ELEMENT_TYPES)
    if not values:
        return None
    return _transposes(values)


OPERATORS: dict[str, Operator] = {
    "Add": broadcast_binary("Add"),
    "Mul": broadcast_binary("Mul"),
    "MatMul": Operator("MatMul", _matmul),
    "Transpose": Operator("Transpose", _transpose),
    "Identity": unary("Identity", element_types=ELEMENT_TYPES),
    "Relu": unary("Relu"),
}
