"""Import valid ONNX graphs and preserve their typed dataflow and attributes."""

import dataclasses
import math

import hypothesis
import onnx
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

import niro
from niro import verify
from tests.helpers.onnx import assert_niro_matches_onnx

from ..strategies import onnx as onnx_st


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
    assert_niro_matches_onnx(module, model.graph)
    for node in model.graph.node:
        hypothesis.event(f"operator={node.op_type}")


@hypothesis.given(data=st.data())
def test_import_preserves_if_graph(data: st.DataObject) -> None:
    model = data.draw(_nested_if_models())
    onnx.checker.check_model(model, full_check=True)
    original = model.SerializeToString()
    module = niro.from_onnx(model)
    assert model.SerializeToString() == original
    assert verify.module(module) is module
    assert_niro_matches_onnx(module, model.graph)


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


@st.composite
def _nested_if_models(draw: DrawFn) -> onnx.ModelProto:
    """Put generated If graphs inside another If to exercise nesting in every case."""
    model = draw(
        onnx_st.models(
            operators=(onnx_st.OPERATORS.if_,),
            min_nodes=1,
            max_nodes=4,
            max_outputs=1,
            max_initializers=3,
            max_seed_initializers=0,
        )
    )
    branch = model.graph
    inputs = list(branch.input)
    del branch.input[:]
    branch.value_info.add().CopyFrom(branch.output[0])
    branch.node.append(
        onnx.helper.make_node(
            "Identity", [branch.output[0].name], ["outer_then_result"]
        )
    )
    branch.output[0].name = "outer_then_result"
    type_ = branch.output[0].type
    tensor_type = type_.tensor_type
    shape = tuple(dimension.dim_value for dimension in tensor_type.shape.dim)
    other = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Identity", ["outer_else_value"], ["outer_else_result"]
            )
        ],
        "outer_else",
        [],
        [onnx.helper.make_value_info("outer_else_result", type_)],
        initializer=[
            onnx.helper.make_tensor(
                "outer_else_value", tensor_type.elem_type, shape, [0] * math.prod(shape)
            )
        ],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "If",
                ["outer_condition"],
                ["outer_result"],
                then_branch=branch,
                else_branch=other,
            )
        ],
        "main",
        [
            *inputs,
            onnx.helper.make_tensor_value_info(
                "outer_condition", onnx.TensorProto.BOOL, ()
            ),
        ],
        [onnx.helper.make_value_info("outer_result", type_)],
    )
    model.graph.CopyFrom(graph)
    return model
