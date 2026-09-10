"""Exercise operator extensions through ONNX validation and execution."""

from collections.abc import Callable

import hypothesis
import onnx
import onnx.reference
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from . import onnx as onnx_st
from .test_onnx import assert_valid_and_executable


@pytest.mark.parametrize(
    ("operator_factory", "expected"),
    [
        (
            lambda: onnx_st.Operator("TopK", _top_k),
            [[2, 1, 5, 4], [2, 1, 2, 1]],
        ),
        (lambda: onnx_st.Operator("Clip", _clip), [[0, 1, 2, 2, 2, 2]]),
        (lambda: onnx_st.Operator("Dropout", _dropout), [[0, 1, 2, 3, 4, 5]]),
        (lambda: onnx_st.Operator("If", _if), [[0, 1, 2, 3, 4, 5]]),
        (
            lambda: onnx_st.Operator("Normalizer", _normalizer, domain="ai.onnx.ml"),
            [[0, 0.5, 1, 0.6, 0.8, 1]],
        ),
    ],
    ids=(
        "multiple_outputs",
        "optional_input",
        "optional_output",
        "subgraphs",
        "domain",
    ),
)
@hypothesis.settings(max_examples=10)
@hypothesis.given(data=st.data())
def test_tensor_operator_extensions(
    operator_factory: Callable[[], onnx_st.Operator],
    expected: list[list[float]],
    data: st.DataObject,
) -> None:
    operator = operator_factory()
    opsets = {"": 14, **({operator.domain: 3} if operator.domain else {})}
    model = data.draw(
        onnx_st.models(
            operators=(onnx_st.Operator("Constant", _source), operator),
            min_nodes=2,
            max_nodes=2,
            max_inputs=0,
            max_initializers=1,
            max_seed_initializers=0,
            max_rank=2,
            max_dim=3,
            opsets=opsets,
        )
    )
    assert_valid_and_executable(model)
    assert {import_.domain: import_.version for import_ in model.opset_import} == opsets
    node = model.graph.node[-1]
    assert node.op_type == operator.op_type
    assert node.domain == operator.domain
    if operator.op_type == "Clip":
        assert node.input[1] == ""
    if operator.op_type == "Dropout":
        assert node.output[1] == ""
    results = onnx.reference.ReferenceEvaluator(model).run(
        [name for name in node.output if name], {}
    )
    assert isinstance(results, list)
    for result, elements in zip(results, expected, strict=True):
        assert result.reshape(-1).tolist() == pytest.approx(elements)


@hypothesis.settings(max_examples=10)
@hypothesis.given(data=st.data())
def test_constant_can_seed_graph_without_inputs(data: st.DataObject) -> None:
    model = data.draw(
        onnx_st.models(
            operators=(onnx_st.Operator("Constant", _source),),
            min_nodes=1,
            max_nodes=1,
            max_inputs=0,
            max_initializers=0,
            max_rank=2,
            max_dim=3,
        )
    )
    assert not model.graph.input
    assert not model.graph.initializer
    assert model.graph.node[0].op_type == "Constant"
    assert_valid_and_executable(model)


@hypothesis.settings(max_examples=10)
@hypothesis.given(data=st.data())
def test_sequence_values_can_feed_later_nodes(data: st.DataObject) -> None:
    model = data.draw(
        onnx_st.models(
            operators=(
                onnx_st.Operator("Constant", _source),
                onnx_st.Operator("SequenceConstruct", _sequence_construct),
                onnx_st.Operator("SequenceLength", _sequence_length),
            ),
            min_nodes=3,
            max_nodes=3,
            max_inputs=0,
            max_initializers=0,
            max_rank=2,
            max_dim=3,
        )
    )
    restored = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(restored, full_check=True)
    assert [node.op_type for node in restored.graph.node] == [
        "Constant",
        "SequenceConstruct",
        "SequenceLength",
    ]
    sequence_node, length_node = restored.graph.node[1:]
    results = onnx.reference.ReferenceEvaluator(restored).run(
        [sequence_node.output[0], length_node.output[0]], {}
    )
    assert isinstance(results, list)
    sequence, length = results
    assert len(sequence) == 2
    assert all(value.shape == (2, 3) for value in sequence)
    assert length.shape == ()
    assert length.dtype == onnx.helper.tensor_dtype_to_np_dtype(onnx.TensorProto.INT64)
    assert length.item() == 2


@hypothesis.settings(max_examples=10)
@hypothesis.given(data=st.data())
def test_reused_rule_protobufs_and_models_are_independent(data: st.DataObject) -> None:
    source_type = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, (2, 3))
    source_attribute = onnx.helper.make_attribute(
        "value",
        onnx.helper.make_tensor("source", onnx.TensorProto.FLOAT, (2, 3), range(6)),
    )
    source_spec = onnx_st.NodeSpec(
        inputs=(), outputs=(source_type,), attributes=(source_attribute,)
    )
    k = onnx.helper.make_tensor("reusable_k", onnx.TensorProto.INT64, (1,), (2,))
    result_types = (
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, (2, 2)),
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, (2, 2)),
    )
    axis = onnx.helper.make_attribute("axis", -1)
    protobufs = (source_type, source_attribute, k, *result_types, axis)
    originals = tuple(value.SerializeToString() for value in protobufs)

    def source(
        context: onnx_st.Context,
    ) -> SearchStrategy[onnx_st.NodeSpec] | None:
        return None if context.values else st.just(source_spec)

    def top_k(
        context: onnx_st.Context,
    ) -> SearchStrategy[onnx_st.NodeSpec] | None:
        if len(context.values) != 1 or context.initializer_slots < 1:
            return None
        return st.just(
            onnx_st.NodeSpec(
                inputs=(context.values[0].name, k),
                outputs=result_types,
                attributes=(axis,),
            )
        )

    strategy = onnx_st.models(
        operators=(
            onnx_st.Operator("Constant", source),
            onnx_st.Operator("TopK", top_k),
        ),
        min_nodes=2,
        max_nodes=2,
        max_inputs=0,
        max_initializers=1,
        max_seed_initializers=0,
    )
    first = data.draw(strategy)
    assert_valid_and_executable(first)
    first.graph.initializer[0].int64_data[0] = 1
    first.graph.node[0].attribute[0].t.float_data[0] = 99
    first.graph.node[1].attribute[0].i = 0
    first.graph.value_info[0].type.tensor_type.elem_type = onnx.TensorProto.INT64

    second = data.draw(strategy)
    assert_valid_and_executable(second)
    assert second.graph.initializer[0].int64_data[0] == 2
    assert second.graph.node[0].attribute[0].t.float_data[0] == 0
    assert second.graph.node[1].attribute[0].i == -1
    assert tuple(value.SerializeToString() for value in protobufs) == originals


def _source(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if context.values or context.limits.max_rank < 2 or context.limits.max_dim < 3:
        return None
    value = onnx.helper.make_tensor(
        "", onnx.TensorProto.FLOAT, (2, 3), (0, 1, 2, 3, 4, 5)
    )
    return st.just(
        onnx_st.NodeSpec(
            inputs=(),
            outputs=(onnx.helper.make_tensor_type_proto(value.data_type, value.dims),),
            attributes=(onnx.helper.make_attribute("value", value),),
        )
    )


def _top_k(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1 or context.initializer_slots < 1:
        return None
    return st.just(
        onnx_st.NodeSpec(
            inputs=(
                context.values[0].name,
                onnx.helper.make_tensor("", onnx.TensorProto.INT64, (1,), (2,)),
            ),
            outputs=(
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, (2, 2)),
                onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, (2, 2)),
            ),
            attributes=(onnx.helper.make_attribute("axis", -1),),
        )
    )


def _clip(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1 or context.initializer_slots < 1:
        return None
    value = context.values[0]
    return st.just(
        onnx_st.NodeSpec(
            inputs=(
                value.name,
                None,
                onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, (), (2,)),
            ),
            outputs=(value.type,),
        )
    )


def _dropout(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1:
        return None
    value = context.values[0]
    return st.just(onnx_st.NodeSpec(inputs=(value.name,), outputs=(value.type, None)))


def _if(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1 or context.initializer_slots < 1:
        return None
    value = context.values[0]
    attributes = []
    for branch, operator in (("then_branch", "Identity"), ("else_branch", "Neg")):
        name = f"{context.name_prefix}_{branch}_result"
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node(operator, [value.name], [name])],
            branch,
            [],
            [onnx.helper.make_value_info(name, value.type)],
        )
        attributes.append(onnx.helper.make_attribute(branch, graph))
    return st.just(
        onnx_st.NodeSpec(
            inputs=(onnx.helper.make_tensor("", onnx.TensorProto.BOOL, (), (True,)),),
            outputs=(value.type,),
            attributes=tuple(attributes),
        )
    )


def _normalizer(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1:
        return None
    value = context.values[0]
    return st.just(
        onnx_st.NodeSpec(
            inputs=(value.name,),
            outputs=(value.type,),
            attributes=(onnx.helper.make_attribute("norm", "MAX"),),
        )
    )


def _sequence_construct(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 1:
        return None
    value = context.values[0]
    return st.just(
        onnx_st.NodeSpec(
            inputs=(value.name, value.name),
            outputs=(onnx.helper.make_sequence_type_proto(value.type),),
        )
    )


def _sequence_length(
    context: onnx_st.Context,
) -> SearchStrategy[onnx_st.NodeSpec] | None:
    if len(context.values) != 2:
        return None
    value = context.values[-1]
    assert value.type.HasField("sequence_type")
    return st.just(
        onnx_st.NodeSpec(
            inputs=(value.name,),
            outputs=(onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, ()),),
        )
    )
