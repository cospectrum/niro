"""Check generated graphs against ONNX's checker and reference execution."""

import math
from collections.abc import Iterator

import hypothesis
import numpy as np
import onnx
import onnx.reference
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from . import onnx as onnx_st


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_if_generation_reaches_nested_depth(depth: int) -> None:
    model = hypothesis.find(
        onnx_st.models(
            operators=(onnx_st.OPERATORS.if_,),
            min_nodes=1,
            max_nodes=1,
            max_inputs=1,
            max_initializers=0,
            max_rank=0,
            element_types=(onnx.TensorProto.BOOL,),
            max_depth=depth,
            max_branch_nodes=16,
        ),
        lambda model: _if_depth(model.graph) == depth,
        settings=hypothesis.settings(max_examples=500, deadline=None),
    )
    assert_valid_and_executable(model)


def test_if_branches_reuse_custom_operator_rules() -> None:
    model = hypothesis.find(
        onnx_st.models(
            operators=(onnx_st.OPERATORS.if_, onnx_st.unary("Neg")),
            min_nodes=1,
            max_nodes=1,
            max_seed_initializers=0,
        ),
        lambda model: any(
            node.op_type == "Neg"
            for graph in _subgraphs(model.graph)
            for node in graph.node
        ),
        settings=hypothesis.settings(max_examples=500, deadline=None),
    )
    assert model.graph.node[0].op_type == "If"
    assert_valid_and_executable(model)


@hypothesis.given(
    data=st.data(),
    max_depth=st.integers(0, 3),
    max_branch_nodes=st.integers(1, 9),
)
def test_recursive_graph_bounds(
    data: st.DataObject,
    max_depth: int,
    max_branch_nodes: int,
) -> None:
    model = data.draw(
        onnx_st.models(
            max_nodes=4,
            max_depth=max_depth,
            max_branch_nodes=max_branch_nodes,
            max_initializers=2,
            max_seed_initializers=0,
        )
    )
    assert _if_depth(model.graph) <= max_depth
    for graph in _subgraphs(model.graph):
        assert (
            sum(len(descendant.node) for descendant in (graph, *_subgraphs(graph)))
            <= max_branch_nodes
        )
        assert len(graph.initializer) <= 2
        assert not graph.input
        for value in (*graph.value_info, *graph.output):
            shape = value.type.tensor_type.shape.dim
            assert len(shape) <= 3
            assert all(0 <= dimension.dim_value <= 4 for dimension in shape)
    assert_valid_and_executable(model)


@pytest.mark.parametrize("initializer_slots", (0, 1))
@hypothesis.given(data=st.data())
def test_if_branches(data: st.DataObject, initializer_slots: int) -> None:
    tensor_type = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, (2,))
    bool_type = onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, ())
    values = (onnx_st.Value("left", tensor_type), onnx_st.Value("right", tensor_type))
    if initializer_slots == 0:
        values += (onnx_st.Value("condition", bool_type),)
    context = onnx_st.Context(
        values=values,
        limits=onnx_st.Limits(1, 2, onnx_st.ELEMENT_TYPES),
        initializer_slots=initializer_slots,
        opsets={"": 14},
        name_prefix="if_test",
    )
    strategy = onnx_st.OPERATORS.if_.strategy(context)
    assert strategy is not None
    spec = data.draw(strategy)
    condition = spec.inputs[0]
    initializers = []
    if isinstance(condition, onnx.TensorProto):
        initializer = onnx.TensorProto()
        initializer.CopyFrom(condition)
        initializer.name = "condition"
        initializers.append(initializer)
        condition = initializer.name
    assert isinstance(condition, str)
    node = onnx.helper.make_node("If", [condition], ["result"])
    node.attribute.extend(spec.attributes)
    output_type = spec.outputs[0]
    assert output_type is not None
    graph = onnx.helper.make_graph(
        [node],
        "if_test",
        [onnx.helper.make_value_info(value.name, value.type) for value in values],
        [onnx.helper.make_value_info("result", output_type)],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 14)]
    )
    onnx.checker.check_model(model, full_check=True)
    for predicate in (False, True):
        feeds = {
            name: onnx.numpy_helper.to_array(
                onnx.helper.make_tensor(name, elem_type, shape, contents)
            )
            for name, elem_type, shape, contents in (
                ("left", onnx.TensorProto.FLOAT, (2,), (1, 2)),
                ("right", onnx.TensorProto.FLOAT, (2,), (3, 4)),
                ("condition", onnx.TensorProto.BOOL, (), (predicate,)),
            )
        }
        if initializers:
            predicate = bool(onnx.numpy_helper.to_array(initializers[0]).item())
            del feeds["condition"]
        branch_name = "then_branch" if predicate else "else_branch"
        branch = next(
            attribute.g
            for attribute in spec.attributes
            if attribute.name == branch_name
        )
        expected = feeds[branch.node[0].input[0]]
        results = onnx.reference.ReferenceEvaluator(model).run(None, feeds)
        assert isinstance(results, list)
        result = results[0]
        assert (result == expected).all()


@hypothesis.given(onnx_st.models())
def test_models_are_valid_and_executable(model: onnx.ModelProto) -> None:
    assert_valid_and_executable(model)
    hypothesis.event(f"nodes={len(model.graph.node)}")
    for operator in {node.op_type for node in model.graph.node}:
        hypothesis.event(f"operator={operator}")


@pytest.mark.parametrize("operator", [rule.op_type for rule in onnx_st.OPERATORS])
@hypothesis.settings(max_examples=40)
@hypothesis.given(data=st.data())
def test_operator_subsets(operator: str, data: st.DataObject) -> None:
    model = data.draw(onnx_st.models(operators=(operator,)))
    assert {node.op_type for node in model.graph.node} <= {operator}
    assert_valid_and_executable(model)


@pytest.mark.parametrize(
    "operator",
    [
        onnx_st.unary("Neg"),
        onnx_st.broadcast_binary("Sub"),
        onnx_st.broadcast_binary("Less", output_element_type=onnx.TensorProto.BOOL),
    ],
    ids=("unary", "binary", "comparison"),
)
@hypothesis.settings(max_examples=40)
@hypothesis.given(data=st.data())
def test_operator_family_extensions(
    operator: onnx_st.Operator, data: st.DataObject
) -> None:
    model = data.draw(
        onnx_st.models(
            operators=(operator,),
            min_nodes=1,
            max_nodes=5,
            max_initializers=0,
            element_types=(onnx.TensorProto.FLOAT, onnx.TensorProto.INT32),
        )
    )
    assert 1 <= len(model.graph.node) <= 5
    assert {node.op_type for node in model.graph.node} == {operator.op_type}
    assert_valid_and_executable(model)


@hypothesis.given(
    data=st.data(),
    max_nodes=st.integers(0, 5),
    max_inputs=st.integers(0, 3),
    max_initializers=st.integers(0, 3),
    max_outputs=st.integers(1, 3),
    max_rank=st.integers(0, 3),
    max_dim=st.integers(0, 3),
    element_type=st.sampled_from(onnx_st.ELEMENT_TYPES),
)
def test_generation_bounds(
    data: st.DataObject,
    max_nodes: int,
    max_inputs: int,
    max_initializers: int,
    max_outputs: int,
    max_rank: int,
    max_dim: int,
    element_type: int,
) -> None:
    if max_inputs + max_initializers == 0:
        max_inputs = 1
    model = data.draw(
        onnx_st.models(
            max_nodes=max_nodes,
            max_inputs=max_inputs,
            max_initializers=max_initializers,
            max_outputs=max_outputs,
            max_rank=max_rank,
            max_dim=max_dim,
            element_types=(element_type,),
        )
    )
    assert len(model.graph.node) <= max_nodes
    assert len(model.graph.input) <= max_inputs
    assert len(model.graph.initializer) <= max_initializers
    assert 1 <= len(model.graph.output) <= max_outputs
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        type_ = value.type.tensor_type
        assert type_.elem_type == element_type
        assert len(type_.shape.dim) <= max_rank
        assert all(0 <= dimension.dim_value <= max_dim for dimension in type_.shape.dim)
    for initializer in model.graph.initializer:
        assert initializer.data_type == element_type
        assert len(initializer.dims) <= max_rank
        assert all(0 <= dimension <= max_dim for dimension in initializer.dims)
    assert_valid_and_executable(model)


@hypothesis.given(onnx_st.models(operators=()))
def test_models_without_nodes(model: onnx.ModelProto) -> None:
    assert not model.graph.node
    assert_valid_and_executable(model)


@pytest.mark.parametrize(
    "strategy",
    [
        onnx_st.models(max_nodes=-1),
        onnx_st.models(max_depth=-1),
        onnx_st.models(max_branch_nodes=0),
        onnx_st.models(max_inputs=-1),
        onnx_st.models(max_initializers=-1),
        onnx_st.models(max_outputs=0),
        onnx_st.models(max_rank=-1),
        onnx_st.models(max_dim=-1),
        onnx_st.models(min_nodes=-1),
        onnx_st.models(min_nodes=2, max_nodes=1),
        onnx_st.models(max_seed_initializers=-1),
        onnx_st.models(max_seed_initializers=4, max_initializers=3),
    ],
)
@hypothesis.given(data=st.data())
def test_invalid_bounds(
    strategy: SearchStrategy[onnx.ModelProto], data: st.DataObject
) -> None:
    with pytest.raises(ValueError):
        data.draw(strategy)


@hypothesis.given(data=st.data())
def test_invalid_subsets_and_missing_sources(data: st.DataObject) -> None:
    with pytest.raises(ValueError, match="at least one input or initializer"):
        data.draw(onnx_st.models(max_inputs=0, max_initializers=0))
    with pytest.raises(ValueError, match="unsupported operators"):
        data.draw(onnx_st.models(operators=("Missing",)))
    with pytest.raises(ValueError, match="element_types"):
        data.draw(onnx_st.models(element_types=()))
    with pytest.raises(ValueError, match="element_types"):
        data.draw(onnx_st.models(element_types=(onnx.TensorProto.STRING,)))


@hypothesis.given(data=st.data())
def test_unreachable_minimum_nodes(data: st.DataObject) -> None:
    with pytest.raises(ValueError, match="cannot satisfy min_nodes"):
        data.draw(onnx_st.models(operators=(), min_nodes=1))


@hypothesis.given(data=st.data())
def test_missing_domain_import(data: st.DataObject) -> None:
    operator = onnx_st.Operator("Example", lambda _: None, domain="example")
    with pytest.raises(ValueError, match="missing opset import"):
        data.draw(onnx_st.models(operators=(operator,)))


@hypothesis.given(data=st.data(), version=st.sampled_from((14, 15, 18, 21)))
def test_opsets_determine_ir_version(data: st.DataObject, version: int) -> None:
    model = data.draw(
        onnx_st.models(
            operators=("Identity",),
            min_nodes=1,
            max_nodes=2,
            opsets={"": version},
        )
    )
    assert model.ir_version == onnx.helper.find_min_ir_version_for(model.opset_import)
    assert_valid_and_executable(model)


@hypothesis.given(data=st.data())
def test_rejects_incompatible_versions(data: st.DataObject) -> None:
    with pytest.raises(ValueError, match="versions must be positive"):
        data.draw(onnx_st.models(opsets={"": 0}))
    with pytest.raises(ValueError, match="require IR version"):
        data.draw(onnx_st.models(opsets={"": 15}, ir_version=7))
    with pytest.raises(ValueError, match="require ONNX opset"):
        data.draw(onnx_st.models(opsets={"": 13}, min_nodes=1))
    with pytest.raises(ValueError, match="Unsupported opset-version"):
        data.draw(onnx_st.models(opsets={"": 999}))


def assert_valid_and_executable(model: onnx.ModelProto) -> None:
    """Check serialized validity and the actual type and shape of every result."""
    restored = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(restored, full_check=True)
    feeds = {}
    for value in restored.graph.input:
        type_ = value.type.tensor_type
        shape = tuple(dimension.dim_value for dimension in type_.shape.dim)
        elements = [index % 5 - 2 for index in range(math.prod(shape))]
        tensor = onnx.helper.make_tensor(value.name, type_.elem_type, shape, elements)
        feeds[value.name] = onnx.numpy_helper.to_array(tensor)

    _assert_graph_executable(
        restored.graph,
        feeds,
        {opset.domain: opset.version for opset in restored.opset_import},
    )


def _assert_graph_executable(
    graph: onnx.GraphProto,
    feeds: dict[str, np.ndarray],
    opsets: dict[str, int],
) -> None:
    """Execute every branch, including paths skipped by its enclosing condition."""
    evaluator = onnx.reference.ReferenceEvaluator(graph, opsets=opsets)
    results = evaluator.run(None, feeds, intermediate=True)
    assert isinstance(results, dict)
    for value in (*graph.value_info, *graph.output):
        type_ = value.type.tensor_type
        result = results[value.name]
        assert result.shape == tuple(
            dimension.dim_value for dimension in type_.shape.dim
        )
        assert result.dtype == onnx.helper.tensor_dtype_to_np_dtype(type_.elem_type)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _assert_graph_executable(attribute.g, results, opsets)


def _subgraphs(graph: onnx.GraphProto) -> Iterator[onnx.GraphProto]:
    """Visit child graphs recursively."""
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield attribute.g
                yield from _subgraphs(attribute.g)


def _if_depth(graph: onnx.GraphProto) -> int:
    """Return the maximum number of nested If nodes."""
    return max(
        (
            1
            + max(
                (
                    _if_depth(attribute.g)
                    for attribute in node.attribute
                    if attribute.type == onnx.AttributeProto.GRAPH
                ),
                default=0,
            )
            for node in graph.node
            if node.op_type == "If"
        ),
        default=0,
    )
