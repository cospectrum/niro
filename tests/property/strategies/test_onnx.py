"""Check generated graphs against ONNX's checker and reference execution."""

import math

import hypothesis
import onnx
import onnx.reference
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from . import onnx as onnx_strategies


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

    declared = {
        value.name: value.type.tensor_type
        for value in (*restored.graph.value_info, *restored.graph.output)
    }
    evaluator = onnx.reference.ReferenceEvaluator(restored)
    results = evaluator.run(list(declared), feeds)
    assert isinstance(results, list)
    for type_, result in zip(declared.values(), results, strict=True):
        assert result.shape == tuple(
            dimension.dim_value for dimension in type_.shape.dim
        )
        assert result.dtype == onnx.helper.tensor_dtype_to_np_dtype(type_.elem_type)


@hypothesis.given(onnx_strategies.models())
def test_models_are_valid_and_executable(model: onnx.ModelProto) -> None:
    assert_valid_and_executable(model)
    hypothesis.event(f"nodes={len(model.graph.node)}")
    for operator in {node.op_type for node in model.graph.node}:
        hypothesis.event(f"operator={operator}")


@pytest.mark.parametrize("operator", onnx_strategies.OPERATORS)
@hypothesis.settings(max_examples=40)
@hypothesis.given(data=st.data())
def test_operator_subsets(operator: str, data: st.DataObject) -> None:
    model = data.draw(onnx_strategies.models(operators=(operator,)))
    assert {node.op_type for node in model.graph.node} <= {operator}
    assert_valid_and_executable(model)


@hypothesis.given(
    data=st.data(),
    max_nodes=st.integers(0, 5),
    max_inputs=st.integers(0, 3),
    max_initializers=st.integers(0, 3),
    max_outputs=st.integers(1, 3),
    max_rank=st.integers(0, 3),
    max_dim=st.integers(0, 3),
    element_type=st.sampled_from(onnx_strategies.ELEMENT_TYPES),
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
        onnx_strategies.models(
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


@hypothesis.given(onnx_strategies.models(operators=()))
def test_models_without_nodes(model: onnx.ModelProto) -> None:
    assert not model.graph.node
    assert_valid_and_executable(model)


@pytest.mark.parametrize(
    "strategy",
    [
        onnx_strategies.models(max_nodes=-1),
        onnx_strategies.models(max_inputs=-1),
        onnx_strategies.models(max_initializers=-1),
        onnx_strategies.models(max_outputs=0),
        onnx_strategies.models(max_rank=-1),
        onnx_strategies.models(max_dim=-1),
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
        data.draw(onnx_strategies.models(max_inputs=0, max_initializers=0))
    with pytest.raises(ValueError, match="unsupported operators"):
        data.draw(onnx_strategies.models(operators=("Missing",)))
    with pytest.raises(ValueError, match="element_types"):
        data.draw(onnx_strategies.models(element_types=()))
    with pytest.raises(ValueError, match="element_types"):
        data.draw(onnx_strategies.models(element_types=(onnx.TensorProto.STRING,)))
