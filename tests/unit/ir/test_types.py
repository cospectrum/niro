import pytest

from niro import ir


def test_rejects_invalid_tensor_dimension_at_construction() -> None:
    with pytest.raises(ValueError):
        ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, -1))


def test_rejects_negative_value_id_at_construction() -> None:
    with pytest.raises(ValueError):
        ir.Value(ir.ValueId(-1), ir.ScalarType.F32)


@pytest.mark.parametrize("value_id", [0, 1, 100])
def test_accepts_nonnegative_value_ids(value_id: int) -> None:
    assert ir.Value(ir.ValueId(value_id), ir.ScalarType.F32).id == value_id


@pytest.mark.parametrize("shape", [None, (), (0, 3), (2, None)])
def test_accepts_tensor_shapes(shape: ir.Shape | None) -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, shape)
    assert tensor.shape == shape
    assert tensor.rank == (None if shape is None else len(shape))
