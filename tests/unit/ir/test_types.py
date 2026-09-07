import pytest

from niro import ir


def test_rejects_invalid_tensor_dimension_at_construction() -> None:
    with pytest.raises(ValueError):
        ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, -1))


def test_rejects_negative_value_id_at_construction() -> None:
    with pytest.raises(ValueError):
        ir.Value(ir.ValueId(-1), ir.ScalarType.F32)
