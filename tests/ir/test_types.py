import pytest

from niro import ir


def test_rejects_invalid_tensor_dimension_at_construction() -> None:
    with pytest.raises(ValueError, match="dimensions cannot be negative"):
        ir.TensorType(element_type=ir.ScalarType.F32, shape=(2, -1))
