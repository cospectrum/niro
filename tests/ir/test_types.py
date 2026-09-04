import pytest

from niro import ir


@pytest.mark.parametrize("shape, rank", [(None, None), ((), 0), ((2, None), 2)])
def test_tensor_rank(shape: ir.Shape | None, rank: int | None) -> None:
    assert ir.TensorType(ir.ScalarType.F32, shape).rank == rank
