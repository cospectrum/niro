import pytest

from niro import ir


def test_supply_allocates_distinct_typed_values() -> None:
    supply = ir.ValueSupply(7)
    first = supply.fresh(ir.ScalarType.I32)
    second = supply.fresh(ir.TensorType(ir.ScalarType.F32, (2, 3)))
    assert first == ir.Value(ir.ValueId(7), ir.ScalarType.I32)
    assert second == ir.Value(ir.ValueId(8), ir.TensorType(ir.ScalarType.F32, (2, 3)))
    assert supply.next_id == 9


def test_supply_clone_is_an_independent_snapshot() -> None:
    supply = ir.ValueSupply()
    supply.fresh(ir.ScalarType.I32)
    trial = supply.clone()
    trial.fresh(ir.ScalarType.I32)
    trial.fresh(ir.ScalarType.I32)
    assert supply.next_id == 1
    assert trial.next_id == 3
    assert supply.fresh(ir.ScalarType.I32).id == 1
    assert trial.next_id == 3


def test_supply_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ir.ValueSupply(-1)
