import pytest

from niro import ir
from niro.onnx.value_table import OnnxValueTable


def value(id: int) -> ir.Value:
    return ir.Value(ir.ValueId(id), ir.ScalarType.I32)


def test_defines_and_looks_up_value() -> None:
    table = OnnxValueTable()
    expected = value(0)

    table.define("x", expected)

    assert table.lookup("x") == expected


def test_rejects_redefinition() -> None:
    table = OnnxValueTable()
    table.define("x", value(0))

    with pytest.raises(ValueError, match="already defined"):
        table.define("x", value(1))


def test_define_many_stops_at_redefinition() -> None:
    table = OnnxValueTable()
    table.define("x", value(0))
    y = value(1)

    with pytest.raises(ValueError, match="already defined"):
        table.define_many(("y", "x"), (y, value(2)))

    assert table.lookup("y") == y


def test_rejects_unknown_value() -> None:
    table = OnnxValueTable()

    with pytest.raises(ValueError, match="unknown ONNX value: 'x'"):
        table.lookup("x")
