import onnx

from niro.frontends.onnx import OnnxOpType


def test_matches_latest_onnx_schema_registry() -> None:
    registered_names = {schema.name for schema in onnx.defs.get_all_schemas()}

    assert {op_type.value for op_type in OnnxOpType} == registered_names


def test_members_are_sorted_alphabetically() -> None:
    assert list(OnnxOpType) == sorted(OnnxOpType, key=lambda op_type: op_type.value)


def test_member_names_preserve_onnx_spelling() -> None:
    assert all(op_type.name == op_type.value for op_type in OnnxOpType)
    assert OnnxOpType.MatMul == "MatMul"
    assert OnnxOpType.TopK == "TopK"
