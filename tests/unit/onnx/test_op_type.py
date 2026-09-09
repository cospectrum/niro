import onnx

from niro.onnx import op_type


def test_matches_latest_onnx_schema_registry() -> None:
    registered_names = {schema.name for schema in onnx.defs.get_all_schemas()}

    assert {op_type.value for op_type in op_type.OnnxOpType} == registered_names


def test_members_are_sorted_alphabetically() -> None:
    assert list(op_type.OnnxOpType) == sorted(
        op_type.OnnxOpType, key=lambda op_type: op_type.value
    )


def test_member_names_preserve_onnx_spelling() -> None:
    assert all(op_type.name == op_type.value for op_type in op_type.OnnxOpType)
    assert op_type.OnnxOpType.MatMul == "MatMul"
    assert op_type.OnnxOpType.TopK == "TopK"
