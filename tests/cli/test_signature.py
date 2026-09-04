from pathlib import Path

import onnx
from onnx import TensorProto, helper
from typer.testing import CliRunner

from niro import ir
from niro.cli import app
from niro.cli.signature import format_signature

runner = CliRunner()


def model() -> onnx.ModelProto:
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [2])
    return helper.make_model(graph=helper.make_graph([], "identity", [value], [value]))


def function(
    inputs: tuple[ir.Type, ...],
    outputs: tuple[ir.Type, ...],
    input_names: tuple[str | None, ...] | None = None,
    output_names: tuple[str | None, ...] | None = None,
) -> ir.Function:
    return ir.Function(
        name="model",
        type=ir.FunctionType(inputs, outputs),
        input_names=input_names,
        output_names=output_names,
    )


def test_command_discovery_and_file_input(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(model(), path)

    root = runner.invoke(app)
    inspect = runner.invoke(app, ["inspect"])
    result = runner.invoke(app, ["inspect", "signature", str(path)])

    assert "inspect" in root.stdout
    assert "signature" in inspect.stdout
    assert result.exit_code == 0
    assert result.stdout == "identity(value: tensor<2xf32>) -> (value: tensor<2xf32>)\n"


def test_stdin_and_input_errors(tmp_path: Path) -> None:
    data = model().SerializeToString()
    implicit = runner.invoke(
        app, ["inspect", "signature", "--input-format", "onnx"], input=data
    )
    explicit = runner.invoke(
        app, ["inspect", "signature", "--input-format", "onnx", "-"], input=data
    )
    unknown = runner.invoke(app, ["inspect", "signature", str(tmp_path / "model.bin")])
    malformed = runner.invoke(
        app,
        ["inspect", "signature", "--input-format", "onnx"],
        input=b"not onnx",
    )

    assert implicit.exit_code == explicit.exit_code == 0
    assert implicit.stdout == explicit.stdout
    assert unknown.exit_code != 0
    assert "cannot infer" in unknown.stderr
    assert malformed.exit_code == 1
    assert "Error:" in malformed.stderr


def test_formats_types_and_partial_names() -> None:
    formatted = format_signature(
        function(
            (
                ir.ScalarType.F32,
                ir.TensorType(ir.ScalarType.I64, (2, None)),
                ir.TensorType(ir.ScalarType.F32, ()),
                ir.TensorType(ir.ScalarType.F32, None),
            ),
            (ir.ScalarType.I32,),
            ("scalar", None, "rank_zero", "unranked"),
            (None,),
        )
    )
    assert formatted == (
        "model(scalar: f32, tensor<2x?xi64>, rank_zero: tensor<f32>, "
        "unranked: tensor<*xf32>) -> (i32)"
    )


def test_wraps_outputs_only() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (1, 128000, 123456789))
    formatted = format_signature(
        function(
            (ir.ScalarType.F32,), (tensor, tensor), ("x",), ("logits", "hidden_state")
        )
    )
    assert formatted == (
        "model(x: f32) -> (\n"
        "  logits: tensor<1x128000x123456789xf32>,\n"
        "  hidden_state: tensor<1x128000x123456789xf32>,\n"
        ")"
    )


def test_wraps_inputs_with_trailing_commas_and_zero_results() -> None:
    tensor = ir.TensorType(ir.ScalarType.F32, (1, 3, 224, 224))
    formatted = format_signature(
        function(
            (tensor, tensor, tensor), (), ("first_input", "second_input", "third_input")
        )
    )
    assert formatted == (
        "model(\n"
        "  first_input: tensor<1x3x224x224xf32>,\n"
        "  second_input: tensor<1x3x224x224xf32>,\n"
        "  third_input: tensor<1x3x224x224xf32>,\n"
        ") -> ()"
    )


def test_one_line_at_exact_boundary() -> None:
    name = "m" * 89
    formatted = format_signature(
        ir.Function(name, ir.FunctionType((ir.ScalarType.F32,), ()))
    )
    assert len(formatted) == 100
    assert "\n" not in formatted
