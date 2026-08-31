from pathlib import Path

import onnx
from onnx import TensorProto, helper
from typer.testing import CliRunner

from niro.cli import app

runner = CliRunner()


def model() -> onnx.ModelProto:
    value = helper.make_tensor_value_info(
        name="value",
        elem_type=TensorProto.FLOAT,
        shape=[2],
    )
    graph = helper.make_graph(
        nodes=[],
        name="identity",
        inputs=[value],
        outputs=[value],
    )
    return helper.make_model(graph=graph)


def test_root_and_emit_show_help() -> None:
    root = runner.invoke(app)
    emit = runner.invoke(app, ["emit"])

    assert "emit" in root.stdout
    assert "mlir" in emit.stdout


def test_emits_mlir_to_stdout_from_onnx_file(tmp_path: Path) -> None:
    input_path = tmp_path / "model.ONNX"
    onnx.save(model(), input_path)

    result = runner.invoke(app, ["emit", "mlir", str(input_path)])

    assert result.exit_code == 0
    assert "builtin.module" in result.stdout
    assert "func.func public @identity" in result.stdout


def test_emits_mlir_to_file(tmp_path: Path) -> None:
    input_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.mlir"
    onnx.save(model(), input_path)

    result = runner.invoke(
        app,
        ["emit", "mlir", str(input_path), "-o", str(output_path)],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "func.func public @identity" in output_path.read_text()


def test_reads_onnx_from_implicit_or_explicit_stdin() -> None:
    data = model().SerializeToString()

    implicit = runner.invoke(
        app,
        ["emit", "mlir", "--input-format", "onnx"],
        input=data,
    )
    explicit = runner.invoke(
        app,
        ["emit", "mlir", "--input-format", "onnx", "-"],
        input=data,
    )

    assert implicit.exit_code == 0
    assert explicit.exit_code == 0
    assert implicit.stdout == explicit.stdout


def test_requires_input_format_for_stdin_or_unknown_extension(
    tmp_path: Path,
) -> None:
    stdin = runner.invoke(app, ["emit", "mlir"])
    unknown = runner.invoke(app, ["emit", "mlir", str(tmp_path / "model.bin")])

    assert stdin.exit_code != 0
    assert "--input-format" in stdin.stderr
    assert unknown.exit_code != 0
    assert "cannot infer" in unknown.stderr
