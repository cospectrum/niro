from pathlib import Path

from . import onnx_models, support


def test_mlir(
    installed_cli: support.InstalledCli,
    onnx_model: onnx_models.ModelCase,
    model_input: tuple[tuple[str | Path, ...], bytes | None],
) -> None:
    arguments, input_data = model_input
    mlir = installed_cli.run("emit", "mlir", *arguments, input_data=input_data)
    assert all(fragment in mlir for fragment in onnx_model.mlir_fragments)
