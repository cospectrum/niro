from pathlib import Path

from e2e.cli.onnx_models import ModelCase
from e2e.cli.support import InstalledCli


def test_mlir(
    installed_cli: InstalledCli,
    onnx_model: ModelCase,
    model_input: tuple[tuple[str | Path, ...], bytes | None],
) -> None:
    arguments, input_data = model_input
    mlir = installed_cli.run("emit", "mlir", *arguments, input_data=input_data)
    assert all(fragment in mlir for fragment in onnx_model.mlir_fragments)
