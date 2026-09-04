from pathlib import Path

from e2e.cli.onnx_models import ModelCase
from e2e.cli.support import InstalledCli


def test_signature(
    installed_cli: InstalledCli,
    onnx_model: ModelCase,
    model_input: tuple[tuple[str | Path, ...], bytes | None],
) -> None:
    arguments, input_data = model_input
    signature = installed_cli.run(
        "inspect", "signature", *arguments, input_data=input_data
    )
    assert signature == onnx_model.signature
