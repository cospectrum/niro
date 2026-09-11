from pathlib import Path

from . import helpers, onnx_models


def test_signature(
    installed_cli: helpers.InstalledCli,
    onnx_model: onnx_models.ModelCase,
    model_input: tuple[tuple[str | Path, ...], bytes | None],
) -> None:
    arguments, input_data = model_input
    signature = installed_cli.run(
        "inspect", "signature", *arguments, input_data=input_data
    )
    assert signature == onnx_model.signature
