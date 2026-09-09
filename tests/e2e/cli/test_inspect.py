from pathlib import Path

from . import onnx_models, support


def test_signature(
    installed_cli: support.InstalledCli,
    onnx_model: onnx_models.ModelCase,
    model_input: tuple[tuple[str | Path, ...], bytes | None],
) -> None:
    arguments, input_data = model_input
    signature = installed_cli.run(
        "inspect", "signature", *arguments, input_data=input_data
    )
    assert signature == onnx_model.signature
