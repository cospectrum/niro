from pathlib import Path
from typing import Literal

import onnx
import pytest

from e2e.cli.onnx_models import ModelCase, onnx_model_cases
from e2e.cli.support import InstalledCli, install_cli

_ONNX_MODEL_CASES = onnx_model_cases()


@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> InstalledCli:
    project_root = Path(__file__).parents[2]
    root = tmp_path_factory.mktemp("installed-cli")
    return install_cli(project_root, root)


@pytest.fixture(
    scope="session",
    params=_ONNX_MODEL_CASES,
    ids=tuple(case.name for case in _ONNX_MODEL_CASES),
)
def onnx_model(request: pytest.FixtureRequest) -> ModelCase:
    assert isinstance(request.param, ModelCase)
    return request.param


@pytest.fixture(params=("path", "stdin"))
def model_input(
    request: pytest.FixtureRequest,
    onnx_model: ModelCase,
    tmp_path: Path,
) -> tuple[tuple[str | Path, ...], bytes | None]:
    input_kind = request.param
    assert input_kind in ("path", "stdin")
    input_kind: Literal["path", "stdin"]
    if input_kind == "stdin":
        return ("--input-format", "onnx"), onnx_model.model.SerializeToString()

    model_path = tmp_path / f"{onnx_model.name}.onnx"
    onnx.save(onnx_model.model, model_path)
    return (model_path,), None
