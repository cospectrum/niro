from pathlib import Path
from typing import Literal

import onnx
import pytest

from . import helpers, onnx_models

_ONNX_MODEL_CASES = onnx_models.onnx_model_cases()


@pytest.fixture(scope="session")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> helpers.InstalledCli:
    project_root = Path(__file__).parents[3]
    root = tmp_path_factory.mktemp("installed-cli")
    return helpers.install_cli(project_root, root)


@pytest.fixture(
    scope="session",
    params=_ONNX_MODEL_CASES,
    ids=tuple(case.name for case in _ONNX_MODEL_CASES),
)
def onnx_model(request: pytest.FixtureRequest) -> onnx_models.ModelCase:
    assert isinstance(request.param, onnx_models.ModelCase)
    return request.param


@pytest.fixture(params=("path", "stdin"))
def model_input(
    request: pytest.FixtureRequest,
    onnx_model: onnx_models.ModelCase,
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
