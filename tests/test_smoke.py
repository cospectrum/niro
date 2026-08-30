import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from niro import hello
from niro.cli import app, main

runner = CliRunner()


def test_hello() -> None:
    assert hello() == "Hello from niro!"


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out == "Hello from niro!\n"


def test_typer_cli_output() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert result.stdout == "Hello from niro!\n"
    assert result.stderr == ""


def test_installed_cli_output() -> None:
    executable = shutil.which("niro")
    assert executable is not None

    result = subprocess.run(
        [executable],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "Hello from niro!\n"
    assert result.stderr == ""
