from typer.testing import CliRunner

from niro.cli import app

runner = CliRunner()


def test_cli() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
