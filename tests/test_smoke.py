import pytest

from niro import hello
from niro.cli import main


def test_hello() -> None:
    assert hello() == "Hello from niro!"


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out == "Hello from niro!\n"
