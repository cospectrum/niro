import sys
from io import StringIO

from xdsl.dialects import builtin

import niro


def test_formats_mlir_as_text() -> None:
    text = niro.format_mlir(builtin.ModuleOp([]))

    assert text == "builtin.module {\n}"


def test_writes_mlir_to_path(tmp_path) -> None:
    output_path = tmp_path / "model.mlir"
    module = builtin.ModuleOp([])

    niro.write_mlir(module, output_path)

    assert output_path.read_text(encoding="utf-8") == niro.format_mlir(module)


def test_writes_mlir_to_stream_without_closing_it() -> None:
    stream = StringIO()
    module = builtin.ModuleOp([])

    niro.write_mlir(module, stream)

    assert stream.getvalue() == niro.format_mlir(module)
    assert not stream.closed


def test_writes_mlir_to_stdout(capsys) -> None:
    module = builtin.ModuleOp([])

    niro.write_mlir(module, sys.stdout)

    assert capsys.readouterr().out == niro.format_mlir(module)
