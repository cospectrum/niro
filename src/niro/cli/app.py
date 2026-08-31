"""Command-line application for Niro."""

import typer

from niro.cli.mlir import emit_mlir

app = typer.Typer(no_args_is_help=True)
emit = typer.Typer(no_args_is_help=True)
emit.command("mlir")(emit_mlir)
app.add_typer(emit, name="emit", help="Emit a target representation.")
