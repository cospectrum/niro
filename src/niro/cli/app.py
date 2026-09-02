"""Command-line application for Niro."""

import typer

from niro.cli.mlir import emit_mlir
from niro.cli.signature import inspect_signature

app = typer.Typer(no_args_is_help=True)
emit = typer.Typer(no_args_is_help=True)
inspect = typer.Typer(no_args_is_help=True)
emit.command("mlir")(emit_mlir)
inspect.command("signature")(inspect_signature)
app.add_typer(emit, name="emit", help="Emit a target representation.")
app.add_typer(inspect, name="inspect", help="Inspect a model.")
