"""Command-line application for Niro."""

import typer

from niro.cli import mlir, signature

app = typer.Typer(no_args_is_help=True)
emit = typer.Typer(no_args_is_help=True)
inspect = typer.Typer(no_args_is_help=True)
emit.command("mlir")(mlir.emit_mlir)
inspect.command("signature")(signature.inspect_signature)
app.add_typer(emit, name="emit", help="Emit a target representation.")
app.add_typer(inspect, name="inspect", help="Inspect a model.")
