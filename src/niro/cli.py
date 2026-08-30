import typer

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    """Run niro."""
