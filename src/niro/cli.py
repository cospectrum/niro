import typer

from niro import hello

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    typer.echo(hello())
