"""Command line entry point."""

import typer

from berrytrack import __version__

app = typer.Typer(no_args_is_help=True, help="BerryTrack research and video tracking tools.")


@app.callback()
def main() -> None:
    """Work with validated data, bounded experiments, and video inference."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)
