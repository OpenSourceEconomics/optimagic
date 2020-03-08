"""This module comprises all CLI capabilities of estimagic."""
import click

from estimagic.dashboard.run_dashboard import run_dashboard

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    """Command-line interface for estimagic."""
    pass


@cli.command()
@click.argument("database", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--port",
    "-p",
    default=1234,
    help="The port the dashboard server will listen on.",
    type=int,
    show_default=True,
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open the dashboard in a browser after startup.",
)
def dashboard(database, port, no_browser):
    """Start the dashboard to visualize optimizations."""
    run_dashboard(database, no_browser, port)
