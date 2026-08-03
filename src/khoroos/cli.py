"""Command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from khoroos.config import (
    PRESETS,
    get_settings,
    params_for_preset,
    parse_overrides,
    thresholds_from_overrides,
)

app = typer.Typer(
    name="khoroos",
    help="Poultry welfare monitoring from farm video.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(name="models", help="Manage model checkpoints.", no_args_is_help=True)
app.add_typer(models_app)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(help="Video file to analyse.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory.")] = Path(
        "khoroos-output"
    ),
    preset: Annotated[
        str, typer.Option("--preset", "-p", help=f"One of: {', '.join(PRESETS)}.")
    ] = "balanced",
    max_seconds: Annotated[
        float | None, typer.Option("--max-seconds", help="Analyse only the first N seconds.")
    ] = None,
    min_confidence: Annotated[
        float | None, typer.Option("--min-confidence", help="Below this, actions are 'uncertain'.")
    ] = None,
    detection_confidence: Annotated[
        float | None,
        typer.Option("--detection-confidence", help="Detector score threshold."),
    ] = None,
    detection_batch_size: Annotated[
        int | None,
        typer.Option(
            "--detection-batch-size",
            help="Frames per detector forward pass. Higher is faster but needs more memory.",
        ),
    ] = None,
    action_batch_size: Annotated[
        int | None,
        typer.Option(
            "--action-batch-size",
            help="Clips per action-model forward pass. Each clip is 64 frames, so this "
            "dominates memory use.",
        ),
    ] = None,
    overlay: Annotated[
        bool, typer.Option("--overlay/--no-overlay", help="Render an annotated video.")
    ] = False,
    set_param: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            "-s",
            metavar="NAME=VALUE",
            help="Override any analysis parameter, e.g. -s window_seconds=3. Repeatable.",
        ),
    ] = None,
    threshold: Annotated[
        list[str] | None,
        typer.Option(
            "--threshold",
            "-t",
            metavar="NAME=VALUE",
            help="Override a welfare alert threshold, e.g. -t min_locomotion_share=0.2.",
        ),
    ] = None,
    device: Annotated[str | None, typer.Option("--device", help="cuda, cpu, mps or auto.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Analyse a video and write results to a directory."""
    _setup_logging(verbose)

    if not video.is_file():
        typer.secho(f"No such video: {video}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    settings = get_settings()
    if device:
        settings.device = device

    try:
        # The named flags are shorthands for the same fields `--set` reaches, so they go
        # through one map. An explicit `--set` wins over its shorthand.
        overrides: dict[str, object] = {
            "max_duration_seconds": max_seconds,
            "min_confidence": min_confidence,
            "detection_confidence": detection_confidence,
            "detection_batch_size": detection_batch_size,
            "action_batch_size": action_batch_size,
        }
        overrides.update(parse_overrides(set_param or []))
        params = params_for_preset(preset, **overrides)
        thresholds = thresholds_from_overrides(parse_overrides(threshold or []))
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None

    from khoroos.pipeline.runner import AnalysisRunner

    typer.echo(f"Analysing {video} [{preset}] on {settings.resolved_device()}")

    with typer.progressbar(length=1000, label="starting") as bar:
        position = 0

        def on_progress(event) -> None:
            # The bar never moves backwards: stage weights are approximate, and a bar that
            # retreats reads as a bug during a run that already takes minutes.
            nonlocal position
            target = min(int(event.progress * 1000), 1000)
            step = max(target - position, 0)
            position += step
            bar.label = event.stage
            bar.update(step)

        artifacts = AnalysisRunner(settings=settings).run(
            video,
            output_dir=output,
            params=params,
            thresholds=thresholds,
            render_overlay=overlay,
            on_progress=on_progress,
        )

    _print_summary(artifacts.result)
    typer.echo("\nWrote:")
    for name, path in artifacts.paths.items():
        typer.echo(f"  {name:12s} {path}")


def _print_summary(result) -> None:
    metrics = result.metrics
    budget = metrics.get("time_budget", {})
    indicators = metrics.get("indicators", {})
    population = metrics.get("population", {})

    typer.echo("")
    typer.secho("Summary", bold=True)
    typer.echo(f"  video            {result.video.filename} ({result.video.duration_seconds:.1f}s)")
    typer.echo(f"  runtime          {result.runtime_seconds:.1f}s")
    typer.echo(f"  birds tracked    {len(result.tracks)}")
    typer.echo(f"  mean in frame    {population.get('mean', 0)}")
    typer.echo(f"  clips classified {len(result.predictions)}")
    typer.echo(f"  observed         {budget.get('total_bird_seconds', 0):.0f} bird-seconds")

    by_class = budget.get("by_class", {})
    if by_class:
        typer.echo("\n  Time budget:")
        for label, values in list(by_class.items())[:8]:
            bar = "█" * int(values["share"] * 30)
            typer.echo(f"    {label:18s} {values['share']:6.1%} {bar}")

    if indicators:
        typer.echo("\n  Welfare indicators:")
        for name, value in indicators.items():
            typer.echo(f"    {name:20s} {value:6.1%}")

    alerts = metrics.get("alerts", [])
    if alerts:
        typer.echo("")
        for alert in alerts:
            colour = typer.colors.YELLOW if alert["level"] == "warning" else typer.colors.CYAN
            typer.secho(f"  [{alert['level']}] {alert['message']}", fg=colour)

    for warning in result.warnings:
        typer.secho(f"  [note] {warning}", fg=typer.colors.MAGENTA)


@app.command()
def ui(
    host: Annotated[str | None, typer.Option("--host", help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Port to listen on.")] = None,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open a browser.")
    ] = False,
    device: Annotated[str | None, typer.Option("--device", help="cuda, cpu, mps or auto.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Launch the web interface."""
    _setup_logging(verbose)

    settings = get_settings()
    if device:
        settings.device = device
    if host:
        settings.host = host
    if port:
        settings.port = port

    try:
        import uvicorn
    except ImportError:
        typer.secho(
            "The web interface needs extra dependencies. Install them with:\n"
            "  pip install 'khoroos[web]'",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from None

    from khoroos.web.app import create_app

    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho(
            f"Serving on {settings.host} exposes Khoroos to your network. It has no "
            f"authentication — only do this on a trusted network.",
            fg=typer.colors.YELLOW,
        )

    url = f"http://{settings.host}:{settings.port}"
    typer.secho(f"Khoroos UI: {url}", fg=typer.colors.GREEN, bold=True)

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    web_app = create_app(settings)
    try:
        uvicorn.run(web_app, host=settings.host, port=settings.port, log_level="info")
    finally:
        web_app.state.jobs.close()


@models_app.command("status")
def models_status() -> None:
    """Show where model checkpoints will be loaded from."""
    from khoroos.models.registry import checkpoint_status

    _print_checkpoints(checkpoint_status(get_settings()))


@models_app.command("download")
def models_download() -> None:
    """Download model checkpoints so later runs work offline."""
    from khoroos.models.registry import CheckpointNotFoundError, resolve_checkpoint

    failed = False
    for kind in ("detector", "action"):
        typer.echo(f"Resolving {kind}...")
        try:
            path = resolve_checkpoint(kind, allow_download=True)
        except CheckpointNotFoundError as exc:
            failed = True
            typer.secho(f"  {kind}: {exc}", fg=typer.colors.RED, err=True)
        else:
            typer.secho(f"  {kind}: {path}", fg=typer.colors.GREEN)
    if failed:
        raise typer.Exit(1)


@app.command()
def info(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show presets, behaviours, thresholds and checkpoint status.

    The same description the web UI builds its controls from.
    """
    from khoroos.environment import describe_environment

    env = describe_environment(get_settings())

    if as_json:
        typer.echo(json.dumps(env, indent=2))
        return

    typer.secho(f"Khoroos {env['version']}", bold=True)
    typer.echo(f"  device      {env['device']}")
    typer.echo(f"  cache       {env['cache_dir']}")

    typer.echo("\n  Presets:")
    for name, values in env["presets"].items():
        default = " (default)" if name == env["default_preset"] else ""
        summary = " ".join(f"{k}={v}" for k, v in values.items())
        typer.echo(f"    {name:10s}{default:10s} {summary}")

    typer.echo("\n  Behaviours:")
    for group, members in env["behaviour_groups"].items():
        typer.echo(f"    {group:12s} {', '.join(members)}")

    typer.echo("\n  Alert thresholds (override with --threshold NAME=VALUE):")
    for name, value in env["thresholds"].items():
        typer.echo(f"    {name:28s} {value}")

    typer.echo("\n  Models:")
    _print_checkpoints(env["models"])


def _print_checkpoints(models: dict) -> None:
    for kind, details in models.items():
        if details["available"]:
            typer.secho(f"    {kind:9s} ✓ {details['path']}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"    {kind:9s} ✗ not found locally (would download {details['hub_repo']})",
                fg=typer.colors.YELLOW,
            )


@app.command()
def version() -> None:
    """Print the installed version."""
    from khoroos import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
