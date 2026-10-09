"""Command line entry point."""

from pathlib import Path
from typing import Annotated

import typer

from berrytrack import __version__

app = typer.Typer(no_args_is_help=True, help="BerryTrack research and video tracking tools.")


@app.callback()
def main() -> None:
    """Work with validated data, bounded experiments, and video inference."""
    from berrytrack.logging import configure_logging

    configure_logging()


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def demo() -> None:
    """Launch the local video application at http://127.0.0.1:7860."""
    from berrytrack.demo import main as launch_demo

    launch_demo()


@app.command()
def track(
    video: Path,
    weights: str = "checkpoints/blueberry-via/best.pt",
    output: Path = Path("artifacts/video"),
    algorithm: str = "berrytrack",
    compare: str | None = None,
    detections: Path | None = None,
    gt: Path | None = None,
    config: Path | None = None,
    max_frames: int = 3000,
    encoder: Path | None = None,
    reuse_backbone: bool = False,
) -> None:
    """Track a video and export annotated H.264, MOT rows, and timings."""
    import json

    from berrytrack.config import TrackerConfig, load_config
    from berrytrack.detection.yolo import YoloDetector
    from berrytrack.pipeline import process_video

    cfg = (
        load_config(config).tracker
        if config
        else TrackerConfig.model_validate({"algorithm": algorithm})
    )
    detector = None if detections else YoloDetector(weights, reuse_features=reuse_backbone)
    result = process_video(
        video,
        output,
        cfg,
        detector,
        detections,
        compare,
        gt,
        max_frames,
        appearance_checkpoint=encoder,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("cache-detections")
def cache(
    sequence: Path,
    weights: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    image_size: int = 640,
) -> None:
    """Cache actual model predictions for fair association comparisons."""
    import json

    from berrytrack.detection.cache import cache_detections

    typer.echo(
        json.dumps(cache_detections(sequence, weights, output, image_size=image_size), indent=2)
    )


@app.command("train-encoder")
def train_encoder_cli(
    sequence: Path,
    output: Path = Path("checkpoints/encoder-local"),
    steps: int = 10,
    dimension: int = 64,
    resume: bool = True,
) -> None:
    """Run a bounded CPU encoder training smoke, with resumable checkpoints."""
    import json

    from berrytrack.embeddings.training import train_encoder

    typer.echo(
        json.dumps(train_encoder([sequence], output, dimension, steps, resume=resume), indent=2)
    )


@app.command("annotation-task")
def annotation_task(video: Path, output: Path, frames: int, license_name: str) -> None:
    """Create a blank manual tracking task; no identities are fabricated."""
    from berrytrack.data.annotation import create_annotation_task

    create_annotation_task(video, output, frames, license_name)


@app.command("export-annotations")
def export_annotations(task: Path, output: Path) -> None:
    """Export only fully reviewed manual annotations to MOT."""
    from berrytrack.data.annotation import export_verified_task

    export_verified_task(task, output)
