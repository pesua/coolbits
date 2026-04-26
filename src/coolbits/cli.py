"""coolbits CLI — analyze, preview, render."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from . import config as config_mod
from . import manifest as mf
from .analyze import pipeline as analyze_pipeline
from .render import pipeline as render_pipeline


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 3rd-party loggers stay at WARNING regardless of -v
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@click.group()
@click.option("--workspace", type=click.Path(path_type=Path), default=Path("./.coolbits"))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def main(ctx: click.Context, workspace: Path, config_path: Path | None, verbose: bool) -> None:
    """Cool-Bits Extractor."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace
    ctx.obj["config"] = config_mod.load(config_path)


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--force", is_flag=True, help="Re-run analysis even if manifest exists.")
@click.option("--skip-clip", is_flag=True)
@click.option("--skip-motion", is_flag=True)
@click.option("--skip-captions", is_flag=True)
@click.pass_context
def analyze(
    ctx: click.Context,
    video: Path,
    force: bool,
    skip_clip: bool,
    skip_motion: bool,
    skip_captions: bool,
) -> None:
    """Run Phase 1 analysis: produces a manifest under <workspace>/manifests/."""
    analyze_pipeline.run(
        video,
        workspace=ctx.obj["workspace"],
        config=ctx.obj["config"],
        force=force,
        skip_clip=skip_clip,
        skip_motion=skip_motion,
        skip_captions=skip_captions,
    )


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--max-lines", type=int, default=None)
@click.pass_context
def preview(ctx: click.Context, video: Path, max_lines: int | None) -> None:
    """Phase 2 selection only — print text listing of selected shots."""
    workspace = ctx.obj["workspace"]
    src_hash = mf.partial_hash(video.resolve())
    man_path = mf.manifest_path(workspace, src_hash)
    if not man_path.exists():
        click.echo(f"No manifest at {man_path}. Run `coolbits analyze` first.", err=True)
        sys.exit(2)
    manifest = mf.load(man_path)
    plan = render_pipeline.plan(manifest, ctx.obj["config"])
    click.echo(render_pipeline.format_preview(manifest, plan, max_lines=max_lines))


@main.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("cut.mp4"))
@click.pass_context
def render(ctx: click.Context, video: Path, out_path: Path) -> None:
    """Phase 2 full render — encode the supercut."""
    workspace = ctx.obj["workspace"]
    src_hash = mf.partial_hash(video.resolve())
    man_path = mf.manifest_path(workspace, src_hash)
    if not man_path.exists():
        click.echo(f"No manifest at {man_path}. Run `coolbits analyze` first.", err=True)
        sys.exit(2)
    manifest = mf.load(man_path)
    plan = render_pipeline.plan(manifest, ctx.obj["config"])
    if not plan.intervals:
        click.echo("No shots selected — nothing to render.", err=True)
        sys.exit(3)
    click.echo(
        f"Rendering {len(plan.intervals)} interval(s), "
        f"total {plan.total_duration_s:.1f}s, to {out_path}"
    )
    render_pipeline.encode(plan, video.resolve(), out_path, ctx.obj["config"])
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
