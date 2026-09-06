"""One analysis, end to end: analyse, export, optionally render an overlay.

:class:`~khoroos.pipeline.analyze.VideoAnalyzer` produces an
:class:`~khoroos.pipeline.types.AnalysisResult` and stops there. But neither the CLI nor
the web UI wants only that — both want the result *plus* the export bundle on disk, plus
an annotated video when asked. This module owns that sequence so "a run" means the same
thing on both surfaces, and so a change to what a run produces lands in one place.

The surfaces keep what is genuinely theirs: the CLI owns its progress bar and summary
table, the web worker owns job state and SSE fan-out. Both call :meth:`AnalysisRunner.run`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from khoroos.config import AnalysisParams, Settings, get_settings
from khoroos.pipeline.analyze import VideoAnalyzer
from khoroos.pipeline.types import AnalysisResult, ProgressEvent

logger = logging.getLogger(__name__)

#: Stages the runner adds after the pipeline's own stages have finished. The pipeline's
#: progress already reaches 1.0 by then, so these report completion rather than fractions.
POST_STAGES = ("export", "overlay")


@dataclass(slots=True)
class RunArtifacts:
    """Everything one run produced."""

    result: AnalysisResult
    output_dir: Path
    exports: dict[str, Path] = field(default_factory=dict)
    overlay: Path | None = None
    #: Set when exports failed and ``require_exports`` was False.
    export_error: str | None = None

    @property
    def paths(self) -> dict[str, Path]:
        """Exports plus the overlay, for callers that just want to list what was written."""
        paths = dict(self.exports)
        if self.overlay is not None:
            paths["overlay"] = self.overlay
        return paths


class AnalysisRunner:
    """Runs analyses and writes their artifacts.

    Holds a :class:`VideoAnalyzer`, so a long-lived caller (the web worker) reuses loaded
    weights across runs while a one-shot caller (the CLI) pays for them once.
    """

    def __init__(
        self,
        analyzer: VideoAnalyzer | None = None,
        settings: Settings | None = None,
        *,
        exporter: Callable[[AnalysisResult, Path], dict[str, Path]] | None = None,
        overlay_renderer: Callable[..., Path] | None = None,
    ) -> None:
        self.exporter = exporter
        self.overlay_renderer = overlay_renderer
        self.settings = settings or (analyzer.settings if analyzer is not None else get_settings())
        self.analyzer = analyzer or VideoAnalyzer(settings=self.settings)

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        params: AnalysisParams | None = None,
        preset: str = "balanced",
        render_overlay: bool = False,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        require_exports: bool = True,
    ) -> RunArtifacts:
        """Analyse ``video_path`` and write its artifacts into ``output_dir``.

        ``require_exports`` is what the two surfaces genuinely disagree on. The CLI was
        asked for files and should fail loudly if it cannot write them; the web worker
        holds the result in memory and would rather complete the job with downloads
        degraded than throw away a multi-minute analysis over a full disk.
        """
        output_dir = Path(output_dir)
        video_path = Path(video_path)

        def emit(stage: str, message: str) -> None:
            if on_progress is not None:
                on_progress(ProgressEvent(stage=stage, progress=1.0, message=message))

        result = self.analyzer.analyze(
            video_path,
            params=params,
            preset=preset,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )

        artifacts = RunArtifacts(result=result, output_dir=output_dir)

        emit("export", "Writing exports")
        artifacts.exports = self._export(result, output_dir, require_exports, artifacts)

        if render_overlay:
            emit("overlay", "Rendering annotated video")
            render = self.overlay_renderer
            if render is None:
                from khoroos.video.writer import render_overlay as render

                render = partial(render, source_factory=self.analyzer.components.source_factory)
            artifacts.overlay = render(
                video_path,
                result,
                output_dir / "annotated.mp4",
                max_seconds=result.video.duration_seconds,
            )

        return artifacts

    def _export(
        self,
        result: AnalysisResult,
        output_dir: Path,
        require_exports: bool,
        artifacts: RunArtifacts,
    ) -> dict[str, Path]:
        export = self.exporter
        if export is None:
            from khoroos.statistics.export import export_all as export

        try:
            return export(result, output_dir)
        except OSError as exc:
            if require_exports:
                raise
            logger.warning("Could not write exports to %s: %s", output_dir, exc)
            artifacts.export_error = str(exc)
            return {}
