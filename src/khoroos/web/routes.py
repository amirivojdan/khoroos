"""HTTP API for the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import queue
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from khoroos.config import (
    PRESETS,
    AnalysisParams,
    params_for_preset,
)
from khoroos.pipeline.jobs import Job, JobManager, JobState

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


def _job_or_404(request: Request, job_id: str) -> Job:
    job = _manager(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    """Everything the frontend needs to render its controls.

    Same description ``khoroos info`` prints, so the two surfaces cannot disagree about
    what presets exist or which checkpoints are present.
    """
    from khoroos.environment import describe_environment

    runner = _manager(request).runner
    if runner is not None:
        return runner.analyzer.describe_environment()
    return describe_environment(request.app.state.settings)


@router.get("/browse")
def browse(request: Request, path: str | None = None) -> dict[str, Any]:
    """List video files in a server-side directory.

    Farm recordings are routinely multi-gigabyte; uploading them through the browser is
    slow and wasteful when the file already sits on the machine running Khoroos.
    """
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open {base}: {exc}") from exc
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    directories: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    try:
        for entry in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    directories.append({"name": entry.name, "path": str(entry)})
                elif entry.suffix.lower() in ALLOWED_SUFFIXES:
                    videos.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "size_mb": round(entry.stat().st_size / 1e6, 1),
                        }
                    )
            except OSError:
                continue
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {base}") from exc

    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "directories": directories,
        "videos": videos,
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _build_params(preset: str, overrides: dict[str, Any]) -> AnalysisParams:
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset {preset!r}")
    try:
        return params_for_preset(preset, **overrides)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs")
async def create_job(
    request: Request,
    file: UploadFile | None = File(None),
    server_path: str | None = Form(None),
    preset: str = Form("balanced"),
    max_seconds: float | None = Form(None),
    min_confidence: float | None = Form(None),
    detection_confidence: float | None = Form(None),
    detection_batch_size: int | None = Form(None),
    action_batch_size: int | None = Form(None),
    action_classes: str | None = Form(None),
    behaviour_groups: str | None = Form(None),
    invalid_boxes: str | None = Form(None),
    render_overlay: bool = Form(False),
) -> dict[str, Any]:
    """Start an analysis from an upload or a path on the server."""
    manager = _manager(request)
    settings = request.app.state.settings

    if file is None and not server_path:
        raise HTTPException(status_code=400, detail="Provide either a file upload or server_path.")

    # Validated before the upload is consumed: farm recordings run to gigabytes, and
    # writing one to disk only to reject it over a typo'd preset wastes minutes.
    params = _build_params(
        preset,
        {
            "max_duration_seconds": max_seconds,
            "min_confidence": min_confidence,
            "detection_confidence": detection_confidence,
            "detection_batch_size": detection_batch_size,
            "action_batch_size": action_batch_size,
            "action_classes": action_classes,
            "behaviour_groups": behaviour_groups,
            "invalid_boxes": invalid_boxes,
        },
    )

    if server_path:
        video_path = Path(server_path).expanduser()
        if not video_path.is_file():
            raise HTTPException(status_code=400, detail=f"No such file: {video_path}")
        if video_path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported video type {video_path.suffix}"
            )
        original_name = video_path.name
    else:
        assert file is not None
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported video type {suffix!r}")

        uploads = settings.jobs_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        video_path = uploads / f"{Path(file.filename or 'video').stem}-{id(file):x}{suffix}"

        limit = settings.max_upload_mb * 1024 * 1024
        written = 0
        with video_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    handle.close()
                    video_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                    )
                handle.write(chunk)
        original_name = file.filename or video_path.name

    job = manager.create_job(
        video_path=video_path,
        original_filename=original_name,
        params=params,
        render_overlay=render_overlay,
    )
    return job.status_dict()


@router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    return {"jobs": [job.status_dict() for job in _manager(request).list_jobs()]}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    return _job_or_404(request, job_id).status_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    _job_or_404(request, job_id)
    cancelled = _manager(request).cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="Job already finished.")
    return {"cancelled": True}


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    """Server-sent events carrying progress until the job reaches a terminal state."""
    manager = _manager(request)
    _job_or_404(request, job_id)
    subscription = manager.subscribe(job_id)

    async def stream():
        loop = asyncio.get_running_loop()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await loop.run_in_executor(None, subscription.get, True, 15.0)
                except queue.Empty:
                    yield ": keepalive\n\n"  # keeps proxies from closing the connection
                    continue

                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("state") in {
                    JobState.COMPLETED.value,
                    JobState.FAILED.value,
                    JobState.CANCELLED.value,
                }:
                    break
        finally:
            manager.unsubscribe(job_id, subscription)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/result")
def get_result(request: Request, job_id: str, include_tracks: bool = True) -> JSONResponse:
    job = _job_or_404(request, job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.state.value}, not completed.")
    return JSONResponse(job.result.to_dict(include_tracks=include_tracks))


@router.get("/jobs/{job_id}/video")
def get_video(request: Request, job_id: str, annotated: bool = False) -> FileResponse:
    """Serve the source or annotated video.

    ``FileResponse`` handles HTTP range requests, which the player needs in order to seek
    when the user clicks the timeline.
    """
    job = _job_or_404(request, job_id)
    path = job.overlay_path if annotated else job.video_path
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Video not available")

    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/jobs/{job_id}/export/{artifact}")
def download_export(request: Request, job_id: str, artifact: str) -> FileResponse:
    """Download one of the generated export files."""
    job = _job_or_404(request, job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail="Job has not completed.")

    filenames = {
        "result": "result.json",
        "metrics": "metrics.json",
        "predictions": "predictions.csv",
        "time_budget": "time_budget.csv",
        "per_bird": "per_bird.csv",
    }
    if artifact not in filenames:
        raise HTTPException(status_code=404, detail=f"Unknown export {artifact!r}")

    path = _manager(request).job_dir(job_id) / filenames[artifact]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path, filename=f"{Path(job.original_filename).stem}-{filenames[artifact]}")


@router.delete("/jobs/{job_id}")
def delete_job(request: Request, job_id: str) -> dict[str, bool]:
    manager = _manager(request)
    try:
        deleted = manager.delete(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True}
