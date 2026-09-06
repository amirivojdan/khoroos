"""Background job management for the web UI.

Analysis is a multi-minute, GPU-bound operation, so it cannot run inside a request
handler. Jobs run on a single background worker thread — one at a time, because the two
models together occupy enough memory that concurrent runs would thrash — and progress is
published to subscribers over queues that the SSE endpoint drains.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from khoroos.config import AnalysisParams, Settings, get_settings
from khoroos.pipeline.types import AnalysisResult, ProgressEvent

if TYPE_CHECKING:  # importing the runner eagerly would pull torch in at web-app startup
    from khoroos.pipeline.runner import AnalysisRunner

logger = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Job:
    """One video analysis request and everything produced from it."""

    job_id: str
    video_path: Path
    original_filename: str
    params: AnalysisParams

    render_overlay: bool = False

    state: JobState = JobState.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    message: str = "Waiting to start"
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    result: AnalysisResult | None = None
    overlay_path: Path | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def status_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
            "error": self.error,
            "filename": self.original_filename,
            "created_at": self.created_at,
            "runtime_s": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
            ),
            "has_overlay": self.overlay_path is not None and self.overlay_path.exists(),
        }


class JobManager:
    """Owns the worker thread, the job registry and progress fan-out."""

    def __init__(
        self, settings: Settings | None = None, runner: AnalysisRunner | None = None
    ) -> None:
        self.settings = settings or (runner.settings if runner is not None else get_settings())
        #: Built on first use so the UI starts before 1.7 GB of weights load. Injectable so
        #: callers — tests, embedders — can supply an already-loaded or stubbed runner.
        self.runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._closed = threading.Event()
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="khoroos-worker")
        self._worker.start()

    # -- registry ----------------------------------------------------------

    def create_job(
        self,
        video_path: Path,
        original_filename: str,
        params: AnalysisParams,
        render_overlay: bool = False,
    ) -> Job:
        if self._closed.is_set():
            raise RuntimeError("Job manager is closed.")
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            video_path=video_path,
            original_filename=original_filename,
            params=params,
            render_overlay=render_overlay,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._subscribers[job.job_id] = []
        self._queue.put(job.job_id)
        logger.info("Queued job %s for %s", job.job_id, original_filename)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state.is_terminal:
                return False
            job._cancel.set()
            queued = job.state == JobState.QUEUED
            if queued:
                job.state = JobState.CANCELLED
                job.message = "Cancelled before starting"
                job.finished_at = time.time()
        if queued:
            self._publish(job)
        return True

    def delete(self, job_id: str) -> bool:
        """Delete a terminal job and its managed artifacts, never an external source video.

        Returns False if the job is absent. Active jobs raise ValueError. Registry changes
        and deletion are serialized so cleanup and HTTP deletion share the same operation.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if not job.state.is_terminal:
                raise ValueError("Cancel the job and wait for it to stop before deleting it.")
            directory = self.settings.jobs_dir / job_id
            if directory.exists():
                shutil.rmtree(directory)
            source = job.video_path
            if source.resolve().is_relative_to(self.settings.jobs_dir.resolve()):
                source.unlink(missing_ok=True)
            del self._jobs[job_id]
            self._subscribers.pop(job_id, None)
            return True

    def job_dir(self, job_id: str) -> Path:
        path = self.settings.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- progress fan-out ---------------------------------------------------

    def subscribe(self, job_id: str) -> queue.Queue:
        """Return a queue receiving this job's status updates until it terminates."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        job = self.get(job_id)
        if job is not None:
            q.put(job.status_dict())
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            if q in subscribers:
                subscribers.remove(q)

    def _publish(self, job: Job) -> None:
        payload = job.status_dict()
        with self._lock:
            subscribers = list(self._subscribers.get(job.job_id, []))
        for q in subscribers:
            with contextlib.suppress(queue.Full):  # queues are unbounded in practice
                q.put_nowait(payload)

    # -- worker -------------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            try:
                self.cleanup_expired()
            except OSError:
                logger.exception("Could not clean up expired jobs")
            try:
                job_id = self._queue.get(timeout=60)
            except queue.Empty:
                continue
            if job_id is None:
                return
            job = self.get(job_id)
            if job is None or job.state.is_terminal:
                continue
            try:
                self._process(job)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Worker crashed on job %s", job_id)

    def _process(self, job: Job) -> None:
        from khoroos.pipeline.analyze import CancelledError
        from khoroos.pipeline.runner import AnalysisRunner

        with self._lock:
            if job.state.is_terminal or job.job_id not in self._jobs:
                return
            job.state = JobState.RUNNING
            job.started_at = time.time()
            job.stage = "loading"
            job.message = "Loading models"
        self._publish(job)

        if self.runner is None:
            self.runner = AnalysisRunner(settings=self.settings)

        def on_progress(event: ProgressEvent) -> None:
            job.stage = event.stage
            job.progress = event.progress
            job.message = event.message
            job.detail = event.detail
            self._publish(job)

        try:
            artifacts = self.runner.run(
                job.video_path,
                output_dir=self.job_dir(job.job_id),
                params=job.params,
                render_overlay=job.render_overlay,
                on_progress=on_progress,
                should_cancel=job._cancel.is_set,
                # A full disk should not throw away a multi-minute analysis: the result is
                # already in memory and served from there, only the downloads are lost.
                require_exports=False,
            )

            job.result = artifacts.result
            job.overlay_path = artifacts.overlay
            if artifacts.export_error:
                logger.warning(
                    "Exports for job %s were not written: %s", job.job_id, artifacts.export_error
                )

            self._finish(job, JobState.COMPLETED, "Analysis complete")

        except CancelledError:
            self._finish(job, JobState.CANCELLED, "Cancelled")
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            job.error = str(exc)
            self._finish(job, JobState.FAILED, f"Failed: {exc}")

    def _finish(self, job: Job, state: JobState, message: str) -> None:
        job.state = state
        job.message = message
        job.finished_at = time.time()
        if state == JobState.COMPLETED:
            job.progress = 1.0
        self._publish(job)

    # -- housekeeping -------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Delete terminal jobs once the configured TTL has elapsed since completion."""
        cutoff = time.time() - self.settings.job_ttl_hours * 3600
        removed = 0
        for job in self.list_jobs():
            if not job.state.is_terminal or job.finished_at is None or job.finished_at >= cutoff:
                continue
            removed += int(self.delete(job.job_id))
        return removed

    def close(self, timeout: float = 5.0) -> None:
        """Cancel unfinished work and ask the background worker to stop."""
        if self._closed.is_set():
            return
        self._closed.set()
        for job in self.list_jobs():
            if not job.state.is_terminal:
                self.cancel(job.job_id)
        self._queue.put(None)
        self._worker.join(timeout=timeout)
