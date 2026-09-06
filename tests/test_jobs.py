"""Job deletion and expiry share one lifecycle operation."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import pytest

from khoroos.config import AnalysisParams, Settings
from khoroos.pipeline.jobs import JobManager, JobState

_REAL_RUN_WORKER = JobManager._run_worker


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(JobManager, "_run_worker", lambda self: self._closed.wait())
    manager = JobManager(Settings(cache_dir=tmp_path / "cache", device="cpu"))
    yield manager
    manager.close()


def test_active_job_cannot_be_deleted(manager, tmp_path):
    source = tmp_path / "external.mp4"
    source.write_bytes(b"video")
    job = manager.create_job(source, source.name, AnalysisParams())
    with pytest.raises(ValueError, match="Cancel"):
        manager.delete(job.job_id)
    assert manager.get(job.job_id) is job
    manager.cancel(job.job_id)
    assert manager.delete(job.job_id)
    assert source.exists()
    assert not manager.delete(job.job_id)
    # A worker that obtained the job before cancellation must not restart it.
    manager._process(job)
    assert job.state == JobState.CANCELLED


def test_expiry_deletes_managed_uploads_and_artifacts(manager):
    source = manager.settings.jobs_dir / "upload.mp4"
    source.write_bytes(b"video")
    job = manager.create_job(source, source.name, AnalysisParams())
    directory = manager.job_dir(job.job_id)
    (directory / "result.json").write_text("{}")
    manager.cancel(job.job_id)
    job.finished_at = 0
    subscription = manager.subscribe(job.job_id)
    assert manager.cleanup_expired() == 1
    assert not source.exists()
    assert not directory.exists()
    assert manager.list_jobs() == []
    manager.unsubscribe(job.job_id, subscription)


def test_concurrent_deletion_has_one_owner(manager, tmp_path):
    job = manager.create_job(tmp_path / "source.mp4", "source.mp4", AnalysisParams())
    manager.cancel(job.job_id)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(manager.delete, [job.job_id, job.job_id]))
    assert sorted(results) == [False, True]


def test_failed_artifact_removal_retains_registry(manager, tmp_path, monkeypatch):
    import shutil

    job = manager.create_job(tmp_path / "source.mp4", "source.mp4", AnalysisParams())
    manager.job_dir(job.job_id)
    manager.cancel(job.job_id)

    def fail(path):
        raise OSError("disk error")

    monkeypatch.setattr(shutil, "rmtree", fail)
    with pytest.raises(OSError):
        manager.delete(job.job_id)
    assert manager.get(job.job_id) is job


def test_recently_finished_old_job_is_retained(manager, tmp_path):
    job = manager.create_job(tmp_path / "source.mp4", "source.mp4", AnalysisParams())
    job.created_at = 0
    manager.cancel(job.job_id)
    assert manager.cleanup_expired() == 0
    assert manager.get(job.job_id) is job


def test_idle_worker_runs_expiry_and_survives_cleanup_failure(manager, monkeypatch):
    attempts = []

    def cleanup():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary storage failure")
        return 0

    class IdleQueue(Queue):
        def get(self, *, timeout):
            assert timeout == 60
            if len(attempts) == 1:
                raise Empty
            return None

    monkeypatch.setattr(manager, "cleanup_expired", cleanup)
    monkeypatch.setattr(manager, "_queue", IdleQueue())
    # The fixture suppresses the background thread; run the real loop synchronously.
    _REAL_RUN_WORKER(manager)
    assert len(attempts) == 2
