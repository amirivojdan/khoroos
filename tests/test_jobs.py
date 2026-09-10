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


@pytest.mark.parametrize("accelerator", ["mps", "cuda:1"])
def test_jobs_reuse_runner_and_release_models_before_device_switch(
    manager, tmp_path, monkeypatch, accelerator
):
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from khoroos import devices
    from khoroos.pipeline import runner

    monkeypatch.setattr(devices, "resolve_device", lambda device: device)
    released = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: released.append("mps"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append("cuda:1"))
    constructed = []
    references = []

    class RecordingRunner:
        def __init__(self, settings):
            if references:
                assert references[-1]() is None, "Previous models must be released before loading"
            self.settings = settings
            constructed.append(settings.device)
            references.append(weakref.ref(self))

        def run(self, *args, **kwargs):
            return SimpleNamespace(result=None, overlay=None, export_error=None)

    monkeypatch.setattr(runner, "AnalysisRunner", RecordingRunner)
    jobs = [manager.create_job(tmp_path / "video.mp4", "video.mp4", AnalysisParams(), device=device)
            for device in ("cpu", accelerator, accelerator, "cpu")]
    for job in jobs:
        manager._process(job)
        assert job.state == JobState.COMPLETED, job.error
        assert job.status_dict()["device"] == job.device
    assert constructed == ["cpu", accelerator, "cpu"]
    assert released == [accelerator]
    assert manager.settings.device == "cpu"


def test_runner_initialization_failure_finishes_job(manager, tmp_path, monkeypatch):
    from khoroos.pipeline import runner

    def fail(**kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(runner, "AnalysisRunner", fail)
    job = manager.create_job(tmp_path / "video.mp4", "video.mp4", AnalysisParams())
    subscription = manager.subscribe(job.job_id)
    manager._process(job)
    assert job.state == JobState.FAILED
    assert job.error == "initialization failed"
    assert job.finished_at is not None
    statuses = []
    while not subscription.empty():
        statuses.append(subscription.get_nowait()["state"])
    assert statuses[-1] == "failed"
    assert manager.runner is None


def test_unavailable_device_at_execution_marks_job_failed(manager, tmp_path, monkeypatch):
    from khoroos import devices

    job = manager.create_job(tmp_path / "video.mp4", "video.mp4", AnalysisParams())

    def unavailable(device):
        raise ValueError("Device became unavailable")

    monkeypatch.setattr(devices, "resolve_device", unavailable)
    manager._process(job)
    assert job.state == JobState.FAILED
    assert "unavailable" in job.error
    assert manager.runner is None


def test_injected_runner_is_not_replaced_on_device_switch(manager, stub_runner, monkeypatch):
    from khoroos import devices

    monkeypatch.setattr(devices, "resolve_device", lambda device: device)
    stub_runner.settings = stub_runner.settings.with_overrides(device="cpu")
    manager.runner = stub_runner
    manager._prepare_runner("cpu")
    with pytest.raises(ValueError, match="injected runner"):
        manager._prepare_runner("mps")
    assert manager.runner is stub_runner
