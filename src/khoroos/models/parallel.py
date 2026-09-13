"""Run one model on several devices at once.

An analysis is a sequence of batches — frames for the detector, clips for the action model
— and no batch depends on another. That makes the GPU work trivially splittable: load the
model once per device, give each device a contiguous share of every batch, run the shares
concurrently, and reassemble the results in input order.

The pipeline above is unchanged by this. It still calls ``detect`` and ``classify`` once
per batch and still receives results in the order it asked for them, so tracking, tracklet
windows and predictions are built from the same values in the same order.

The concurrency itself changes no arithmetic: handed the same share, a replica returns
bit-identical results whether it ran alone or beside another. What does shift is the batch
*shape* each forward pass sees, and batched GPU inference is not bit-exact across shapes —
exactly what already happens when ``detection_batch_size`` is changed. Scores move in their
last decimal places, which can flip a detection sitting right on the confidence threshold.
Aggregate behaviour is unaffected; reproducing a run exactly needs the same device count
as well as the same batch sizes.

Shares are cut from the batch rather than each device being handed a whole batch, so
``detection_batch_size`` and ``action_batch_size`` stay *totals*: per-device memory use is
what the same configuration would use on one GPU. Two GPUs therefore halve the batch each
one sees; raise the batch sizes if you would rather keep them busy with the larger batches
they can still hold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from khoroos.config import Settings, get_settings
from khoroos.interfaces import Detector, VideoClassifier, component_name

if TYPE_CHECKING:
    import torch

    from khoroos.annotations import Detections

logger = logging.getLogger(__name__)


def split_evenly(count: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous ``(start, stop)`` spans covering ``count`` items over up to ``parts`` workers.

    Remainders go to the earliest spans, so three items over two devices give the first
    device two. Empty spans are never produced: fewer items than devices simply uses fewer
    devices, which is what happens to the last batch of most runs.
    """
    if count <= 0 or parts <= 0:
        return []
    parts = min(parts, count)
    size, extra = divmod(count, parts)
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        spans.append((start, stop))
        start = stop
    return spans


class ReplicaPool:
    """One model loaded per device, with a worker thread per replica.

    The pool owns the threads for the life of the replicas, because the web worker reuses
    loaded models across jobs and a per-batch pool would pay thread setup thousands of
    times per video.
    """

    def __init__(self, replicas: Sequence[Any]) -> None:
        if not replicas:
            raise ValueError("A parallel model needs at least one replica.")
        self.replicas = list(replicas)
        self.devices = [str(getattr(replica, "device", "cpu")) for replica in self.replicas]
        self._pool = None
        if len(self.replicas) > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(
                max_workers=len(self.replicas), thread_name_prefix="khoroos-device"
            )

    @property
    def primary(self) -> Any:
        """The replica that answers questions about the model rather than running it."""
        return self.replicas[0]

    def map[T](self, count: int, work: Callable[[Any, int, int], T]) -> list[T]:
        """Run ``work(replica, start, stop)`` over the shares of ``count`` items, in order.

        Each replica receives at most one share per call and the call does not return until
        every share is done, so no replica is ever used by two threads at once — which is
        what makes it safe for the replicas themselves to be ordinary single-device models.
        """
        spans = split_evenly(count, len(self.replicas))
        if self._pool is None or len(spans) < 2:
            return [work(self.primary, start, stop) for start, stop in spans]

        futures = [
            self._pool.submit(work, replica, start, stop)
            for replica, (start, stop) in zip(self.replicas, spans, strict=False)
        ]
        # Every share is waited on even after one fails, so a failure does not leave a
        # device running a forward pass into a batch the caller has already given up on.
        results: list[T] = []
        error: BaseException | None = None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:  # noqa: BLE001 - re-raised once all shares are done
                error = error or exc
        if error is not None:
            raise error
        return results

    def close(self) -> None:
        """Stop the worker threads. The pool runs single-device afterwards."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


class _Parallel:
    """Shared plumbing: metadata comes from the primary replica, work from all of them."""

    def __init__(self, replicas: Sequence[Any]) -> None:
        self.pool = ReplicaPool(replicas)
        self.replicas = self.pool.replicas
        self.devices = self.pool.devices
        # Named and carded as the checkpoint it is, not as this wrapper: which devices ran
        # a model is a deployment detail, and results should not record it as a model change.
        self.checkpoint = getattr(self.pool.primary, "checkpoint", None)

    @property
    def name(self) -> str:
        return component_name(self.pool.primary)

    def close(self) -> None:
        self.pool.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.name}, devices={self.devices})"


class ParallelDetector(_Parallel, Detector):
    """A detector replicated across devices; each batch of frames becomes one share each."""

    def detect(
        self,
        images,
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.60,
        box_padding: float = 0.0,
    ) -> list[Detections]:
        frames = list(images)

        def work(replica: Detector, start: int, stop: int) -> list[Detections]:
            return replica.detect(
                frames[start:stop],
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                box_padding=box_padding,
            )

        detections: list[Detections] = []
        for share in self.pool.map(len(frames), work):
            detections.extend(share)
        return detections


class ParallelClassifier(_Parallel, VideoClassifier):
    """A clip classifier replicated across devices, sharing out each batch of clips."""

    def __init__(self, replicas: Sequence[Any]) -> None:
        super().__init__(replicas)
        primary = self.pool.primary
        self.classes = list(primary.classes)
        self.num_frames = primary.num_frames
        # Probabilities from different replicas end up in one array, so the columns have to
        # mean the same thing on every device.
        for replica in self.replicas[1:]:
            if list(replica.classes) != self.classes:
                raise ValueError("Replicas of a classifier must expose identical classes.")

    def classify(self, clips: list[torch.Tensor]) -> np.ndarray:
        clips = list(clips)
        if not clips:
            return np.zeros((0, len(self.classes)), dtype=np.float32)

        def work(replica: VideoClassifier, start: int, stop: int) -> np.ndarray:
            return replica.classify(clips[start:stop])

        return np.concatenate(self.pool.map(len(clips), work), axis=0)


def build_detector(settings: Settings | None = None, devices: Sequence[str] | None = None):
    """Load the detector on each configured device, parallel only when there are several."""
    from khoroos.models.detector import ChickenDetector

    return _build(ChickenDetector, ParallelDetector, settings, devices)


def build_recognizer(settings: Settings | None = None, devices: Sequence[str] | None = None):
    """Load the action model on each configured device, parallel only when several."""
    from khoroos.models.action import ActionRecognizer

    return _build(ActionRecognizer, ParallelClassifier, settings, devices)


def _build(model, parallel, settings: Settings | None, devices: Sequence[str] | None):
    settings = settings or get_settings()
    devices = list(devices) if devices else settings.resolved_devices()
    if len(devices) == 1:
        # One device stays exactly what it was: no wrapper, no threads, no indirection.
        return model(settings=settings.with_overrides(device=devices[0]))
    logger.info("Loading %s on %s", model.__name__, ", ".join(devices))
    return parallel([model(settings=settings.with_overrides(device=device)) for device in devices])
