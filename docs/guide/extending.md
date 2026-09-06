# Replacing pipeline components

Khoroos separates algorithms from orchestration. The default pipeline still uses RT-DETRv2,
SORT tracking, stable sliding-window crops, and V-JEPA2. You can replace each stage without
editing the analyzer, CLI, or web worker.

## Module boundaries

| Module | Responsibility | Extension point |
| --- | --- | --- |
| `interfaces.py` | Parent classes and model metadata conventions | `Detector`, `VideoClassifier`, `Tracker`, `VideoReader` |
| `annotations/` | Validate and translate per-image detection annotations | `AnnotationCodec` |
| `models/` | Model preprocessing, inference, checkpoint loading | Inject detector and recognizer instances |
| `tracking/` | Association, track windows, crop extraction | Tracker factory, tracklet builder, clip extractor |
| `pipeline/components.py` | Compose the stages and their defaults | `PipelineComponents` |
| `pipeline/analyze.py` | Sequence stages, progress, cancellation, results | `VideoAnalyzer` |
| `statistics/` | Descriptive behavior summaries and exports | Metrics function and exporter |
| `pipeline/runner.py` | Analyze, export, render | `AnalysisRunner` |
| `web/`, `cli.py` | User interaction | Share the analyzer and runner |

Model instances are reused across videos. Readers and trackers are created **per video**.
The analyzer closes its reader on success, failure, cancellation, or explicit generator
closure. Close an `iter_analyze` generator if you stop consuming it early. Analyzer instances
are intended for sequential use; create separate instances for concurrent runs.

## Implement a detector or classifier

Derive from the public parent classes. Constructors are yours to design: no checkpoint,
GPU, framework, registry registration, or model-card file is required.

```python
import numpy as np

from khoroos import Detector, VideoClassifier
from khoroos.annotations import Detections


class MyDetector(Detector):
    def __init__(self, engine):
        self.engine = engine

    def detect(self, images, confidence_threshold=0.3,
               nms_threshold=0.6, box_padding=0.0):
        # Adapt your engine's API here, applying the supplied filtering and padding.
        results = self.engine.predict(
            images, confidence=confidence_threshold,
            nms=nms_threshold, padding=box_padding,
        )
        return [Detections(r.boxes_xyxy, r.scores, r.class_ids) for r in results]


class MyClassifier(VideoClassifier):
    def __init__(self, engine, classes):
        self.engine = engine
        self.classes = list(classes)  # Must match engine probability-column order.
        self.num_frames = 32

    def classify(self, clips):
        return np.asarray(self.engine.predict_probabilities(clips), dtype=np.float32)
```

`engine` above represents your own inference implementation; adapt those calls to its API.
The classifier must return an `(N, C)` array of finite probabilities in `[0, 1]`, with rows
summing to at most one. Full-vocabulary softmax outputs normally sum to one. Clips are RGB
`uint8` tensors shaped `(T, C, H, W)`. Leave `num_frames = None` to receive all crop frames.
Your classifier owns resizing, normalization, and any padding or additional frame sampling.

A detector returns one result per input frame, including empty frames. Boxes always use
pixel `xyxy`, scores have shape `(N,)`, and empty boxes have shape `(0, 4)`. Existing
`(boxes, scores)` tuple implementations remain supported through `Detections.coerce` at the
pipeline boundary; new detectors should return `Detections`. `Detections` additionally retains
category IDs for annotation interchange and can be unpacked as `boxes, scores = result`.
Canonical detections own immutable arrays, so input aliases cannot change a validated result.
The tracking interface follows one population: filter a general-purpose detector to the
species of interest before passing it to this pipeline. It does not associate by category.

Optional model attributes: `name` supplies provenance; `model_card` supplies a dictionary
with a `classification_report`. Otherwise provenance uses the implementation class name.
Existing checkpoint-based models still load their adjacent `model_card.json`.

## Replace tracking and other stages

A tracker implements two methods and may report `lost_track_count` (default zero):

```python
from khoroos import Tracker


class MyTracker(Tracker):
    def __init__(self, engine):
        self.engine = engine

    def update(self, frame_index, time_seconds, boxes, scores):
        # Adapt engine observations to source frame indices and timestamps.
        return self.engine.update(frame_index, time_seconds, boxes, scores)

    def finalize(self):
        # Return khoroos.pipeline.types.Track objects, with unique IDs and
        # TrackObservation objects ordered by source frame index.
        return self.engine.to_khoroos_tracks()
```

Wire your implementations at construction time:

```python
from khoroos import AnalysisRunner, PipelineComponents, VideoAnalyzer

# detector, classifier and make_tracking_engine are your application objects.
components = PipelineComponents(
    tracker_factory=lambda params: MyTracker(make_tracking_engine(params)),
)
analyzer = VideoAnalyzer(
    detector=detector, recognizer=classifier, components=components,
)
runner = AnalysisRunner(analyzer=analyzer)
artifacts = runner.run("farm.mp4", "results/")
```

A tracker receives updates only on detector frames. Use the supplied source timestamps if
your motion model needs elapsed time. The default tracker factory converts `track_max_age`
from source frames to tracker updates; custom factories decide how to interpret their own
algorithm parameters.

`PipelineComponents` also accepts `source_factory`, `tracklet_builder`, `clip_extractor`,
and `metrics`. These are ordinary callables; their exact contracts and defaults are in
`pipeline/components.py`. A custom reader derives from `VideoReader`. A custom metrics
function receives predictions, video metadata, frame counts, and keyword arguments
`classes`, `behaviour_groups`, and `bin_seconds`, and returns a JSON-compatible dict.

`AnalysisRunner` accepts `exporter(result, output_dir)` and
`overlay_renderer(video_path, result, output_path, max_seconds=...)`. The exporter returns
an artifact-name-to-Path mapping. The built-in renderer uses the analyzer's `source_factory`
and reads dimensions and frame rate from `VideoReader.info`. It creates and closes a separate
reader for the rendering pass, including on encoder failure. A supplied custom renderer owns
its own input handling; its existing function signature is unchanged.

To use the same composition in the web service:

```python
from khoroos.web.app import create_app

app = create_app(runner=runner)
```

The built-in web visualizations and download routes expect the standard metrics and export
keys. Additional custom outputs are available through the Python runner; custom UI views
and download routes are application code.

## Select classes of interest

Selection is explicit, ordered, and validated against the classifier's declared vocabulary:

```python
from khoroos import AnalysisParams

result = analyzer.analyze(
    "farm.mp4",
    params=AnalysisParams(action_classes=["feeding", "drinking"]),
)
```

The same setting works through the CLI:

```bash
khoroos analyze farm.mp4 --set action_classes=feeding,drinking
```

The web API accepts `action_classes=feeding,drinking` as a form field. In the browser,
use the optional **Behaviors of interest** input.

For standalone use, wrap any classifier with `SelectedClasses(classifier, classes)`.
The built-in `ActionRecognizer(classes=[...])` also supports selection directly. Its full
vocabulary comes from checkpoint `id2label`; custom checkpoints need not use ChickenAct.
Passing classes selects existing outputs; it does not retrain a classifier or rename logits.

The selected class with the highest original probability wins. Probabilities are **not
renormalized**: selecting only a class with probability 0.08 still gives confidence 0.08.
The usual confidence threshold determines whether that prediction becomes `uncertain`.
Per-run selection never mutates the cached model. Unknown, empty, or duplicate label lists
raise `ValueError`. The label `uncertain` is reserved.

Time budgets, timelines, spatial summaries, and population statistics support custom labels.
Default metrics are descriptive; no welfare scores, alerts, or reference thresholds are applied.
Inject a metrics function to compute additional measurements.

## Configure descriptive behavior groups

Groups are optional descriptive aggregations. Supply defaults on `Settings`, override them
per run through `AnalysisParams.behaviour_groups`, or pass `{}` to disable grouping.
Membership must be disjoint; `uncertain` and `ungrouped` are reserved group names.

```python
from khoroos import Settings, AnalysisParams, VideoAnalyzer

settings = Settings(behaviour_groups={"movement": ["walking", "running"]})
analyzer = VideoAnalyzer(settings=settings, recognizer=classifier)
result = analyzer.analyze(
    "farm.mp4", params=AnalysisParams(behaviour_groups={"intake": ["feeding", "drinking"]}),
)
```

Only members in the selected vocabulary contribute to group summaries. Effective grouping
is recorded in the result parameters, and the browser builds its legend from result groups.
The CLI accepts JSON with `--set 'behaviour_groups={"intake":["feeding","drinking"]}'`;
the web API accepts that JSON as the `behaviour_groups` form field.

Environment descriptions use injected classifier labels or local checkpoint `config.json`
metadata, without loading weights or downloading anything. If metadata is absent, `classes`
is empty and `classes_source` is `unavailable`; invalid metadata also sets `metadata_error`.
The built-in `ACTION_CLASSES` remains a reference constant, not the advertised vocabulary of
an unknown or custom model. The web app describes the classifier on its injected runner.

## Handle invalid detections explicitly

The built-in detector clamps all box coordinates to the frame, including with zero padding.
It excludes invalid raw geometry and boxes with no area after clamping, recording their number
in `Detections.dropped_count`. The pipeline includes the total in result processing notes.

For legacy tuple outputs, `AnalysisParams(invalid_boxes="error")` is the default. Set it to
`"drop"` to remove non-finite or degenerate boxes and report how many were removed. Canonical
detector implementations can use `Detections.from_raw(..., invalid_boxes="drop")` themselves.
Malformed array shapes, scores, and category IDs always raise errors; they are not recoverable
box geometry. Confidence filtering and NMS are separate and do not count as invalid boxes.

## Switch annotation protocols

Annotation serialization is independent of detector architecture. The internal tracker and
cropper always receive pixel `xyxy` boxes. Use a codec at the boundary:

```python
from khoroos.annotations import CocoCodec, YoloCodec

# Map YOLO class 0 to category 7 and YOLO class 1 to category 42.
yolo = YoloCodec(category_ids=[7, 42])
coco = CocoCodec(image_id=12, annotation_id=100)

yolo_text = yolo.encode(detections, width=1280, height=720)
detections = yolo.decode(yolo_text, width=1280, height=720)
coco_records = coco.encode(detections, width=1280, height=720)

# All Detector subclasses inherit this convenience method.
records_per_frame = detector.detect_annotations(
    frames, coco, image_sizes=[(1280, 720)] * len(frames),
)
```

YOLO uses standard five-column normalized detection labels and cannot preserve confidence
scores; decoding assigns 1.0. See the [Ultralytics format documentation](https://docs.ultralytics.com/datasets/detect/).
COCO records preserve category IDs and scores, use pixel `x, y, width, height`, and are
compatible with the bounding-box conventions in the [COCO API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).

These are **per-image bounding-box codecs**, not dataset loaders or training pipelines.
For COCO datasets, decode `document["annotations"]` with the desired `image_id`; other images
are ignored. When assembling a dataset, provide its `images` and `categories` metadata and
allocate unique image and annotation IDs across calls. Segmentation, crowd metadata, and
keypoints are not round-tripped. YOLO class-name metadata is likewise managed by the caller.
Out-of-bounds or degenerate annotations are rejected rather than silently clipped.

Use `annotation_codec("yolo", category_ids=[...])` or `annotation_codec("coco", image_id=...)`
for configuration-driven selection. Add another protocol by deriving from `AnnotationCodec`
and implementing `encode` and `decode`; pass the resulting instance directly.
