# Python API

## Analyze one video

The high-level function returns an `AnalysisResult` without writing files:

```python
from khoroos import analyze_video

result = analyze_video("farm.mp4", preset="balanced")

print(result.video.duration_seconds)
print(result.metrics["time_budget"])
for prediction in result.predictions[:5]:
    print(prediction.track_id, prediction.label, prediction.confidence)
```

## Configure a run

Build validated parameters from a preset and explicit overrides:

```python
from khoroos import analyze_video, params_for_preset

params = params_for_preset(
    "balanced",
    window_seconds=3.0,
    detection_stride=1,
)

result = analyze_video("farm.mp4", params=params)
```

## Write the export bundle

`AnalysisRunner` performs the same analyze-and-export sequence as the command line and web
worker. Reusing one runner also reuses the loaded model weights:

```python
from khoroos import AnalysisRunner, params_for_preset

runner = AnalysisRunner()
params = params_for_preset("balanced")

for video in ("monday.mp4", "tuesday.mp4"):
    artifacts = runner.run(
        video,
        output_dir=f"results/{video}",
        params=params,
        render_overlay=False,
        on_progress=lambda event: print(event.stage, f"{event.progress:.0%}"),
    )
    print(artifacts.paths)
```

## Cancel long-running work

Library callers can provide a callback that returns `True` when the pipeline should stop:

```python
from threading import Event

from khoroos import AnalysisRunner

cancelled = Event()
runner = AnalysisRunner()
runner.run(
    "farm.mp4",
    output_dir="results/",
    should_cancel=cancelled.is_set,
)
```

See the [API reference](../reference/index.md) for signatures and result types.

For custom detectors, classifiers, trackers, annotation formats, and class selection, see
[Replacing pipeline components](extending.md).
