<p align="center">
  <img src="https://raw.githubusercontent.com/amirivojdan/khoroos/main/khoroos_banner.png" alt="Khoroos: poultry behavior analysis" width="100%">
</p>

<p align="center">
  <a href="https://github.com/amirivojdan/khoroos/actions/workflows/test.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/amirivojdan/khoroos/test.yml?branch=main&amp;label=tests&amp;color=00A693"></a>
  <a href="https://pypi.org/project/khoroos/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/khoroos?color=00A693"></a>
  <a href="https://github.com/amirivojdan/khoroos/actions/workflows/publish.yml"><img alt="PyPI publishing workflow" src="https://img.shields.io/github/actions/workflow/status/amirivojdan/khoroos/publish.yml?label=publish&amp;color=00A693"></a>
  <a href="https://pypi.org/project/khoroos/"><img alt="Python 3.12 and newer" src="https://img.shields.io/badge/python-3.12%2B-00A693"></a>
  <a href="https://github.com/amirivojdan/khoroos/blob/main/LICENSE"><img alt="PolyForm Noncommercial License 1.0.0" src="https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-00A693"></a>
</p>

Khoroos turns poultry videos into bird tracks, behavior timelines, and descriptive statistics.
Use it to measure how observed birds spend their time, compare activity across a recording,
and export data for further analysis.

Developed at the [UT Smart Agriculture Lab](https://www.ut-smartagriculture.com/).

## Features

- **Detection and tracking:** locate chickens and follow their trajectories across frames.
- **Behavior recognition:** classify single-bird clips into 15 behaviors, or select a subset
  such as feeding and drinking.
- **Descriptive statistics:** time budgets, behavior bouts, detection counts, spatial summaries,
  and per-track measurements.
- **Interactive review:** inspect tracks and behavior timelines alongside the video in a browser.
- **Exports:** save JSON, CSV tables, and an optional annotated video.
- **Replaceable components:** bring your own detector, classifier, tracker, or statistics
  functions. Convert bounding-box annotations between YOLO and COCO formats.

## Get started

Requires Python 3.12+ and FFmpeg. A GPU is recommended for inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "khoroos[web]"
khoroos models download
khoroos ui
```

The default models are public; no Hugging Face login is required. Downloads are cached for
later runs, including offline use.

Open `http://127.0.0.1:8000`, select a video, choose a detail level, and start the analysis.
Use the timeline to seek through the recording, select a track to follow one bird, and download
the results when the run finishes.

### Docker

With Docker and the NVIDIA Container Toolkit installed:

```bash
git clone https://github.com/amirivojdan/khoroos.git
cd khoroos
docker compose up --build
```

Open `http://localhost:8000`. Startup downloads the models automatically and reuses them on
later starts. Model files and results persist in a volume; the web job list resets on restart.
See the [Docker guide](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/docker.md) for requirements and storage details.

## Command line

Start with a short section of video:

```bash
khoroos analyze farm.mp4 -o results/ --max-seconds 60
```

Analyze a full recording and save an annotated video:

```bash
khoroos analyze farm.mp4 -o results/ --overlay
```

Report selected behaviors:

```bash
khoroos analyze farm.mp4 -o results/ --set action_classes=feeding,drinking
```

Choose `--preset fast`, `balanced` (default), or `thorough` to adjust sampling detail.
If GPU memory runs out, reduce `--action-batch-size` first. See the
[CLI guide](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/command-line.md) for all options.

## Python

Analyze a video directly:

```python
from khoroos import analyze_video

result = analyze_video("farm.mp4", preset="balanced")
print(result.metrics["time_budget"])
```

Reuse the models across recordings and save each export bundle:

```python
from pathlib import Path
from khoroos import AnalysisRunner

runner = AnalysisRunner()
for video in Path("videos").glob("*.mp4"):
    runner.run(video, output_dir=Path("results") / video.stem)
```

See the [Python guide](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/python-api.md) for parameters and progress callbacks.

## Results

Each exported analysis includes:

| File | Contents |
| --- | --- |
| `result.json` | Tracks, predictions, run parameters, and metadata |
| `metrics.json` | Behavior budgets, bouts, detection counts, spatial summaries, and timelines |
| `predictions.csv` | One row per classified clip |
| `time_budget.csv` | Duration and proportion per behavior |
| `per_bird.csv` | Statistics per track |
| `annotated.mp4` | Video with boxes and labels, when `--overlay` is enabled |

Statistics describe model-assigned labels. Uncertain predictions are reported separately,
and track IDs represent trajectories rather than verified animal identities.
See [Outputs](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/outputs.md) for field definitions and duration calculations.

## Extend Khoroos

Subclass `Detector`, `VideoClassifier`, or `Tracker` to use your own algorithms.
`VideoAnalyzer` accepts custom models, and `PipelineComponents` configures the reader, tracker,
cropper, and metrics. Class labels and behavior groups are configurable.

Follow the [extension guide](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/extending.md) for working examples, or the
[configuration guide](https://github.com/amirivojdan/khoroos/blob/main/docs/guide/configuration.md) for devices, model sources, and cache paths.

## Development

From a source checkout:

```bash
uv sync --locked --extra web --extra dev --extra docs
uv run pytest
uv run ruff check src tests
uv run mkdocs serve
```

Tests use stub models and generated videos; no model download is needed.

## Resources and licenses

- [Documentation](https://github.com/amirivojdan/khoroos/blob/main/docs/index.md)
- [ChickenAct dataset](https://zenodo.org/records/20672799) and [training notebook](https://github.com/amirivojdan/khoroos/blob/main/notebooks/vjepa2_chicken_action_recognition.ipynb)
- Public models: [chicken detector](https://huggingface.co/amirivojdan/chicken_rtdetrv2) and [action classifier](https://huggingface.co/amirivojdan/chicken_vjepa2_action), licensed under CC BY-NC-SA 4.0.
- Code: [PolyForm Noncommercial License 1.0.0](https://github.com/amirivojdan/khoroos/blob/main/LICENSE).
