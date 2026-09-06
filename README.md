<p align="center">
  <img src="khoroos_banner.png" alt="Khoroos: poultry behavior analysis" width="100%">
</p>

Khoroos is an open research toolkit for video-based poultry behavior analysis. It turns poultry-house footage into individual bird trajectories, behavior timelines, and quantitative welfare statistics.

Developed at the [UT Smart Agriculture Lab](https://www.ut-smartagriculture.com/).

## Install

Requires Python 3.12+ and FFmpeg. A GPU is recommended for inference.

```bash
git clone https://github.com/amirivojdan/khoroos.git
cd khoroos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[web]"
```

Both default models are **private**. Your Hugging Face account needs read access to:

| Model | Repository |
| --- | --- |
| Chicken detector | [amirivojdan/chicken_rtdetrv2](https://huggingface.co/amirivojdan/chicken_rtdetrv2) |
| 15-class action classifier | [amirivojdan/chicken_vjepa2_action](https://huggingface.co/amirivojdan/chicken_vjepa2_action) |

Authenticate and download the weights:

```bash
hf auth login
khoroos models download
khoroos models status
```

You can also authenticate with `HF_TOKEN`. Downloads are cached per repository and reused
offline. Khoroos does not search the old `notebooks/checkpoints/` folders.

## Analyze a video

Start the web interface:

```bash
khoroos ui
```

Open `http://127.0.0.1:8000`, select a video, and run an analysis. The results show tracks,
a behavior timeline, and downloadable statistics.

Or use the command line:

```bash
khoroos analyze farm.mp4 -o results/
```

Common options:

```bash
# Analyze the first minute.
khoroos analyze farm.mp4 -o results/ --max-seconds 60

# Use denser sampling and save an annotated video.
khoroos analyze farm.mp4 -o results/ --preset thorough --overlay

# Select the behavior classes to report.
khoroos analyze farm.mp4 -o results/ --set action_classes=feeding,drinking
```

The default preset is `balanced`; `fast` samples less often and `thorough` samples more often.
If GPU memory runs out, reduce `--action-batch-size` first.

### Python

```python
from khoroos import analyze_video

result = analyze_video("farm.mp4", preset="balanced")
print(result.metrics["time_budget"])
```

Use `AnalysisRunner` to reuse models across videos and save exports. See the
[Python guide](docs/guide/python-api.md).

### Docker

With Docker and the NVIDIA Container Toolkit installed:

```bash
export HF_TOKEN=hf_your_read_token
docker compose up --build
```

Open `http://localhost:8000`. Models and results persist in the `khoroos-data` volume.
The web job list is reset on restart.
See the [Docker guide](docs/guide/docker.md) for host requirements, batch runs, and troubleshooting.

## Outputs

| File | Contents |
| --- | --- |
| `result.json` | Tracks, predictions, run parameters, and metadata |
| `metrics.json` | Time budgets, groups, bouts, population counts, spatial summaries, and timelines |
| `predictions.csv` | One row per classified clip |
| `time_budget.csv` | Duration and proportion per behavior |
| `per_bird.csv` | Statistics per track |

These are model-derived measurements. Uncertain predictions are reported separately; missed
detections and identity switches can affect track summaries. See [Outputs](docs/guide/outputs.md)
for field definitions and duration calculations.

## Customize

Subclass `Detector`, `VideoClassifier`, or `Tracker` and pass your implementation to
`VideoAnalyzer`. `PipelineComponents` supplies the tracker, video reader, cropper, and metrics
functions. YOLO and COCO bounding-box codecs are available in `khoroos.annotations`.

The [extension guide](docs/guide/extending.md) covers the required methods and working examples.
The [configuration guide](docs/guide/configuration.md) covers model repositories, devices,
cache paths, and analysis parameters.

## Development

```bash
uv sync --extra web --extra dev --extra docs
uv run pytest
uv run ruff check src tests
uv run mkdocs serve
```

Tests use stub models and generated videos; no model download is needed.

## Data and training

- [ChickenAct dataset](https://zenodo.org/records/20672799)
- [Action-model training notebook](notebooks/vjepa2_chicken_action_recognition.ipynb)
- [Documentation](docs/index.md)
- [PolyForm Noncommercial License 1.0.0](LICENSE)
