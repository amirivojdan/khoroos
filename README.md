<p align="center">
  <img src="khoroos_banner.png" alt="Khoroos, an open toolkit for poultry welfare analysis" width="100%">
</p>

Khoroos is an open research toolkit for video-based poultry behavior analysis. It turns poultry-house footage into individual bird trajectories, behavior timelines, and quantitative welfare indicators. Developed at the [UT Smart Agriculture Lab](https://www.ut-smartagriculture.com/).

---

## What it does

Point Khoroos at a recording of a poultry house and it will:

1. **Detect** every bird in frame — RT-DETRv2, fine-tuned on ChickenDet
2. **Track** each bird across the recording, giving it a stable identity
3. **Cut** each track into short, stable single-bird clips
4. **Classify** each clip into a 15-behaviour ethogram — V-JEPA2 ViT-L, fine-tuned on ChickenAct
5. **Summarise** the run into welfare indicators, a behaviour timeline and exportable tables

## Install

```bash
uv tool install "khoroos[web]"     # or: pip install "khoroos[web]"
khoroos models download            # fetch published weights once
```

FFmpeg must be present on the system — torchcodec decodes video through it.

> **Pre-release.** The detector is published; the fine-tuned action checkpoint is not yet.
> Until it is, point `KHOROOS_ACTION_PATH` at a local checkpoint directory
> containing the fine-tuned V-JEPA2 model and processor files.

## Docker (Ubuntu + NVIDIA GPU)

The container keeps the existing PyTorch inference path and installs the locked CUDA 13
dependencies on Ubuntu 24.04. It starts the web interface on port 8000 and downloads both
model checkpoints before accepting traffic. Checkpoints and job data live in a named volume,
so subsequent starts reuse them.

The host needs an NVIDIA driver compatible with CUDA 13 (driver 580 or newer), Docker, and
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The container does not install or replace the host driver.

The action checkpoint is currently private. Give the container a read-only Hugging Face token
at runtime; it is not copied into the image:

```bash
export HF_TOKEN=hf_your_read_token
docker compose up --build
```

Open `http://localhost:8000`. Follow startup and model-download progress with:

```bash
docker compose logs -f khoroos
```

The named `khoroos-data` volume is mounted at `/var/lib/khoroos` and contains downloaded
weights, Hugging Face cache data, uploads and results. Restarting the service preserves it:

```bash
docker compose restart khoroos
```

To use a mounted token file instead of `HF_TOKEN`, bind it read-only and set `HF_TOKEN_FILE`.
The entrypoint reads the file without adding the token to an image layer:

```bash
docker run --rm --gpus all \
  -v /secure/hf_token:/run/secrets/hf_token:ro \
  -e HF_TOKEN_FILE=/run/secrets/hf_token \
  -p 8000:8000 \
  -v khoroos-data:/var/lib/khoroos \
  khoroos:local
```

Model prefetch is enabled by default. Disable it for image diagnostics or when supplying local
checkpoint paths:

```bash
KHOROOS_PREFETCH_MODELS=0 docker compose run --rm khoroos khoroos info
```

The image can also run a batch analysis. Mount the input and output directory and override the
default web command:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN \
  -e KHOROOS_DEVICE=cuda \
  -v khoroos-data:/var/lib/khoroos \
  -v "$PWD:/work" \
  khoroos:local \
  khoroos analyze /work/farm.mp4 -o /work/results
```

Useful checks and troubleshooting:

```bash
# Confirm the GPU is visible inside the running service.
docker compose exec khoroos python -c \
  "import torch; print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"

# Confirm FFmpeg and TorchCodec are installed without downloading models.
docker run --rm -e KHOROOS_PREFETCH_MODELS=0 khoroos:local ffmpeg -version
docker run --rm -e KHOROOS_PREFETCH_MODELS=0 khoroos:local python -c \
  "import torchcodec; print(torchcodec.__version__)"
```

- `could not select device driver ... gpu`: install/configure the NVIDIA Container Toolkit,
  then restart Docker.
- `CUDA driver version is insufficient`: upgrade the host NVIDIA driver to a CUDA 13-compatible
  release.
- `401` or `Repository Not Found` during startup: confirm `HF_TOKEN` can read
  `amirivojdan/vjepa2-chicken-action`.
- To start without a GPU for diagnostics, use `docker run` without `--gpus all`, set
  `KHOROOS_DEVICE=cpu`, and disable prefetch. Full V-JEPA2 analysis on CPU will be slow.

## Use

### Web interface

```bash
khoroos ui
```

Opens `http://127.0.0.1:8000`. Drop or select a video, choose a detail level, and watch
progress live.

The results page pairs a video player with a behaviour timeline. Click the timeline to jump
the video there, or click a bird's row to follow it; the others dim, so one animal can be
watched through a crowded frame. Paths and recognised behaviours are drawn onto the video live
in the browser from the track records, so seeking stays instant and `--overlay` is only needed
when you want an annotated file to keep or send on. Windows the model was not confident about
are left unmarked — see [reading the output](#how-to-read-the-output-honestly).

The server binds to localhost. `--host 0.0.0.0` exposes it to your network, and Khoroos has no
authentication — only do that on a trusted network.

### Command line

```bash
khoroos analyze farm.mp4 -o results/
khoroos analyze farm.mp4 -o results/ --preset thorough --overlay
khoroos analyze farm.mp4 -o results/ --max-seconds 300      # first 5 minutes only
khoroos analyze farm.mp4 -o results/ --detection-batch-size 32 --action-batch-size 8
```

Writes `result.json`, `metrics.json`, `predictions.csv`, `time_budget.csv` and `per_bird.csv` —
the same bundle the web UI exports, because both call the same code.

Any analysis parameter can be set by name with `--set`, and any alert threshold with
`--threshold`. Both reject unknown names, so a typo fails instead of silently giving you a
default run:

```bash
khoroos analyze farm.mp4 -o results/ \
    --set window_seconds=3 --set detection_stride=1 \
    --threshold min_locomotion_share=0.2
```

`khoroos info` lists every parameter, preset, behaviour group and threshold, plus which
checkpoints are present. It prints exactly what the web UI builds its controls from, so the two
surfaces cannot disagree about what a run means. Add `--json` for scripts.

### Python

The CLI and the web UI are thin surfaces over the library, so anything they do is available
directly:

```python
from khoroos import analyze_video

result = analyze_video("farm.mp4", preset="balanced")

print(result.metrics["indicators"])
# {'comfort_index': 0.197, 'locomotion_score': 0.101, 'inactivity_ratio': 0.378, ...}

for prediction in result.predictions[:5]:
    print(prediction.track_id, prediction.label, round(prediction.confidence, 2))
```

`analyze_video` returns the result and stops there. To also write the export bundle and an
annotated video — what `khoroos analyze` and the web worker both do — use the runner:

```python
from khoroos import AnalysisRunner, params_for_preset, thresholds_from_overrides

runner = AnalysisRunner()                       # loads the weights once
for video in ("monday.mp4", "tuesday.mp4"):     # reused across runs
    artifacts = runner.run(
        video,
        output_dir=f"results/{video}",
        params=params_for_preset("balanced", window_seconds=3.0),
        thresholds=thresholds_from_overrides({"min_locomotion_share": 0.2}),
        on_progress=lambda event: print(event.stage, f"{event.progress:.0%}"),
    )
    print(artifacts.result.metrics["indicators"], artifacts.paths)
```

## The ethogram

15 behaviours, grouped for reporting:

| Group | Behaviours |
| --- | --- |
| **Comfort** | preening, dust bathing, wing flapping, stretching, body shaking |
| **Locomotion** | walking, running |
| **Inactive** | resting, standing |
| **Feeding & drinking** | feeding, drinking |
| **Foraging** | litter pecking, litter scratching |
| **Other** | head scratching, pooping |

## Welfare indicators

| Indicator | What it is | Why it matters |
| --- | --- | --- |
| Time-activity budget | share of bird-time per behaviour | the core ethogram output |
| Comfort behaviour index | share of comfort behaviours | suppressed comfort behaviour is associated with crowding, poor litter and stress |
| Locomotion score | share of walking + running | low locomotion with high inactivity is a leg-health / lameness signal |
| Inactivity ratio | share of resting + standing | read together with locomotion |
| Feeding & drinking | share and bout structure | nutrition, heat stress, water-line faults |
| Foraging | litter interaction | litter quality and foraging opportunity |
| Bird count & spatial map | detections per frame, occupancy grid | crowding, feeder/drinker hotspots, dead zones |

## Presets

| Preset | Detection | Window step | Detection batch | Action batch | Use for |
| --- | --- | --- | --- | --- | --- |
| `fast` | every 4th frame | 2 s | 24 | 6 | a first look at long footage |
| `balanced` | every 2nd frame | 1 s | 16 | 4 | the default |
| `thorough` | every frame | 0.5 s | 12 | 4 | maximum temporal detail |

A preset changes only how the footage is sampled and batched, never how a result is
interpreted.

**Batch sizes** are how much work each model does per forward pass. Larger is faster but holds
more in memory; lower them if a run fails with an out-of-memory error. Action clips are 64
frames each, so `action_batch_size` dominates peak memory — halve it before touching the
detector. Both are settable per run on either surface (in the browser, under **Advanced
settings → Batch sizes**), and the preset's value applies when you leave them alone.

## How to read the output honestly

Khoroos reports **indicators**, not diagnoses. Some deliberate design choices back that up:

- **Uncertain time is never hidden.** Clips the model cannot confidently label are reported as
  `uncertain` rather than folded into a class. A large share suggests the footage differs from
  the training data, and the budget should be read with caution. The one place it is not shown
  is the video overlays, which leave those windows unmarked: a label on a bird reads as an
  observation in a way a bar in a budget does not.
- **Every figure carries its sample size** in bird-seconds. Indicators computed from less than
  a minute of observation are shown but never trigger an alert.
- **Per-class reliability is shown next to the numbers.** The action model is a frozen-encoder
  linear probe trained on an imbalanced dataset; rare behaviours (`pooping`, `wing_flapping`)
  are recognised far less reliably than common ones. Alerts are suppressed for classes the
  model is measurably weak at.
- **Per-bird figures are approximate.** Identity switches happen in a dense flock, so a "bird"
  is really a track. Flock-level figures are unaffected and are the ones to quote.
- **Thresholds are yours.** The defaults are starting points; set them for your house, breed
  and age. Khoroos flags deviations, it does not decide what they mean.

## Configuration

Any setting can be overridden with a `KHOROOS_`-prefixed environment variable:

```bash
KHOROOS_DEVICE=cpu khoroos analyze farm.mp4 -o out/
KHOROOS_DETECTOR_PATH=/path/to/checkpoint khoroos models status
```

## Development

```bash
uv sync
uv run pytest          # no model weights needed
uv run ruff check src/ tests/
```

The test suite drives the full pipeline with stub models over a generated video, so it runs
anywhere in a few seconds.

### Documentation

The documentation site is built with MkDocs:

```bash
uv sync --extra docs
uv run mkdocs serve
```

Open `http://127.0.0.1:8000` while the development server is running. To perform the same
strict build used for validation, run `uv run mkdocs build --strict`.

### Architecture

The command line and the web UI are two surfaces over one library. Neither owns any analysis
logic:

```text
khoroos/cli.py            khoroos/web/         ← surfaces: arguments, progress display,
                                                 job state, HTTP. No analysis logic.
        └──────┬──────────────────┘
               ↓
    pipeline/runner.py     AnalysisRunner  — analyse → export → optional overlay
    pipeline/analyze.py    VideoAnalyzer   — the six-stage pipeline, emits ProgressEvent
    pipeline/types.py      AnalysisResult  — the versioned schema both surfaces render
    config.py              AnalysisParams, WelfareThresholds, presets
    environment.py         what this install can do — `khoroos info` and GET /api/config
    models/ tracking/ video/ welfare/
```

The rule is that anything a user can ask for must be expressible in `config.py` and executable
through `AnalysisRunner`. That is what keeps the two surfaces honest: `khoroos analyze` and the
UI's start button reach the same code, so a run means the same thing either way, and a new
capability appears in both by construction rather than by remembering to add it twice.

## Data & models

- **ChickenAct** action-recognition dataset — [Zenodo record 20672799](https://zenodo.org/records/20672799)
- Detector: RT-DETRv2 fine-tuned for single-class chicken detection
- Action model: [V-JEPA2 ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) fine-tuned on ChickenAct (64 frames, 256 px)
- Reproducible action-model training workflow — [notebook](notebooks/vjepa2_chicken_action_recognition.ipynb)

## License

See [LICENSE](LICENSE).
