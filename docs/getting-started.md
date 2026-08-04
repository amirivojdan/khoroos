# Getting started

## Requirements

Khoroos requires Python 3.12 or newer and FFmpeg. GPU inference requires a supported PyTorch
accelerator; CPU inference is available but the action model is slow on CPU.

## Install

Install the command-line tool and web interface with `uv`:

```bash
uv tool install "khoroos[web]"
```

Or use `pip` in an existing environment:

```bash
python -m pip install "khoroos[web]"
```

Download the model checkpoints before the first analysis:

```bash
khoroos models download
khoroos models status
```

The detector checkpoint is public. If the action checkpoint is not available from the Hub,
set `KHOROOS_ACTION_PATH` to a local directory containing the fine-tuned V-JEPA2 model and
processor files.

## Verify the environment

```bash
khoroos version
khoroos info
```

`khoroos info` reports the selected device, cache location, available checkpoints, presets,
behaviour groups, and alert thresholds.

## Run a first analysis

Start with a short section of video while checking the camera view and resource use:

```bash
khoroos analyze farm.mp4 --output results/ --max-seconds 60
```

The default `balanced` preset detects on every second frame and classifies overlapping
two-second windows. When the run finishes, open `results/metrics.json` for the summary and
`results/predictions.csv` for one row per classified window.

## Next steps

- Use the [command-line guide](guide/command-line.md) for presets, overrides, and batch sizes.
- Use the [web interface](guide/web-interface.md) for interactive review.
- Read [Outputs](guide/outputs.md) before building downstream data processing.
