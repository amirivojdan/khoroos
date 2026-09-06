# Getting started

## Requirements

Khoroos requires Python 3.12 or newer and FFmpeg. GPU inference requires a supported PyTorch
accelerator; CPU inference is available but the action model is slow on CPU.

## Install

Install the command-line tool and web interface from this repository:

```bash
git clone https://github.com/amirivojdan/khoroos.git
cd khoroos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[web]"
```

The default [detector](https://huggingface.co/amirivojdan/chicken_rtdetrv2) and
[action classifier](https://huggingface.co/amirivojdan/chicken_vjepa2_action) are private.
Authenticate with an account that has read access to both, then download the weights:

```bash
hf auth login
khoroos models download
khoroos models status
```

`HF_TOKEN` can be used instead of interactive login. Downloads are cached per repository
and reused offline. Old notebook checkpoint folders are not searched.

## Verify the environment

```bash
khoroos version
khoroos info
```

`khoroos info` reports the selected device, cache location, available checkpoints, presets,
and behaviour groups.

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
