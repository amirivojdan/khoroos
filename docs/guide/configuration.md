# Configuration

Khoroos has two layers of configuration:

| Layer | Scope | How to set it |
| --- | --- | --- |
| Process settings | Device, paths, cache, and web server | `KHOROOS_*` environment variables |
| Analysis parameters | Detection, tracking, clips, and classification | Preset, CLI `--set`, or `AnalysisParams` |

## Environment variables

Every `Settings` field has a `KHOROOS_`-prefixed environment variable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `KHOROOS_DEVICE` | `auto` | `cuda`, `cpu`, `mps`, or automatic selection |
| `KHOROOS_USE_BF16` | `true` | Use bfloat16 autocast on CUDA |
| `KHOROOS_CACHE_DIR` | Platform cache directory | Model, upload, and job storage |
| `KHOROOS_DETECTOR_REPO` | `amirivojdan/chicken_rtdetrv2` | Hugging Face detector repository |
| `KHOROOS_ACTION_REPO` | `amirivojdan/chicken_vjepa2_action` | Hugging Face action repository |
| `KHOROOS_DETECTOR_PATH` | unset | Local detector checkpoint directory |
| `KHOROOS_ACTION_PATH` | unset | Local action checkpoint directory |
| `KHOROOS_HOST` | `127.0.0.1` | Web bind address |
| `KHOROOS_PORT` | `8000` | Web port |
| `KHOROOS_MAX_UPLOAD_MB` | `4096` | Maximum web upload size |
| `KHOROOS_JOB_TTL_HOURS` | `24` | Hours to retain terminal web jobs after completion, within a server session |

For example:

```bash
KHOROOS_DEVICE=cpu khoroos analyze farm.mp4 --output results/
KHOROOS_ACTION_REPO=my-account/my-classifier khoroos models download
```

Boolean settings accept values supported by Pydantic settings, such as `true` and `false`.

## Model sources

By default, both models come from the private Hugging Face repositories listed above.
Run `hf auth login` or set `HF_TOKEN` with read access, then run `khoroos models download`.
`khoroos models status` shows the checkpoint paths that inference will use.

Hub snapshots are cached under `KHOROOS_CACHE_DIR/weights`, separated by repository and
revision. A cached checkpoint is reused offline. Changing a repository setting selects that
repository's cache; the old `weights/detector`, `weights/action`, and `notebooks/checkpoints`
directories are ignored.

For a custom local model, explicitly set `KHOROOS_DETECTOR_PATH` or `KHOROOS_ACTION_PATH`,
or pass `checkpoint=` to the model constructor. Explicit paths take precedence over the Hub.
Unset these variables to return to the default repositories.

## Analysis parameters

`AnalysisParams` validates confidence thresholds, batch sizes, tracking limits, clip geometry,
and run duration. Use `khoroos info` to inspect the current defaults or see the
[configuration API reference](../reference/config.md).

Named presets establish a starting point. Explicit overrides are applied after the preset, so
they always win.

## Device validation and reusable settings

Device syntax is validated before checkpoint resolution: `auto`, `cpu`, `mps`, `cuda`, or
`cuda:<nonnegative index>`. Hardware availability is checked when the selected backend is used.
Assignment is validated too. Use `settings.with_overrides(device="cpu")` to create a validated
copy; the CLI uses this approach so overrides do not mutate global defaults.

Behavior groups are configured with `Settings.behaviour_groups` or per-run
`AnalysisParams.behaviour_groups`. An empty mapping disables them. `invalid_boxes` selects
strict geometry validation (`error`, the default) or counted removal (`drop`) for raw detector
outputs. See [Replacing components](extending.md) for normalization and grouping contracts.
