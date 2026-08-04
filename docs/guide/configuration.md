# Configuration

Khoroos has three layers of configuration:

| Layer | Scope | How to set it |
| --- | --- | --- |
| Process settings | Device, paths, cache, and web server | `KHOROOS_*` environment variables |
| Analysis parameters | Detection, tracking, clips, and classification | Preset, CLI `--set`, or `AnalysisParams` |
| Welfare thresholds | Alert conditions and reliability gates | CLI `--threshold` or `WelfareThresholds` |

## Environment variables

Every `Settings` field has a `KHOROOS_`-prefixed environment variable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `KHOROOS_DEVICE` | `auto` | `cuda`, `cpu`, `mps`, or automatic selection |
| `KHOROOS_USE_BF16` | `true` | Use bfloat16 autocast on CUDA |
| `KHOROOS_CACHE_DIR` | Platform cache directory | Model, upload, and job storage |
| `KHOROOS_DETECTOR_PATH` | unset | Local detector checkpoint directory |
| `KHOROOS_ACTION_PATH` | unset | Local action checkpoint directory |
| `KHOROOS_HOST` | `127.0.0.1` | Web bind address |
| `KHOROOS_PORT` | `8000` | Web port |
| `KHOROOS_MAX_UPLOAD_MB` | `4096` | Maximum web upload size |
| `KHOROOS_JOB_TTL_HOURS` | `24` | Retention time for completed web jobs |

For example:

```bash
KHOROOS_DEVICE=cpu khoroos analyze farm.mp4 --output results/
KHOROOS_ACTION_PATH=/models/action khoroos models status
```

Boolean settings accept values supported by Pydantic settings, such as `true` and `false`.

## Analysis parameters

`AnalysisParams` validates confidence thresholds, batch sizes, tracking limits, clip geometry,
and run duration. Use `khoroos info` to inspect the current defaults or see the
[configuration API reference](../reference/config.md).

Named presets establish a starting point. Explicit overrides are applied after the preset, so
they always win.

## Alert thresholds

Thresholds control whether computed indicators produce warnings. They do not change detection,
tracking, classification, or the numeric indicators themselves. Alerts are suppressed when the
observation time or measured per-class reliability is below its configured minimum.
