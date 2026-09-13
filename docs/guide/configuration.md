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
| `KHOROOS_DEVICE` | `auto` | `cuda`, `cpu`, `mps`, automatic selection, or several devices: `all` / `cuda:0,cuda:1` |
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

By default, both models come from the public Hugging Face repositories listed above.
Run `khoroos models download`; no login or token is required for these models.
`khoroos models status` shows the checkpoint paths that inference will use.

If you select a private replacement repository, authenticate with `hf auth login` or set
`HF_TOKEN` with read access to that repository.

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

Device syntax is validated before checkpoint resolution: `auto`, `all`, `cpu`, `mps`, `cuda`,
`cuda:<nonnegative index>`, `cuda:all`, or a comma-separated list of those. Hardware
availability is checked when the selected backend is used.
The [web interface](web-interface.md#choose-a-device) also supports choosing an available
device per job; the process setting supplies the initial selection.
Assignment is validated too. Use `settings.with_overrides(device="cpu")` to create a validated
copy; the CLI uses this approach so overrides do not mutate global defaults.

## Region of interest

`roi` restricts an analysis to a rectangle of the frame — a feeder, a drinker, one pen —
given as `x1,y1,x2,y2` in **fractions of width and height** rather than pixels, so the same
setting survives a change of recording resolution:

```bash
khoroos analyze farm.mp4 --roi 0.1,0.25,0.8,0.9
```

In the web interface, drag a rectangle over the preview frame under **Region of interest**.

The detector still runs on the whole frame at the scale it was trained on; the region
decides which of its findings are kept. **A bird belongs to the region when the centre of
its box falls inside it** — centre rather than any-overlap, so a bird at the boundary counts
for whichever side it is mostly on instead of flickering in and out as its box grows and
shrinks between frames. Everything downstream follows from the filtered detections, so
tracks, tracklets, behaviour predictions, counts and the time budget are all restricted to
the region without any further configuration.

Two consequences worth knowing:

- **It does not make the analysis faster.** The detector's preprocessor resizes every frame
  to a fixed size, so the model does the same work whether or not a region is set. A region
  is for asking a narrower question, not for saving time.
- **A bird that walks out of the region leaves the analysis**, and its track ends there. A
  region that cuts through an area birds move across will fragment tracks.

Every run records the region it used in `result.json` under `params.roi`, and the run
warns how many detections it kept and ignored — or says plainly when the region held none,
which usually means the rectangle is in the wrong place.

## Using several GPUs

A spec naming more than one device splits **each analysis** across them: both models are
loaded once per device, and every batch — detector frames and action clips alike — is cut
into one contiguous share per device, run concurrently, and reassembled in input order.

```bash
khoroos analyze farm.mp4 --device all          # every GPU on the machine
khoroos analyze farm.mp4 --device cuda:0,cuda:1  # two named GPUs
```

`auto` still means *one* best device, so a machine that grows a second GPU does not
silently change what existing configurations do. Using every GPU is always opt-in.

Because shares are cut from the batch rather than each device receiving a whole batch,
`detection_batch_size` and `action_batch_size` stay **totals**: per-device memory matches
what the same configuration uses on one GPU. Two GPUs therefore halve the batch each one
sees. To give each GPU the batch it would have had alone, multiply both by the device count:

```bash
khoroos analyze farm.mp4 --device all --detection-batch-size 32 --action-batch-size 8
```

This matters most for `action_batch_size`, which defaults to 4 and would otherwise leave
each of two GPUs working on two clips at a time.

Both stages benefit. Detection is a straight batch split. In the classify stage only the
forward pass is shared out — clips are still cut from the video one at a time on the main
thread — but that extraction is a small part of the stage (around 6% in a local
measurement, against a V-JEPA2 forward pass of roughly 380 ms per clip), so the ceiling on
two GPUs stays close to 2x there too.

### Reproducibility

Splitting changes no arithmetic of its own: given the same share, a device returns exactly
what it would have returned alone. But each forward pass now sees a smaller batch, and
batched GPU inference is not bit-exact across batch shapes — the same effect you get today
by changing `detection_batch_size`. Scores shift in their last decimal places, which can
flip a detection sitting right on `detection_confidence`. Aggregate behaviour is unaffected,
but reproducing a run exactly requires the same device count as well as the same batch sizes.

A device named twice is used once, and one unavailable device rejects the whole list.
Mixing a CPU into a GPU list is accepted but rarely worth it: every batch waits for its
slowest share.

Behavior groups are configured with `Settings.behaviour_groups` or per-run
`AnalysisParams.behaviour_groups`. An empty mapping disables them. `invalid_boxes` selects
strict geometry validation (`error`, the default) or counted removal (`drop`) for raw detector
outputs. See [Replacing components](extending.md) for normalization and grouping contracts.
