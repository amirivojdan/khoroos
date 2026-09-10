# Web interface

Complete [installation and model download](../getting-started.md), then launch the server:

```bash
khoroos ui
```

Open `http://127.0.0.1:8000`, choose a video, select a detail level, and start the analysis.
Progress updates appear while the pipeline detects, tracks, classifies, and summarizes the
footage.

## Choose a device

Use **Run on** before starting an analysis to select **Automatic**, **CPU**, an available
GPU (listed by index and model name), or **Apple GPU** when available. Automatic prefers
CUDA, then Apple GPU, then CPU. CPU inference can be slow.

The list describes hardware visible to the Khoroos server, not the computer displaying
the browser. In Docker, only GPUs exposed to the container appear, and GPU indices are
local to that container.

The initial selection follows the server's `--device` or `KHOROOS_DEVICE` setting. For example:

```bash
khoroos ui --device cuda:1
```

Each queued job keeps its selected device, and both models run on that device. Jobs still
run one at a time. Consecutive jobs on the same device reuse loaded models; switching
devices reloads them and adds startup time. Unavailable selections are rejected, and
execution errors are reported instead of silently falling back to another device.

Applications embedding a custom runner must configure its device directly; the web worker
does not relocate or replace injected models when a different device is requested.

Under advanced options, **Behaviors of interest** lists the active classifier's behaviors
as checkboxes. All are selected by default; clear the selection and click the behaviors
you want, or uncheck those you do not need. Select at least one behavior. If model metadata
is not installed yet, analysis includes all behaviors; reload after the first analysis to
make the choices available.

## Save tracklet clips

In **Advanced settings**, enable **Save raw tracklets**, **Save classified tracklets**, or
both. Each checkbox reveals its own destination field and **Browse** button. Browse folders
on the machine running Khoroos, or type a new directory to create when analysis starts.

The defaults are `KHOROOS_CACHE_DIR/tracklets/raw` and
`KHOROOS_CACHE_DIR/tracklets/classified` (normally under `~/.cache/khoroos`). Each analysis
creates a unique subfolder so repeated runs do not overwrite clips. These defaults also
live in the persistent data volume when using the provided Docker Compose configuration.
In Docker, custom destinations must be paths inside the container; use a bind mount to
save directly into a host folder.

Raw exports contain the cropped, temporally sampled clips supplied to the classifier.
Classified exports contain those same clips grouped into predicted behavior folders,
including `uncertain`, with a JSON file containing predictions and source information beside
each MP4. Labels are not burned into the video. See [Outputs](outputs.md) for format details.

Both options are off by default. Saving clips adds encoding time and disk usage. Completed
results display the actual export directories. Saved clips remain after the web job expires
or is deleted, and partial runs can leave completed clips for review.

## Review results

The results page pairs the video with a behaviour timeline. Select a point on the timeline to
seek the video, or select a bird row to emphasize one track. Tracks and recognized behaviours
are rendered in the browser, so seeking does not require a pre-rendered overlay.

Windows below the configured confidence threshold are left unmarked in the video and are
reported as `uncertain` in the data exports.

## Network access

The server binds to localhost by default. To access it from another device on a trusted
network:

```bash
khoroos ui --host 0.0.0.0 --port 8000
```

!!! danger "No built-in authentication"

    Khoroos does not authenticate web users. Do not expose the server to the public internet.
    Put it behind an authenticated reverse proxy if remote access is required.

## Storage

Uploads and result files are stored below the configured cache directory. The job list and
queue live in memory: restarting the server clears the list and interrupts pending work.
Saved files remain on disk but are not restored into the UI. Export needed results before
restarting.

During a server session, terminal jobs expire `KHOROOS_JOB_TTL_HOURS` after completion.
The worker checks between analyses and every minute while idle. Files left from earlier
sessions are not automatically cleaned up. See
[Configuration](configuration.md).
