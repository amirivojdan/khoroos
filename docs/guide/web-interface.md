# Web interface

Install the `web` extra and launch the local server:

```bash
python -m pip install "khoroos[web]"
khoroos ui
```

Open `http://127.0.0.1:8000`, choose a video, select a detail level, and start the analysis.
Progress updates appear while the pipeline detects, tracks, classifies, and summarizes the
footage.

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

Uploads, results, and job state are stored below the configured cache directory. Completed web
jobs expire according to `KHOROOS_JOB_TTL_HOURS`; see [Configuration](configuration.md).
