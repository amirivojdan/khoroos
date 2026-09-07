# Outputs

A command-line run and `AnalysisRunner.run()` write the same export bundle.

| File | Contents | Typical use |
| --- | --- | --- |
| `result.json` | Full versioned result, including tracks and predictions | Archival and application integration |
| `metrics.json` | Behavior time budgets, population summaries, and spatial statistics | Summary reporting |
| `predictions.csv` | One row per classified track window | Statistical analysis |
| `time_budget.csv` | Flock-level row per behaviour | Ethogram and budget reporting |
| `per_bird.csv` | Approximate per-track behaviour shares | Exploratory individual-level analysis |
| `annotated.mp4` | Optional rendered boxes and labels | Sharing and visual review |

## Optional tracklet exports

Set `AnalysisParams.raw_tracklets_dir` or `AnalysisParams.classified_tracklets_dir` to
enable clip exports independently. The CLI accepts these through `--set`, for example:

```bash
khoroos analyze farm.mp4 -o results/ --set raw_tracklets_dir=clips/raw --set classified_tracklets_dir=clips/classified
```

Each chosen root gets a unique `<video-name>-<run-id>` subfolder. Raw MP4s and JSON sidecars
are saved directly inside it; classified clips and sidecars go into behavior subfolders.
Custom labels that contain path separators or other special characters receive a safe,
disambiguated folder name; their exact labels remain in the JSON. Below-threshold predictions
go into `uncertain`.

These are accepted action windows, not complete bird trajectories or rejected candidate
windows. The exported frames are the cropped RGB frames passed to the classifier, before
its preprocessing, encoded as H.264. Temporal sampling matches inference, and playback rate
preserves the source window's duration. Odd crop dimensions are padded by one edge pixel for
MP4 compatibility. Encoding is lossy; these files are not lossless tensor archives.

Each sidecar records the source filename, track ID, source start/end times, frame indices of
the source window (before sampling), encoded frame count, crop dimensions, and playback rate.
Classified sidecars additionally include the assigned label, uncertainty flag, confidence,
representative box, top predictions, and all returned class probabilities.

The actual run directories are recorded in `result.json` under `params.tracklet_exports`.
Raw clips are saved before their batch is classified, so they remain available if classification
fails. Completed files from failed or cancelled runs are retained. Normal web job cleanup does
not remove exports stored outside the managed jobs directory.

## Result schema

`result.json` includes a `schema_version`. Breaking changes to its serialized structure require
a version change, so downstream consumers should check this field before parsing.

The top-level object contains:

- `video`: source metadata and analyzed duration
- `params`: effective analysis parameters
- `model`: checkpoint provenance, classes, and reliability metadata
- `tracks`: time-indexed bird boxes
- `predictions`: labels, confidence, alternatives, and boxes for each window
- `metrics`: aggregated budgets, population statistics, bouts, and spatial summaries
- `warnings`: limitations detected during the run
- `frame_counts`: time-indexed bird counts
- `runtime_s`: total analysis runtime

Coordinates are expressed as `(x1, y1, x2, y2)` pixels in the source-video coordinate system.
Times are expressed in seconds.

## Uncertain predictions

When the best action probability is below `min_confidence`, the prediction label is
`uncertain`, its `uncertain` flag is true, and the leading class probabilities remain available
in `result.json`. Do not silently discard uncertain time when computing a custom budget.

## Per-bird caution

A bird identifier represents a track, not a permanently identified animal. Occlusion and dense
flocks can cause identity switches. Flock-level summaries are more robust than comparisons
between individual track identifiers.
