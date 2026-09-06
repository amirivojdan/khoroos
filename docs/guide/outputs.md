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
