# Understanding the statistics

Khoroos extracts measurements from detections, tracks, and model-assigned behavior labels.
It does not infer welfare status or recommend interventions.

## Duration and proportions

`time_budget.total_bird_seconds` is the combined observation duration across tracks.
Overlapping classification windows split their overlap halfway between their centers, so
budgets do not count overlapping bird-time twice. Gaps remain unobserved.

Each class has a duration, a share of all observed time, a share of confidently labeled time,
and a window count. Uncertain time remains a separate category. Proportions describe the
sampled track windows, not the entire flock or all frames in a recording.

## Counts and bouts

Population statistics summarize raw detections on sampled frames. Track count is the number
of confirmed trajectories, not a verified count of unique animals.

Bouts combine adjacent same-label intervals within a track. Reported bout counts and durations
depend on the window size, sampling stride, confidence cutoff, and tracking continuity.

## Spatial summaries

The spatial grid counts classified windows by their representative box centers and reports
the dominant predicted behavior in each occupied cell. Counts are window counts, not seconds
of occupancy. No interpretation is attached to the spatial distribution.

## Model metadata and processing notes

Confidence, uncertain share, and observation duration document how measurements were produced.
Per-class evaluation F1 and test support are included only when supplied in a checkpoint's
`model_card.json` or by an injected classifier; a Hugging Face README alone does not supply
these values. Processing notes report
missing tracks, rejected clips, and lost trajectories. They do not produce domain alerts.

## Schema migration

Result schema **2.0** removes `metrics.indicators`, `metrics.alerts`, and `metrics.thresholds`.
Descriptive behavior groups use `maintenance` in place of `comfort`.
`khoroos.statistics` replaces `khoroos.welfare`; update metrics and export imports.
`WelfareThresholds`, `thresholds_from_overrides`, the `thresholds` Python argument, and the
CLI `--threshold` option have been removed. Custom metrics callables now receive only
`classes`, `behaviour_groups`, and `bin_seconds` as keyword options after predictions, video, and frame counts.
