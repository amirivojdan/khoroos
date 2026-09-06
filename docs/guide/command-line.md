# Command line

## Analyze a video

```bash
khoroos analyze VIDEO [OPTIONS]
```

Common examples:

```bash
khoroos analyze farm.mp4 --output results/
khoroos analyze farm.mp4 --output results/ --preset thorough --overlay
khoroos analyze farm.mp4 --output results/ --max-seconds 300
khoroos analyze farm.mp4 --device cuda --detection-batch-size 32 --action-batch-size 8
```

The `--overlay` option writes an annotated video in addition to the data exports. It is not
needed for the browser visualization, which draws tracks and behaviour labels live.

## Presets

| Preset | Detection stride | Window step | Detection batch | Action batch | Suggested use |
| --- | ---: | ---: | ---: | ---: | --- |
| `fast` | 4 frames | 2 s | 24 | 6 | First look at long footage |
| `balanced` | 2 frames | 1 s | 16 | 4 | Default analysis |
| `thorough` | 1 frame | 0.5 s | 12 | 4 | Maximum temporal detail |

A preset controls sampling and batching, not how the result is interpreted.

## Override parameters

Frequently changed settings have named options. Every analysis parameter can also be set with
repeatable `--set NAME=VALUE` options:

```bash
khoroos analyze farm.mp4 --output results/ \
  --set window_seconds=3 \
  --set detection_stride=1 \
  --set min_crop_size=120
```

Use `khoroos info` to inspect presets and checkpoint availability.

## Tune memory use

Batch sizes control how much work each model performs in one forward pass. Larger batches can
be faster but require more accelerator memory. Action clips contain 64 frames, so
`action_batch_size` usually dominates peak memory. If a run exhausts memory, halve the action
batch first, then reduce the detection batch:

```bash
khoroos analyze farm.mp4 --action-batch-size 2 --detection-batch-size 8
```

## Model management

```bash
khoroos models status
khoroos models download
```

Downloaded checkpoints are stored in the configured cache directory and reused by later runs.
