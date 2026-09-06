# Khoroos

![Khoroos, a toolkit for poultry behavior analysis](assets/khoroos_banner.png)

Khoroos is a research toolkit for turning poultry-house video into individual bird
tracks, behaviour timelines, and descriptive behavior statistics.

It combines bird detection, multi-object tracking, stable clip extraction, action
classification, and behavior statistics in one reproducible pipeline. You can use it through a
web interface, from the command line, or as a Python library.

Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).

## What the pipeline produces

1. Birds are detected in sampled video frames.
2. Detections are linked into tracks with stable identifiers.
3. Each track is divided into short, stable single-bird clips.
4. Clips are classified against the 15-behaviour ChickenAct ethogram.
5. Results are summarized as time budgets, behavior statistics and exportable tables.

## Choose an interface

=== "Web interface"

    ```bash
    khoroos ui
    ```

    Upload a video, configure the run, follow its progress, and inspect the result in a browser.

=== "Command line"

    ```bash
    khoroos analyze farm.mp4 --output results/
    ```

    Run repeatable batch analyses and write a complete export bundle.

=== "Python"

    ```python
    from khoroos import analyze_video

    result = analyze_video("farm.mp4", preset="balanced")
    print(result.metrics["time_budget"])
    ```

    Integrate the analysis pipeline into notebooks, scripts, or other applications.

!!! note "Measurement scope"
    Statistics summarize model-assigned labels on sampled track windows. Results include
    observation duration, confidence, and processing notes. See
    [Understanding the statistics](guide/interpreting-results.md) for definitions.
