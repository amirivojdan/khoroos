# API reference

Khoroos exposes its supported high-level interface from the `khoroos` package:

```python
from khoroos import (
    AnalysisRunner,
    AnalysisParams,
    AnalysisResult,
    WelfareThresholds,
    analyze_video,
    params_for_preset,
    thresholds_from_overrides,
)
```

Use `analyze_video()` when you only need an in-memory result. Use `AnalysisRunner` when the run
should also write the standard exports or an annotated video.

The reference pages are generated from type annotations and source docstrings:

- [Configuration](config.md) documents settings, presets, parameters, and thresholds.
- [Analysis pipeline](analysis.md) documents the in-memory analyzer.
- [Runner](runner.md) documents artifact generation.
- [Result types](types.md) documents the versioned result model and progress events.
