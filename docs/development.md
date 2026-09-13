# Development

## Set up the environment

Clone the repository and install the test, web, and documentation extras:

```bash
git clone https://github.com/amirivojdan/khoroos.git
cd khoroos
uv sync --locked --extra dev --extra web --extra docs
uv run pytest
uv run ruff check src tests
```

Tests use stub models and generated videos; they do not need downloaded weights.

GitHub Actions runs `.github/workflows/test.yml` on pushes and pull requests, testing
Python 3.12 and 3.14 on Ubuntu with the locked dependencies. Inference uses CPU, and Hub
access is disabled during tests. The Python 3.12 job also runs Ruff and the strict docs build.
You can start the workflow manually from **Actions → Tests → Run workflow**.

See [Publish to PyPI](publishing.md) for Trusted Publisher setup and the release process.

## Design and maintainability criteria

Review changes against these concrete questions:

| Criterion | What to check in Khoroos |
| --- | --- |
| Cohesion | Does a function have one reason to change? Keep user interaction, scheduling, model inference, and serialization separate. |
| Dependency direction | CLI and web call pipeline APIs; models and tracking do not depend on HTTP or browser state. Avoid reading another layer's private fields. |
| Explicit contracts | State tensor shapes, units, label order, timestamp semantics, and ownership. Validate external data at the boundary. |
| Resource lifetime | Identify who opens and closes readers, models, threads, and files, including exceptions and cancellation. |
| Local reasoning | Keep related values in one record instead of synchronized lists; use small helpers with explicit inputs instead of shared mutable context. |
| Compatibility | Preserve public extension signatures and result schemas. Treat changes to measurement semantics as behavior changes, even when JSON keys stay the same. |
| Testability | Test observable behavior and failure paths with injected components; reserve real-model checks for preprocessing parity and scientific validation. |
| Bounded work | Keep image tensors within batch scope. Separate output-size growth from temporary working memory, and measure before optimizing. |

Use the existing Ruff configuration, test suite, and strict documentation build as the
baseline checks. Passing them is necessary but does not establish model accuracy or prove
the absence of concurrency bugs. Avoid renaming stable APIs or adding frameworks solely to
make the architecture look more uniform.

### Ownership and orchestration

`VideoAnalyzer` schedules stages and progress. `pipeline.classification.classify_batch`
prepares and classifies one batch, with each crop's source window and export index held
together. It returns predictions, allowing image tensors to leave scope before the next
batch. Raw exports are completed before the model call, preserving them if inference fails.

Readers and trackers belong to one run; models can be reused across sequential runs.
`VideoAnalyzer` and `AnalysisRunner` support context managers for explicit cleanup, including
on exceptions. Entering a context does not load weights. Exiting calls `close()`, including
the close methods of injected models; callers sharing models should manage that lifetime
explicitly and avoid overlapping contexts. These objects are intended for sequential use.

The FastAPI lifespan constructs `JobManager` at startup. Merely calling `create_app()` does
not start a worker or create job directories. Tests and embedders must run the ASGI lifespan
before accessing `app.state.jobs`; `with TestClient(app)` does this. Shutdown requests
cancellation and waits for the worker. A model call may outlast the shutdown timeout, so
the worker itself releases managed models after it stops using them. Injected runners are
borrowed by the manager and remain the caller's responsibility.

Use monotonic clocks for elapsed durations and throughput. Wall-clock timestamps remain
appropriate for recorded job creation/completion times. Copy nested configuration values
when they become run records, so later settings edits cannot rewrite historical metadata.

### Review findings and remaining work

The September 2026 review found useful module boundaries, validated extension contracts,
lazy model loading, and a substantial stub-based test suite. It also identified the following
remaining design debt. These are open findings, not guarantees supplied by a passing suite.

| Priority | Finding | Next bounded improvement |
| --- | --- | --- |
| High | `build_tracklets` measures coverage after interpolation; sparse evidence can look fully observed. | Define observation support separately from interpolated geometry, with stride-aware fixtures and labeled-footage validation. |
| High | JS overlay lookup, Python overlay lookup, and metric interval assignment use different overlap/uncertainty rules. | Extract a shared temporal-support contract and test both language implementations with one fixture. |
| High | Upload names use `id(file)`; source deletion infers ownership from location. | Use exclusive unique upload creation and explicit source ownership; cover interrupted uploads and deletion failures. |
| High | Optional overlay/export failures and cancellation do not have a complete artifact lifecycle. | Record artifact availability separately and publish finished files atomically. |
| Medium | Job state and SSE delivery span threads, unbounded subscriber queues, and one-shot browser recovery. | Publish consistent status snapshots and bound subscriber buffering; test reconnect and terminal-event delivery. |
| Medium | The browser script combines setup, network state, rendering, interpolation, and interaction; asset-string tests do not execute these behaviors. | Extract pure lookup functions as needed and add browser tests for setup, progress, keyboard access, and review. |
| Medium | Metrics repeatedly partition the same predictions, while crop extraction repeats source-frame requests. | Share the interval partition within a metrics call; benchmark a bounded batch-level decode-reuse helper. |
| Medium | Full results and dense geometry are retained for long recordings; job history disappears after restart. | Add durable per-job manifests and compact review queries with explicit memory limits. |
| Medium | Source time assumes constant FPS, and serialized predictions retain only rounded top-k values. | Specify timestamp and archival contracts before introducing variable-rate support or result replay. |
| Medium | Type checking is not enforced; locked CI does not cover fresh dependency resolution. | Introduce a gradual core-contract type baseline and a fresh-wheel decode/import smoke check. |
| Low | `analyze_video()` has no explicit model cleanup, and its injected-model ownership is unspecified. | Define borrowed versus owned model lifetime before adding automatic cleanup; use an explicitly managed analyzer for repeated calls. |

Prioritize correctness and ownership ahead of broad file splits. In particular, a full
frontend rewrite, a new workflow engine, or removing documented convenience methods is not
a prerequisite for fixing these findings.

## Documentation

Start the live-reloading documentation server:

```bash
uv run mkdocs serve
```

The site is available at `http://127.0.0.1:8000`. Changes below `docs/` trigger an automatic
rebuild.

## Validate the site

```bash
uv run mkdocs build --strict
```

Strict mode treats broken internal links, missing navigation entries, and documentation-plugin
warnings as build failures. The generated site is written to `site/`, which is ignored by Git.

## Container validation

```bash
docker compose config --quiet
docker build -t khoroos:local .
docker run --rm -e KHOROOS_DEVICE=cpu khoroos:local khoroos info
docker run --rm khoroos:local python -c "import torchcodec; print(torchcodec.__version__)"
```

The image installs the locked dependencies and a non-editable Khoroos package. Source changes
reuse the dependency layer. The runtime image includes the installed web assets and excludes
notebooks, training checkpoints, and development tools.

## Documentation layout

```text
mkdocs.yml                 # Site configuration and navigation
docs/
├── index.md               # Landing page
├── getting-started.md
├── guide/                 # Task-oriented user documentation
└── reference/             # API pages generated from docstrings
```

## Writing API documentation

API pages use mkdocstrings. Add or improve a Python docstring in `src/khoroos/`, then reference
the object from a Markdown file:

```text
::: khoroos.pipeline.types.AnalysisResult
```

Keep task instructions in the user guide and reserve the reference section for precise API
contracts.
