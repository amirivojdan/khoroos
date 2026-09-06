# Development

## Set up the environment

From the repository root, install the test, web, and documentation extras:

```bash
uv sync --locked --extra dev --extra web --extra docs
uv run pytest
uv run ruff check src tests
```

Tests use stub models and generated videos; they do not need downloaded weights.

See [Publish to PyPI](publishing.md) for Trusted Publisher setup and the release process.

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
