# Documentation development

## Set up the environment

From the repository root, install the documentation dependency group:

```bash
uv sync --extra docs
```

Start the live-reloading development server:

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

## Project layout

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
