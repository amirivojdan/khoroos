# Publish to PyPI

The workflow is `.github/workflows/publish.yml`. GitHub only discovers workflows in
`.github/workflows/`; placing `publish.yml` directly in `.github/` will not run it.

A published GitHub release builds and uploads the wheel and source distribution.
A manual workflow run builds and checks them without publishing. No PyPI API token is needed.

## 1. Create the GitHub environment

In [repository settings](https://github.com/amirivojdan/khoroos/settings/environments), create
an environment named **`pypi`**. This name must match the workflow and PyPI configuration.
You can add a required reviewer if you want a manual approval before each upload.

## 2. Register the Trusted Publisher

Sign in to PyPI and open [account publishing settings](https://pypi.org/manage/account/publishing/).
Under **Add a new pending publisher**, choose GitHub and enter:

| Field | Value |
| --- | --- |
| PyPI project name | `khoroos` |
| Owner | `amirivojdan` |
| Repository name | `khoroos` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The workflow name is the filename, not its path or display title. A pending publisher creates
the project on the first successful upload; it does not reserve the name beforehand.
See [PyPI's pending-publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

If you already own the PyPI project by the time you do this, add the publisher from that
project's **Publishing** settings instead, using the same GitHub values.

## 3. Check the release build

Commit and push the reviewed package changes and workflow to `main`.
On GitHub, open **Actions → Publish to PyPI → Run workflow**, select `main`, and run it.

The build checks package metadata, README rendering, the license, and required web assets.
Download the `distributions` artifact to inspect the wheel and source archive.
The publish job is skipped for this manual run. Run the test suite separately before releasing:

```bash
uv sync --locked --extra dev --extra web
uv run pytest
uv run ruff check src tests
```

## 4. Publish a GitHub release

The current package version is `0.1.0` in `pyproject.toml`.
In [GitHub Releases](https://github.com/amirivojdan/khoroos/releases/new):

1. Create tag **`v0.1.0`** at the commit containing the reviewed changes.
2. Add release notes and click **Publish release**.
3. Watch **Actions → Publish to PyPI**. Approve the `pypi` deployment if you configured a reviewer.

The workflow rejects a tag that differs from `v` plus the package version. It builds with
read-only repository access, then uploads those same artifacts in a separate job with
`id-token: write`. PyPI uses that identity to authenticate the upload.
See [PyPI's Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## 5. Verify installation

In a fresh Python 3.12+ environment:

```bash
python -m pip install "khoroos[web]"
khoroos version
khoroos models download
```

The wheel contains application code and web assets. Model weights download separately;
FFmpeg is a system dependency.

For later releases, increment `version` in `pyproject.toml`, run `uv lock`, commit the updated
files, and publish a matching `vX.Y.Z` release. PyPI does not allow replacing uploaded files
with the same filenames; fixes to an already published package need a new version.

## Troubleshooting

- **Publisher mismatch:** check the owner, repository, workflow filename, and environment on PyPI.
- **Workflow does not run:** check that the file is committed under `.github/workflows/` and
  that you published the release rather than saving a draft or only pushing a tag.
- **Deployment waiting:** approve the GitHub environment review if one is configured.
- **Filename already used:** bump the package version and publish a new release.
