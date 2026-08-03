# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04

FROM ghcr.io/astral-sh/uv:0.9.11 AS uv

FROM ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libpython3.12 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/opt/khoroos/.venv

WORKDIR /opt/khoroos

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra web

FROM base AS runtime

RUN groupadd --gid 10001 khoroos \
    && useradd --uid 10001 --gid khoroos --create-home --shell /usr/sbin/nologin khoroos \
    && install --directory --owner=khoroos --group=khoroos /var/lib/khoroos

WORKDIR /opt/khoroos

COPY --from=builder --chown=khoroos:khoroos /opt/khoroos /opt/khoroos
COPY --chown=khoroos:khoroos --chmod=755 docker/entrypoint.sh /usr/local/bin/khoroos-entrypoint

ENV PATH=/opt/khoroos/.venv/bin:$PATH \
    HF_HOME=/var/lib/khoroos/huggingface \
    KHOROOS_CACHE_DIR=/var/lib/khoroos \
    KHOROOS_HOST=0.0.0.0 \
    KHOROOS_PORT=8000 \
    KHOROOS_PREFETCH_MODELS=1

VOLUME ["/var/lib/khoroos"]
EXPOSE 8000

USER khoroos

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3).close()"]

ENTRYPOINT ["khoroos-entrypoint"]
CMD ["khoroos", "ui", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
