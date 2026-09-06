# Docker

The image runs Ubuntu 24.04, Python 3.12, and the PyTorch dependencies pinned in `uv.lock`.
Models are downloaded at runtime, not included in the image.

## Requirements

The Compose service requires:

- Docker Engine and [Compose 2.30.0 or newer](https://docs.docker.com/reference/compose-file/services/#gpus).
- An NVIDIA GPU supported by the locked PyTorch build.
- An NVIDIA driver compatible with CUDA 13 (R580 or newer; see
  [NVIDIA's compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/)).
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configured for Docker.

The default [detector](https://huggingface.co/amirivojdan/chicken_rtdetrv2) and
[action classifier](https://huggingface.co/amirivojdan/chicken_vjepa2_action) are public.
No Hugging Face token is required.

## Start the web interface

From the repository root:

```bash
docker compose up --build -d
docker compose logs -f khoroos
```

The first start downloads about 1.7 GB of model files. Open `http://localhost:8000` after the
server starts. `docker compose ps` shows the health status. The health check verifies that the
web interface responds; model inference starts with the first analysis.

The published port is bound to localhost. To change the port:

```bash
KHOROOS_PORT=8080 docker compose up -d
```

For access from a trusted network, also set `KHOROOS_BIND_HOST=0.0.0.0`. The web interface has
no authentication; use an authenticated reverse proxy for remote access.

## Storage and restarts

Compose mounts its `<project>_khoroos-data` volume at `/var/lib/khoroos`:

| Directory | Contents |
| --- | --- |
| `weights/` | Hugging Face snapshots, separated by model repository |
| `huggingface/` | Hugging Face client data |
| `jobs/` | Uploaded videos and analysis files |

Files and downloaded models survive container recreation and `docker compose down`.
`docker compose down -v` deletes the volume and its contents.

The job list and queue are held in memory. Restarting clears the browser's job list and
interrupts active analyses; it does not resume them. Existing files remain in the volume,
but are not reloaded into the UI or covered by its job-expiry cleanup after a restart.
Export needed results before stopping the service.

During a session, terminal jobs expire 24 hours after completion by default. Cleanup runs
between analyses and once a minute while idle.

After changing configuration or code, use `docker compose up -d --build`.
`docker compose restart` only restarts the existing container.

## Analyze a file

With the web service running, a one-off container can share its model cache and save a batch
result in the same volume:

```bash
docker compose run --rm --no-deps \
  -v "$PWD:/input:ro" \
  khoroos khoroos analyze /input/farm.mp4 -o /var/lib/khoroos/batch/farm

docker compose cp khoroos:/var/lib/khoroos/batch/farm ./results
```

Avoid running batch inference alongside an active web analysis on the same GPU.
The container runs as UID/GID `10001:10001`. Bind-mounted inputs must be readable by that user;
if you mount an output or cache directory yourself, it must also be writable by that user.
The example above uses the managed volume for writes.

## Authentication and startup options

Authentication is optional for the default public models. For private replacement models,
pass `HF_TOKEN` at runtime, or mount a token file readable by the container user and set
`HF_TOKEN_FILE`:

```bash
docker compose run --rm --no-deps \
  -v /absolute/path/hf_token:/run/secrets/hf_token:ro \
  -e HF_TOKEN_FILE=/run/secrets/hf_token \
  khoroos khoroos models download
```

The file takes precedence over `HF_TOKEN`. Never add a token to the Dockerfile.

`KHOROOS_PREFETCH_MODELS` controls downloads before the requested command:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Prefetch for `khoroos ui` and `khoroos analyze` |
| `1` | Prefetch before any command |
| `0` | Skip startup prefetch |

Commands such as `khoroos info`, `khoroos models status`, and `ffmpeg -version` do not prefetch
in `auto` mode. Disabling prefetch does not disable downloads requested by inference itself.
Complete cached snapshots can be used offline without a token.

## Checks and troubleshooting

```bash
# Check GPU access inside the running service.
docker compose exec khoroos python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

# Check the image without a GPU or model download.
docker run --rm khoroos:local ffmpeg -version
docker run --rm -e KHOROOS_DEVICE=cpu khoroos:local khoroos info
docker run --rm khoroos:local python -c "import torchcodec; print(torchcodec.__version__)"
```

- **GPU device-driver error:** check the host driver and NVIDIA Container Toolkit setup.
- **401 or repository not found:** check the configured repository IDs. If using private
  replacements, confirm the token has read access; remove stale credentials for public access.
- **Permission denied:** check bind-mount permissions for UID/GID 10001.
- **Out of GPU memory:** reduce the action batch size and run one analysis at a time.
- **Unhealthy during startup:** inspect `docker compose logs khoroos`; downloads may still be
  running. The health check allows ten minutes for startup. Docker does not restart a container
  solely because it is unhealthy.
