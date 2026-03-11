# Getting Started

This guide covers day-to-day usage with Docker Compose:

- setting up and running the ETL
- choosing the right container image
- troubleshooting container builds
- adding new Python packages to images

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Git submodules initialized:

```bash
git submodule update --init --recursive
```

## Choose a Docker service

| Service | Dockerfile | Base | Approx. size | Best for |
|---|---|---|---|---|
| `etl-dev` | `docker/Dockerfile.dev` | `python:3.11-slim` | ~300 MB | Fast ETL iteration (no PhysicsNeMo/PyTorch) |
| `etl` | `docker/Dockerfile.physicsnemo-cpu` | `python:3.11-slim` | ~1 GB | Full CPU stack from PyPI |
| `etl-ngc` | `docker/Dockerfile.ngc` | `nvcr.io/nvidia/physicsnemo/physicsnemo:25.11` | ~13 GB | NVIDIA pre-tested stack |

All services run on Apple Silicon (`arm64`) and Intel (`amd64`) without a GPU.

## Build and run

### Option A: direct run from host terminal

```bash
docker compose build etl-dev
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py --config-name lid_driven'
```

Replace `etl-dev` with `etl` or `etl-ngc` if needed.

The `lid_driven` config is defined in `src/moose_etl/config/lid_driven.yaml`:

```yaml
defaults:
  - moose_etl
  - _self_

etl:
  processing:
    num_processes: 4
  source:
    input_dir: ../data/lid-driven
    data_dir: ../data/lid-driven
  sink:
    output_dir: ../data/processed/lid-driven
```

### Option A2: same run with CLI overrides (no dedicated yaml)

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py \
  etl.source.input_dir=../data/lid-driven \
  etl.source.data_dir=../data/lid-driven \
  etl.sink.output_dir=../data/processed/lid-driven \
  etl.processing.num_processes=4'
```

### Create your own config

Use `lid_driven.yaml` as a template for a new dataset:

```bash
cp src/moose_etl/config/lid_driven.yaml src/moose_etl/config/my_case.yaml
```

Edit these keys in `src/moose_etl/config/my_case.yaml`:

- `etl.source.input_dir`
- `etl.source.data_dir`
- `etl.sink.output_dir`
- (optional) `etl.processing.num_processes`

Run with your new config name:

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py --config-name my_case'
```

### Option B: interactive shell

```bash
docker compose run --rm etl-dev
```

Then inside the container:

```bash
cd src
python run_etl.py --config-name lid_driven
```

## Input and output conventions

| Pattern | Description |
|---|---|
| `{sim_name}.e` | Exodus II mesh + element fields |
| `{sim_prefix}_out_{probe_name}_{timestep:04d}.csv` | CSV line probes |

- Output directory (with `--config-name lid_driven`): `data/processed/lid-driven/`
- Output format: one `{sim_name}.zarr` per simulation
- Exodus and CSV prefixes do not need to match

## FNO training and evaluation

Use the `etl` or `etl-ngc` service for PhysicsNeMo + PyTorch scripts.
Edit these templates first:

- `src/config/train_fno.yaml`
- `src/config/eval_fno.yaml`

### Train

```bash
docker compose run --rm etl bash -lc 'cd src && python train_fno.py --config config/train_fno.yaml'
```

### Evaluate

```bash
docker compose run --rm etl bash -lc 'cd src && python eval_fno.py --config config/eval_fno.yaml'
```

CLI flags override YAML values:

```bash
docker compose run --rm etl bash -lc 'cd src && python train_fno.py --config config/train_fno.yaml --epochs 50'
```

## Logs

During build:

```bash
docker compose build --progress=plain etl-ngc
```

During runtime:

```bash
docker compose logs -f etl-ngc
docker compose logs --tail=100 etl-ngc
```

`docker compose run --rm ...` removes the container when it exits, including its
stored logs. Omit `--rm` if you need to inspect logs after a run.

## Troubleshooting builds

If you see TLS errors such as `CERTIFICATE_VERIFY_FAILED` or `UnknownIssuer`,
your environment may require a corporate CA certificate.

### Add CA file (`etl-dev`, `etl`, `etl-ngc`)

Place a CA cert in `docker/certs/` (`.pem`, `.crt`, `.cer`) and rebuild:

```bash
docker compose build --no-cache etl-dev  # or etl / etl-ngc
```

`etl-dev` validates custom certs and skips files that are malformed or leaf
certificates (`CA:FALSE`) instead of CA certificates (`CA:TRUE`).

You can also point to a different host cert directory at build time:

```bash
CA_CERT_DIR=/path/to/certs docker compose build --no-cache etl-dev
```

### Pass CA via environment variable (`etl`, `etl-ngc`)

```bash
EXTRA_CA_CERT_B64="$(base64 < /path/to/your-org-ca.crt | tr -d '\n')" \
docker compose build etl                  # or etl-ngc
```

### Bypass TLS as last resort

For `etl-dev`:

```bash
PIP_TRUSTED_HOST_FLAGS="--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org" \
docker compose build etl-dev
```

For `etl`:

```bash
UV_ALLOW_INSECURE_HOST_FLAGS="--allow-insecure-host pypi.org --allow-insecure-host files.pythonhosted.org" \
docker compose build etl
```

For `etl-ngc`:

```bash
PIP_TRUSTED_HOST_FLAGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org" \
docker compose build etl-ngc
```

### Corporate proxy

```bash
HTTP_PROXY=http://proxy.example.com:8080 \
HTTPS_PROXY=http://proxy.example.com:8080 \
NO_PROXY=localhost,127.0.0.1 \
docker compose build etl                  # or etl-ngc
```

### Apple Silicon: force `amd64`

```bash
DOCKER_PLATFORM=linux/amd64 docker compose build etl       # or etl-ngc
DOCKER_PLATFORM=linux/amd64 docker compose run --rm etl    # or etl-ngc
```

### Skip full PhysicsNeMo install (`etl` only)

```bash
INSTALL_PHYSICSNEMO=0 docker compose build etl
```

## Add new Python packages to Docker images

Do not rely on `pip install` in a running container for persistent changes. Add
packages to Dockerfiles, then rebuild.

1. Select service(s) that need the dependency.
2. Edit the matching Dockerfile:

| Service | Dockerfile | Install command style |
|---|---|---|
| `etl-dev` | `docker/Dockerfile.dev` | `pip install ...` |
| `etl` | `docker/Dockerfile.physicsnemo-cpu` | `uv ... pip install --system ...` |
| `etl-ngc` | `docker/Dockerfile.ngc` | `pip install ...` |

3. Rebuild and rerun:

```bash
docker compose build etl-dev
docker compose run --rm etl-dev
```

4. Verify inside container:

```bash
python -c "import your_package; print(your_package.__version__)"
```

## Next references

- [ETL pipeline internals](../dev/etl_pipeline.md)
- [Dataset API](../dev/dataset.md)
- [FNO training and evaluation details](../dev/fno_train_eval.md)
