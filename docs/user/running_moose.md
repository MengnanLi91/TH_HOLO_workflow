# Running MOOSE Simulations

The multifid-th repository consumes MOOSE outputs but does **not** ship a
MOOSE runtime. The ETL pipeline (`src/cases/moose_grid/`,
`src/cases/alpha_d/etl/`) expects Exodus `.e` files plus CSV probes that
already exist on disk; producing them is a separate, out-of-band step
that needs its own toolchain (OpenMPI, libMesh, PETSc, the MOOSE
framework itself).

The **canonical multifid-th containers** (`multifid-th-cpu.sif`,
`multifid-th-gpu.sif`, etc., described in
[Getting Started](getting_started.md)) provide the Python / PhysicsNeMo
stack used for training and evaluation. They are deliberately MOOSE-free
to keep image size manageable and the two stacks decoupled.

For authoritative MOOSE installation guidance always defer to the
official documentation at
<https://mooseframework.inl.gov/getting_started/installation/index.html>.
The rest of this page documents only the container path that the
multifid-th project tests against on HPC.

## INL Apptainer container (preferred on INL HPC)

MOOSE's INL build pipeline publishes pre-built Apptainer images to
`mooseharbor.hpc.inl.gov`. This is the fastest way to get a working
MOOSE on a Linux HPC node and is what the multifid-th ETL pipeline is
exercised against.

### Obtain the SIF

The project's reference recipe is
`/data/lim2/containers/moose-dev-openmpi-x86_64.def` (committed locally,
not in this repo). It bootstraps from the INL ORAS registry:

```text
BootStrap: oras
From: mooseharbor.hpc.inl.gov/moose-dev/moose-dev-openmpi-x86_64:next-2026.01.27
```

Build it once:

```bash
apptainer build moose-dev-openmpi-x86_64.sif \
    /data/lim2/containers/moose-dev-openmpi-x86_64.def
```

The first build pulls the base image from `mooseharbor.hpc.inl.gov`,
which requires INL-NCRC credentials (`MOOSE-NCRC` SIF signing key,
fingerprint `0CFFCAB55E806363601C442D211817B01E0911DB`). Non-INL users
should pick one of the other install paths below.

The recipe also installs `pyfmi` from conda-forge and shrinks the
image via `conda clean -a -y`. A `pyfmi`-augmented variant is also
available locally as
`/data/lim2/containers/moose-dev-openmpi-x86_64_pyfmi.sif`.

### Run a MOOSE input file

```bash
# Bind the project so the container can see inputs / outputs:
apptainer exec \
    --bind /path/to/project:/path/to/project \
    /data/lim2/containers/moose-dev-openmpi-x86_64.sif \
    moose-opt -i path/to/input.i
```

`moose-opt` is the standard launcher for a built MOOSE application
inside the container; for parallel runs prefix with `mpiexec`:

```bash
apptainer exec --bind /path/to/project:/path/to/project \
    /data/lim2/containers/moose-dev-openmpi-x86_64.sif \
    mpiexec -n 8 moose-opt -i path/to/input.i
```

If you are running a downstream app rather than core MOOSE (e.g. a
custom executable built against MOOSE), invoke that binary instead —
the container provides the libMesh / PETSc / OpenMPI runtime it links
against. See MOOSE's
[HPC install guide](https://mooseframework.inl.gov/getting_started/installation/hpc_install_moose.html)
for app-build patterns inside containers.

### Verify the container

```bash
apptainer exec /data/lim2/containers/moose-dev-openmpi-x86_64.sif \
    bash -lc 'which moose-opt && moose-opt --version'
```

## Alternative install paths

The multifid-th pipeline does not care *how* you produced the Exodus
files, only that they exist on disk in the layout the ETL configs
expect. Any of MOOSE's officially supported install methods is fine:

| Path | When to use | Official guide |
|---|---|---|
| **Conda (Linux/macOS)** | Local development on a workstation; no admin needed | <https://mooseframework.inl.gov/getting_started/installation/conda.html> |
| **Conda pre-built MOOSE binary** | Training / quick onboarding; no compilation | <https://mooseframework.inl.gov/getting_started/installation/moose_conda_binary.html> |
| **Docker** | Reproducible local builds; cross-platform | <https://mooseframework.inl.gov/getting_started/installation/docker.html> |
| **WSL** | Windows host | <https://mooseframework.inl.gov/getting_started/installation/windows.html> |
| **Build from source (GCC)** | Custom compiler flags, downstream-app development | <https://mooseframework.inl.gov/getting_started/installation/manual_installation_gcc.html> |
| **Build from source (LLVM)** | LLVM/Clang toolchains | <https://mooseframework.inl.gov/getting_started/installation/manual_installation_llvm.html> |
| **Offline / air-gapped** | No internet on target machine | <https://mooseframework.inl.gov/getting_started/installation/offline_installation.html> |

For a quick local conda install on Linux, the official commands are:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge
export PATH=$HOME/miniforge/bin:$PATH
conda init --all
# Restart shell, then:
conda update --all --yes
conda config --add channels https://conda.software.inl.gov/public
conda create -n moose moose-dev=2026.05.08=mpich
conda activate moose
```

MOOSE's stated minimum requirements: a POSIX-compliant Unix-like OS
(Linux or macOS), 8 GB RAM (16 GB for debug builds), 30 GB disk, a
supported GCC (9.0.0–13.3.1) or LLVM/Clang (14.0.6–19), and
Python 3.10–3.13. ARM (Apple Silicon) is supported for the conda path.

## Handing MOOSE outputs to the multifid-th ETL

After your MOOSE run produces `.e` files (plus optional CSV probes),
exit the moose-dev container and switch to the multifid-th ETL flow,
which lives in a separate container:

```bash
# Run the ETL inside the PhysicsNeMo-flavoured multifid-th container:
apptainer exec --bind /path/to/project:/path/to/project \
    multifid-th-cpu.sif \
    bash -lc 'cd /path/to/project/src && python cases/moose_grid/run_etl.py'
```

The ETL config (`src/cases/moose_grid/configs/etl.yaml` for the
lid-driven flow, or `src/cases/alpha_d/configs/etl.yaml` for the alpha-D
case) controls where the ETL reads the `.e` / CSV files from and where
it writes the Zarr stores the training pipeline consumes. See
[ETL pipeline](../dev/etl_pipeline.md) for the data-source / transform /
sink layout and [Getting Started](getting_started.md) for the
multifid-th container options.

The two containers are deliberately kept separate because MOOSE wants
its own conda toolchain (libMesh + PETSc + custom-compiled OpenMPI)
while PhysicsNeMo wants a clean PyTorch wheel-based environment.
Merging them would either bloat the image or break one of the stacks.
