# Valdi: Value Diffusion World Models

Training experiments managed with [uv](https://docs.astral.sh/uv/) and YAML configs.

## Setup

Clone the repo and create the environment from the lockfile. With `uv.lock` and `pyproject.toml` present, this installs the exact pinned dependencies:

```bash
git clone git@github.com:Kit115/ValueDiffusionWorldModels.git
cd ValueDiffusionWorldModels
uv sync
```

## Running an experiment

Pick one of the supplied configs (or copy one as a starting point and edit it), then launch training:

```bash
# Use a supplied config directly
uv run train.py configs/default_diffusion.yaml

# Or base your own on an existing one
cp configs/default_diffusion.yaml configs/my_experiment.yaml
# ...edit configs/my_experiment.yaml...
uv run train.py configs/my_experiment.yaml
```

## Resuming a run

Each run is written to `experiments/<config_name>/<run_index>/`. To resume from the latest checkpoint, point `train.py` at that run directory and pass `--resume`:

```bash
uv run train.py experiments/my_experiment/0/ --resume
```

