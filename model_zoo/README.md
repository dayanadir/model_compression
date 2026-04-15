## Model Zoo pipeline

Run from repository root:

```bash
python3 -m pip install --user -r requirements-model-zoo.txt
PYTHONPATH=./graph_metanetworks-main python3 -m model_zoo --config configs/cifar10_smoke_test.yaml
```

To run in parallel across specific GPUs, set `run.gpu_ids` in the YAML config.
For example, `gpu_ids: [1, 2, 3, 4]` launches 4 shard processes in parallel
and each process trains a disjoint subset of model IDs.

### Run safely with tmux

Use the helper script from repo root:

```bash
./scripts/run_model_zoo_tmux.sh start
```

Useful commands:

```bash
./scripts/run_model_zoo_tmux.sh status   # check if still running
./scripts/run_model_zoo_tmux.sh logs     # follow logs (tail -f)
./scripts/run_model_zoo_tmux.sh attach   # attach to live tmux output
./scripts/run_model_zoo_tmux.sh stop     # stop the session
```

Optional overrides:

```bash
SESSION_NAME=zoo_run_apr14 CONFIG_PATH=configs/cifar10_default.yaml ./scripts/run_model_zoo_tmux.sh start
```

Optional: copy `.env.model_zoo.example` to `.env.model_zoo` and source it before running:

```bash
cp .env.model_zoo.example .env.model_zoo
set -a && source .env.model_zoo && set +a
```

### W&B large sweep (all model indices)

1) Generate a sweep YAML that enumerates all model indices from your config:

```bash
PYTHONPATH=./graph_metanetworks-main ./scripts/generate_wandb_sweep_yaml.py \
  --config configs/cifar10_default.yaml \
  --output sweeps/cifar10_default_wandb_sweep.yaml \
  --entity <optional-team-or-user>
```

If omitted, W&B project defaults to `model_compression`.

2) Launch sweep agents in tmux across GPUs 1,2,3,4:

```bash
GPU_IDS=1,2,3,4 SWEEP_YAML=sweeps/cifar10_default_wandb_sweep.yaml \
  ./scripts/run_wandb_sweep_tmux.sh start
```

3) Monitor status:

```bash
./scripts/run_wandb_sweep_tmux.sh status   # windows/agents health
./scripts/run_wandb_sweep_tmux.sh attach   # live tmux output
LOG_GPU=1 ./scripts/run_wandb_sweep_tmux.sh logs
```

Notes:
- The sweep is a grid over `model_index` so each run trains one deterministic model.
- By default this logs metadata artifacts; add `--wandb-log-artifact` in the sweep
  command if you also want full `weights.pt` uploaded (much larger storage usage).
