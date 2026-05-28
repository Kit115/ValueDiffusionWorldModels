import csv
from pathlib import Path
from dataclasses import dataclass, fields, asdict

import yaml


@dataclass
class TrainConfig:
    num_epochs:          int = 5000
    updates_per_epoch:   int = 60
    seed_trajectories:   int = 10
    batch_size:          int = 256
    checkpoint_interval: int = 100
    log_window_size:     int = 100
    buffer_capacity:     int = 1000
    trajectory_length:   int = 600

    # exploration-noise schedule
    max_std:     float = 0.25
    min_std:     float = 0.05
    hold_steps:  int   = 250
    decay_steps: int   = 250

    def exploration_std(self, step):
        s = max(0, int(step))
        if s < self.hold_steps:
            return self.max_std
        if self.decay_steps == 0:
            return self.min_std
        t = s - self.hold_steps
        if t < self.decay_steps:
            return self.max_std + (self.min_std - self.max_std) * (t / self.decay_steps)
        return self.min_std


def _build(cls, d):
    known   = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_config(path, algo_cls):
    """Parse a YAML file into (TrainConfig, algo_cls instance)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extra = set(raw) - {"train", "algorithm"}
    if extra:
        raise ValueError(f"Unexpected top-level config sections: {sorted(extra)}")
    return _build(TrainConfig, raw.get("train", {})), _build(algo_cls, raw.get("algorithm", {}))


def save_config(path, train_cfg, algo_cfg):
    """Write the frozen, normalised config actually used by a run."""
    with open(path, "w") as f:
        yaml.safe_dump(
            {"train": asdict(train_cfg), "algorithm": asdict(algo_cfg)},
            f, sort_keys=False,
        )


def setup_run_dir(config_path, base="experiments"):
    """Create experiments/<config-stem>/run_<n+1>/ and return its Path."""
    exp_dir = Path(base) / Path(config_path).stem
    exp_dir.mkdir(parents=True, exist_ok=True)
    used = [int(p.name[4:]) for p in exp_dir.glob("run_*") if p.name[4:].isdigit()]
    run_dir = exp_dir / f"run_{max(used, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def write_log(path, returns, window):
    """Rewrite log.csv from scratch so it always matches the latest checkpoint."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_trajectories", "avg_return", "last_return"])
        for i, ret in enumerate(returns):
            lo  = max(0, i + 1 - window)
            avg = sum(returns[lo:i + 1]) / (i + 1 - lo)
            w.writerow([i + 1, f"{avg:.4f}", f"{ret:.4f}"])
