import dataclasses
import hashlib
import json
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _short_hash(obj) -> str:
    d = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def build_experiment_dirs(cfg) -> tuple[Path, Path, bool]:
    """
    Returns (output_dir, log_dir, already_exists) with the structure:
      <root>/outputs/<experiment_name>/<data_hash>/<model_hash>/<config_hash>/
      <root>/logs/<experiment_name>/<data_hash>/<model_hash>/<config_hash>/

    `already_exists` is True only if the run fully completed (results.json present).
    config.json is always written at the start so partial/failed runs are traceable.
    """
    data_hash   = _short_hash(cfg.data)
    model_hash  = _short_hash(cfg.model)
    config_hash = _short_hash({
        "train":    dataclasses.asdict(cfg.train),
        "prune":    dataclasses.asdict(cfg.prune),
        "finetune": dataclasses.asdict(cfg.finetune),
        "seed":     cfg.seed,
    })

    stem = Path(cfg.experiment_name) / data_hash / model_hash / config_hash

    output_dir = _PROJECT_ROOT / "outputs" / stem
    log_dir    = _PROJECT_ROOT / "logs"    / stem

    already_exists = (output_dir / "results.json").exists()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    return output_dir, log_dir, already_exists


def setup_logging(log_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # Console handler: add once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # File handler: swap per experiment so each run gets its own log
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)
    fh = logging.FileHandler(log_dir / "run.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
