"""Determinism and fingerprinting, so runs that should be identical provably are.

Comparing WHICH neurons two pruning methods remove is only meaningful when both
saw the same data split and the same trained weights. Nothing in a pruning
config affects the stages before pruning, so in principle two runs with the same
seed already agree -- but "in principle" is not evidence, and three things break
it in practice:

  * cudnn autotuning (`benchmark = True`) picks different kernels run to run, and
    non-deterministic reductions on CUDA then perturb the trained weights;
  * model init draws from the GLOBAL torch RNG *after* the dataset builder has
    drawn from it, so changing anything about the data pipeline silently shifts
    the initialization;
  * DataLoader(shuffle=True) with no explicit generator inherits whatever RNG
    state training happens to start with.

`seed_everything` fixes the first, re-seeding immediately before model
construction fixes the second, and passing an explicit generator to the loaders
fixes the third. The digests here are the evidence: record them per run and two
runs are comparable exactly when their `data` and `trained` digests match.

Digests are of raw tensor bytes, so they are exact -- not tolerant. That is the
point: a digest mismatch means the comparison is invalid, and a tolerance would
hide precisely the drift we are trying to detect.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch


# ── seeding ──────────────────────────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG this project draws from, and optionally pin the backends.

    `deterministic=True` disables cudnn autotuning and asks for deterministic
    kernels. That costs some throughput and a few ops have no deterministic
    implementation -- we warn rather than raise, so a model that needs one still
    runs (its trained weights just stop being bit-reproducible, which the
    fingerprints will reveal).
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:                             # pragma: no cover
            logging.warning(f"could not enable deterministic algorithms: {exc}")
    else:
        torch.backends.cudnn.benchmark = True


def torch_generator(seed: int) -> torch.Generator:
    """A standalone generator, for DataLoader shuffling that must not depend on
    how much global RNG earlier stages consumed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── digests ──────────────────────────────────────────────────────────────────

_DIGEST_CHARS = 16


def _short(h: "hashlib._Hash") -> str:
    return h.hexdigest()[:_DIGEST_CHARS]


def tensor_digest(*tensors: torch.Tensor | np.ndarray | None) -> str:
    """Digest of raw tensor bytes, in the order given. Shape and dtype are
    folded in, so a reshape or a dtype change is a different digest."""
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"<none>")
            continue
        a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return _short(h)


def state_dict_digest(model: torch.nn.Module) -> str:
    """Digest of a model's parameters and buffers, keyed by name so that a
    reordering of the module tree is still detected."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_digest(t).encode())
    return _short(h)


def dataset_digest(bundle: Any) -> str:
    """Digest of the actual split contents -- inputs and labels of train, val
    and test. This is what makes "the same data portion" checkable: two runs
    agree iff this string agrees."""
    h = hashlib.sha256()
    for split in ("train_ds", "val_ds", "test_ds"):
        ds = getattr(bundle, split, None)
        h.update(split.encode())
        if ds is None:
            h.update(b"<none>")
            continue
        for t in getattr(ds, "tensors", ()):
            h.update(tensor_digest(t).encode())
    return _short(h)


def _plain(obj: Any) -> Any:
    """Config objects -> JSON-able, with dict keys sorted for stability."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name))
                for f in sorted(dataclasses.fields(obj), key=lambda f: f.name)}
    if isinstance(obj, dict):
        return {str(k): _plain(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def config_digest(*objs: Any) -> str:
    """Digest of configs. Use it to key a cache or to assert that two runs
    differ ONLY in the part you meant to vary (e.g. same model/data/training
    digest, different pruning digest)."""
    h = hashlib.sha256()
    h.update(json.dumps([_plain(o) for o in objs], sort_keys=True).encode())
    return _short(h)


# ── the record that makes runs comparable ────────────────────────────────────

def run_fingerprint(bundle: Any, model_init: torch.nn.Module | None,
                    model_trained: torch.nn.Module | None, seed: int,
                    config: Any = None) -> dict[str, Any]:
    """The block to store in a run's results.

    `data` and `init` must match for two runs to be comparing the same starting
    point; `trained` must match for a comparison of pruning decisions to be
    valid at all. `pre_prune` is the alias used by the comparison helpers.
    """
    fp: dict[str, Any] = {"seed": seed, "data": dataset_digest(bundle)}
    if model_init is not None:
        fp["init"] = state_dict_digest(model_init)
    if model_trained is not None:
        fp["trained"] = state_dict_digest(model_trained)
        fp["pre_prune"] = fp["trained"]
    if config is not None:
        fp["model_data_training"] = config_digest(
            getattr(config, "model", None), getattr(config, "data", None),
            getattr(config, "training", None), seed)
        fp["pruning"] = config_digest(getattr(config, "pruning", None))
    return fp


def check_comparable(fingerprints: dict[str, dict], keys: tuple[str, ...] = ("data", "trained")
                     ) -> tuple[bool, list[str]]:
    """(ok, problems) for a {label: fingerprint} mapping.

    Two runs are comparable when they started from the same data and the same
    trained weights. Anything else -- including a differing `pruning` digest --
    is expected and not reported.
    """
    problems: list[str] = []
    for key in keys:
        seen: dict[str, list[str]] = {}
        for label, fp in fingerprints.items():
            if key not in fp:
                problems.append(f"{label}: missing '{key}' digest")
                continue
            seen.setdefault(fp[key], []).append(label)
        if len(seen) > 1:
            groups = "; ".join(f"{d} <- {', '.join(sorted(ls))}"
                               for d, ls in sorted(seen.items()))
            problems.append(
                f"'{key}' digest differs, so these runs are NOT comparable: {groups}")
    return not problems, problems
