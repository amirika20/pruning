"""Modular arithmetic dataset: all p^2 pairs of (a + b) mod p as token sequences."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.registry import DatasetBundle, register_dataset


@register_dataset("modular_add")
def modular_add(
    data_seed: int = 0,
    p: int = 97,  # prime modulus
    train_ratio: float = 0.5,
) -> DatasetBundle:
    """Input tokens: [a, op, b, =] where op=p and eq=p+1 (vocab size = p+2).
    Labels: (a + b) % p as long integers in [0, p)."""
    rng = torch.Generator().manual_seed(data_seed)

    a_vals = torch.arange(p, dtype=torch.long).repeat_interleave(p)  # [p^2]
    b_vals = torch.arange(p, dtype=torch.long).repeat(p)             # [p^2]
    op_tok = torch.full((p * p,), p, dtype=torch.long)               # "+" token
    eq_tok = torch.full((p * p,), p + 1, dtype=torch.long)           # "=" token
    X = torch.stack([a_vals, op_tok, b_vals, eq_tok], dim=1)         # [p^2, 4]
    Y = (a_vals + b_vals) % p                                        # [p^2]

    idx = torch.randperm(p * p, generator=rng)
    n_train = int(p * p * train_ratio)

    return DatasetBundle(
        train_ds=TensorDataset(X[idx[:n_train]], Y[idx[:n_train]]),
        val_ds=TensorDataset(X[idx[n_train:]], Y[idx[n_train:]]),
        input_shape=(4,),
        output_dim=p,
        task="multiclass",
        extra={"vocab_size": p + 2, "seq_len": 4, "p": p},
    )
