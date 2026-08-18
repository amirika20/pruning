"""WikiText language-modeling builder: tokenized fixed-length sequences.

The standard perplexity protocol (GPTQ/OSSCAR): each split's articles are
joined, tokenized once, and cut into contiguous `seq_len`-token chunks. The
train split provides `n_train` randomly drawn chunks (the calibration set of
one-shot methods like OSSCAR); validation/test chunks are taken in document
order -- perplexity over the full official test split is the number papers
report. Labels equal inputs; the next-token shift happens in the trainer's
`causal_lm` loss.

Downloads WikiText via HuggingFace `datasets` and the tokenizer via
`transformers` (both cached under ~/.cache/huggingface).
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.registry import DatasetBundle, register_dataset


def _chunk(ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    n = ids.shape[0] // seq_len
    return ids[: n * seq_len].view(n, seq_len)


@register_dataset("wikitext")
def wikitext(
    data_seed: int = 0,
    tokenizer: str = "facebook/opt-125m",
    seq_len: int = 512,
    n_train: int = 32,
    n_val: int | None = 8,
    n_test: int | None = None,  # None = the full official test split
    dataset_name: str = "wikitext-2-raw-v1",
) -> DatasetBundle:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    raw = load_dataset("Salesforce/wikitext", dataset_name)

    def encode(split: str) -> torch.Tensor:
        text = "\n\n".join(raw[split]["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
        return _chunk(ids, seq_len)

    train_chunks = encode("train")
    val_chunks = encode("validation")
    test_chunks = encode("test")

    rng = torch.Generator().manual_seed(data_seed)
    pick = torch.randperm(train_chunks.shape[0], generator=rng)[:n_train]
    x_train = train_chunks[pick]
    x_val = val_chunks[:n_val] if n_val is not None else val_chunks
    x_test = test_chunks[:n_test] if n_test is not None else test_chunks

    return DatasetBundle(
        train_ds=TensorDataset(x_train, x_train.clone()),
        val_ds=TensorDataset(x_val, x_val.clone()),
        test_ds=TensorDataset(x_test, x_test.clone()),
        input_shape=(seq_len,),
        output_dim=len(tok),
        task="causal_lm",
        extra={"tokenizer": tokenizer, "seq_len": seq_len},
    )
