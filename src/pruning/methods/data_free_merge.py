"""Data-free neuron merging -- Srinivas & Babu, "Data-free Parameter Pruning
for Deep Neural Networks", BMVC 2015 (arXiv:1507.06149).

Recreated from the paper (no official code was released):

* Sec 3: for a layer with weight-sets W_i (input weights + bias) and outgoing
  coefficients a_i, two similar weight-sets are redundant: delete neuron j and
  perform the 'surgery' a_i <- a_i + a_j. The expected squared change of the
  output activations is bounded by the saliency

      s_{i,j} = <a_j^2> * ||eps_{i,j}||^2        (Eq. 4)

  where <a_j^2> averages the squared outgoing weights of the *removed* neuron
  j over all next-layer neurons.

* Sec 3.3 (weight normalization): ReLU is positively homogeneous
  (max(0, ax) = a*max(0, x)), so every weight-set is scaled to unit norm and
  the factor alpha_k = ||W'_k|| is folded into the next layer's coefficients
  (a_hat = alpha * a). We keep the model itself in its original
  parameterization and fold the factors into the merge instead:
  a_i += (alpha_j / alpha_i) * a_j -- algebraically identical.

* Sec 3.4 (heuristics): normalize only the non-bias weights, and use a
  similarity with 'sensible-sized' contributions from weights and biases:

      ||eps_{i,j}|| = ||Wn_i - Wn_j|| / ||Wn_i + Wn_j||  +  |bn_i - bn_j| / |bn_i + bn_j|

  with Wn = W'/alpha (unit norm) and bn = b/alpha.

* Sec 3 (procedure): compute the full saliency matrix once; repeatedly pick
  the minimum entry (i', j'), delete j', update a_i' <- a_i' + a_j', then
  remove row/column j' and recompute column i' (only <a_i'^2> changed).

* Sec 4 (how many to remove): the data-free criterion takes the mode of the
  histogram of saliency values as the cutoff and prunes while the minimum
  saliency stays below it; `cutoff_fraction` then keeps that fraction of the
  predicted removals (the paper uses fractions 0.25/0.5/0.75 for large
  networks). `n_remove`/`prune_fraction` bypass the cutoff with an explicit
  budget.

Fully-connected style layers only (mlp, resmlp, transformer FFN), as in the
paper -- conv layers with BatchNorm between the filter and its consumer have
no exact outgoing-weight surgery.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from src.models.registry import MergeOp, PrunableModel
from src.pruning.registry import PruneContext, PruneDecision, PruningMethod, register_pruning_method

_TINY = 1e-12


@register_pruning_method("data_free_merge")
class DataFreeMerge(PruningMethod):
    """Saliency-ranked pairwise neuron merging (Srinivas & Babu, BMVC 2015).

    Exactly one of the budget params should be set; with none set, the
    paper's data-free histogram-mode cutoff decides how many to remove.

    prune_fraction:   remove this fraction of the layer's neurons.
    n_remove:         remove exactly this many neurons.
    cutoff_fraction:  scale the histogram-mode prediction (paper Sec 4 uses
                      0.25/0.5/0.75 for large nets). Default 1.0.
    histogram_bins:   bins for the saliency histogram of the auto cutoff.
    """

    def __init__(
        self,
        prune_fraction: float | None = None,
        n_remove: int | None = None,
        cutoff_fraction: float = 1.0,
        histogram_bins: int = 100,
    ):
        if prune_fraction is not None and n_remove is not None:
            raise ValueError("set at most one of prune_fraction / n_remove")
        self.prune_fraction = prune_fraction
        self.n_remove = n_remove
        self.cutoff_fraction = cutoff_fraction
        self.histogram_bins = histogram_bins

    # ── saliency construction ────────────────────────────────────────────

    @staticmethod
    def _epsilon_sq(W_n: np.ndarray, b_n: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """[H, H] matrix of ||eps_{i,j}||^2 (Sec 3.4 similarity, squared).

        For the unit-norm rows of W_n, ||Wn_i - Wn_j|| / ||Wn_i + Wn_j||
        = sqrt((1 - c) / (1 + c)) with c = cos(angle between the rows).
        Invalid rows (zero-norm or excluded neurons) become +inf.
        """
        c = np.clip(W_n @ W_n.T, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight_term = np.sqrt((1.0 - c) / (1.0 + c + _TINY))
            bias_term = np.abs(b_n[:, None] - b_n[None, :]) / (np.abs(b_n[:, None] + b_n[None, :]) + _TINY)
        eps_sq = (weight_term + bias_term) ** 2
        eps_sq[~valid, :] = np.inf
        eps_sq[:, ~valid] = np.inf
        np.fill_diagonal(eps_sq, np.inf)
        return eps_sq

    def _budget(self, saliency: np.ndarray, n_valid: int) -> tuple[int, float]:
        """(max removals, saliency cutoff). Explicit budgets disable the
        cutoff; otherwise the histogram mode (Sec 4) is the cutoff and the
        budget is uncapped (up to n_valid - 1, so one neuron always survives)."""
        max_removals = n_valid - 1
        if self.n_remove is not None:
            return min(self.n_remove, max_removals), np.inf
        if self.prune_fraction is not None:
            return min(int(round(self.prune_fraction * n_valid)), max_removals), np.inf

        finite = saliency[np.isfinite(saliency)]
        if finite.size == 0:
            return 0, np.inf
        # The paper's Fig 2(b) histograms the bulk of the distribution (its
        # x-axis stops at ~5 while a_j^2-driven outliers reach orders of
        # magnitude higher). Equal-width bins over the full range would dump
        # the whole bulk into the first bin and put the "mode" above nearly
        # every pair, so bin only up to the 95th percentile.
        bulk = finite[finite <= np.percentile(finite, 95)]
        counts, edges = np.histogram(bulk, bins=self.histogram_bins)
        mode = int(np.argmax(counts))
        cutoff = float((edges[mode] + edges[mode + 1]) / 2)
        if self.cutoff_fraction >= 1.0:
            return max_removals, cutoff
        # Paper Sec 4: take a fraction of the *number* the cutoff predicts.
        predicted = self._count_below_cutoff(saliency.copy(), cutoff, max_removals)
        return int(round(self.cutoff_fraction * predicted)), np.inf

    @staticmethod
    def _count_below_cutoff(saliency: np.ndarray, cutoff: float, max_removals: int) -> int:
        """Dry-run of the greedy loop (without tracking merges) to count how
        many removals the cutoff predicts. Column updates from surgery only
        raise <a_i^2>, so ignoring them gives the paper's upfront estimate."""
        n = 0
        while n < max_removals:
            i, j = np.unravel_index(np.argmin(saliency), saliency.shape)
            if not np.isfinite(saliency[i, j]) or saliency[i, j] > cutoff:
                break
            saliency[j, :] = np.inf
            saliency[:, j] = np.inf
            n += 1
        return n

    # ── selection ────────────────────────────────────────────────────────

    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> PruneDecision:
        layer = model.prunable_layer(layer_idx)
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                "data_free_merge supports fully-connected layers only (the paper's Sec 3.4 "
                f"heuristics are FC-specific); layer {layer_idx} of {type(model).__name__} "
                f"is a {type(layer).__name__}"
            )

        W = layer.weight.data.detach().cpu().numpy().astype(np.float64)  # [H, d]
        b = layer.bias.data.detach().cpu().numpy().astype(np.float64)    # [H]
        A = model.outgoing_weights(layer_idx).detach().cpu().numpy().astype(np.float64)  # [H, fan_out]
        H = W.shape[0]

        alpha = np.linalg.norm(W, axis=1)  # Sec 3.4: non-bias norm only
        valid = alpha > 1e-8
        for k in ctx.already_selected:
            valid[k] = False
        n_valid = int(valid.sum())
        if n_valid < 2:
            return PruneDecision(remove=[])

        # Sec 3.3 normalization: unit-norm weight-sets, factor folded into
        # the outgoing coefficients.
        safe_alpha = np.where(valid, alpha, 1.0)
        W_n = W / safe_alpha[:, None]
        b_n = b / safe_alpha
        A_hat = A * safe_alpha[:, None]

        eps_sq = self._epsilon_sq(W_n, b_n, valid)          # [H, H], symmetric
        a_sq = (A_hat ** 2).mean(axis=1)                    # <a_hat_j^2> per neuron
        saliency = a_sq[None, :] * eps_sq                   # s_{i,j}: remove j, merge into i

        budget, cutoff = self._budget(saliency, n_valid)

        removed: list[int] = []
        merges: list[MergeOp] = []
        while len(removed) < budget:
            i, j = np.unravel_index(np.argmin(saliency), saliency.shape)
            if not np.isfinite(saliency[i, j]) or saliency[i, j] > cutoff:
                break
            # Surgery: a_i <- a_i + a_j (normalized space); in the model's
            # original parameterization the transfer is scaled by alpha_j/alpha_i.
            merges.append(MergeOp(removed=int(j), survivor=int(i), scale=float(alpha[j] / alpha[i])))
            removed.append(int(j))
            A_hat[i] += A_hat[j]
            a_sq[i] = (A_hat[i] ** 2).mean()
            # Remove row/column j; recompute column i (only <a_i^2> changed).
            saliency[j, :] = np.inf
            saliency[:, j] = np.inf
            saliency[:, i] = a_sq[i] * eps_sq[:, i]
            saliency[np.array(removed), i] = np.inf
            saliency[i, i] = np.inf

        return PruneDecision(remove=removed, merges=merges)
