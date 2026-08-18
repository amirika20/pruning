"""OSSCAR -- Meng, Ibrahim, Behdin, Hazimeh, Ponomareva & Mazumder,
"OSSCAR: One-Shot Structured Pruning in Vision and Language Models with
Combinatorial Optimization", ICML 2024 (arXiv:2403.12983).

Ported from the official implementation (github.com/mazumder-lab/OSSCAR,
`prune_algo.py` / `osscar_prune.py`). Dense layers correspond to their
ksize=1; conv layers to ksize=kH*kW (the paper's ResNet50/MobileNet setup,
Section 4: a conv becomes a matrix multiply over unfolded input patches).

The method prunes input structures of the *consumer* layer -- in this repo's
terms, the neurons/channels of prunable layer `idx`, whose activations X feed
the consumer's weights B ([H*kk, fan_out]; kk=1 for Linear consumers, kH*kW
for Conv2d consumers, so each prunable unit owns a contiguous group of kk
rows). It minimizes the layer-wise reconstruction loss over the surviving
weights W:

    L(W) = 1/2 Tr(W^T H W) - Tr(G^T W),   H = X^T X,   G = H @ B_dense

(the quadratic form of ||X B_dense - X W||_F^2, with X the consumer's inputs
-- unfolded im2col patches for conv, as in the official get_hessian), subject
to whole row-groups of W being zeroed. Following the official code:

* H is damped: H += lambda2 * mean(diag(H)) * I, and G is built from the
  damped H, so the target is the ridge-regularized dense output.
* Dead inputs (diag(H) == 0 over a unit's whole group: neurons/channels that
  never activate on the calibration data) have their rows of B zeroed and are
  pruned first, for free.
* `fastprune` (their OSSCAR_fastprune): greedy elimination. Per step, each
  kept unit j is scored by the exact objective increase of zeroing its group
  with an optimal update of the rest (OBS-style):
      score_j = 1/2 Tr(W_j^T inv([H^-1]_jj) W_j)
  (= ||W_j||^2 / (2 [H^-1]_jj) for ksize=1). The `update_iter` lowest-scoring
  units are removed per step, with the weights and H^-1 downdated in closed
  form (block matrix inversion), so no re-factorization is needed until the
  final exact re-solve on the support.
* `local_search` (their OSSCAR_local_search): swap refinement. Per
  iteration, remove the `local_swap` kept units with the lowest scores,
  then re-add the `local_swap` pruned units with the highest re-entry
  gains (Schur-complement form), recompute the exact solution, and accept
  only if the objective improved -- otherwise restore the best support and
  stop (the official defaults' behavior).
* The returned consumer weights are the exact least-squares solution on the
  final support (pruned rows zero); the consumer's bias is untouched.

One-shot: designed to run without fine-tuning (pair with finetune.epochs: 0
for the paper's protocol). Supports Linear and Conv2d consumers -- i.e. the
FC-style models (mlp, resmlp, transformer FFN) and conv models exposing
outgoing_module/outgoing_weights (resnet)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import PrunableModel
from src.pruning.registry import PruneContext, PruneDecision, PruningMethod, register_pruning_method


def _group_rows(n_units: int, kk: int, device) -> torch.Tensor:
    """[n_units, kk]: the rows of XtX/W owned by each prunable unit."""
    return torch.arange(n_units * kk, device=device).view(n_units, kk)


def _expand(mask: torch.Tensor, kk: int) -> torch.Tensor:
    """Unit-level bool mask [H] -> row-level mask [H*kk]."""
    return mask if kk == 1 else mask.repeat_interleave(kk)


def _support_inverse(XtX: torch.Tensor, prune_rows: torch.Tensor) -> torch.Tensor:
    """Full-size matrix whose kept-row block is inv(XtX[kept, kept]) and
    whose pruned rows/columns are zero -- the embedded support inverse the
    official code maintains."""
    supp = (~prune_rows).nonzero(as_tuple=True)[0]
    inv = torch.zeros_like(XtX)
    inv[supp[:, None], supp] = torch.linalg.inv(XtX[supp[:, None], supp])
    return inv


def _solve_support(XtX: torch.Tensor, XtY: torch.Tensor, prune_rows: torch.Tensor) -> torch.Tensor:
    """Exact least-squares weights on the kept rows, zero elsewhere."""
    supp = (~prune_rows).nonzero(as_tuple=True)[0]
    W = torch.zeros_like(XtY)
    W[supp] = torch.linalg.inv(XtX[supp[:, None], supp]) @ XtY[supp]
    return W


def _objective(W: torch.Tensor, XtX: torch.Tensor, XtY: torch.Tensor) -> torch.Tensor:
    return (-W * XtY + 0.5 * W * (XtX @ W)).sum()


def _removal_scores(
    W: torch.Tensor, XtX_inv: torch.Tensor, prune_list: torch.Tensor, kk: int
) -> torch.Tensor:
    """OBS-style objective increase of zeroing each kept unit's row-group:
    1/2 Tr(W_j^T inv([H^-1]_jj) W_j). (Pruned units' zero blocks are
    regularized to stay invertible, then masked to +inf -- as in the
    official code's +prune_list on the denominator.)"""
    if kk == 1:
        scores = (W ** 2).sum(dim=1) / (2.0 * (torch.diag(XtX_inv) + prune_list.to(W.dtype)))
        scores[prune_list] = torch.inf
        return scores
    n_units = prune_list.shape[0]
    rows = _group_rows(n_units, kk, W.device)
    blocks = XtX_inv[rows[:, :, None], rows[:, None, :]]  # [H, kk, kk]
    eye = torch.eye(kk, dtype=W.dtype, device=W.device)
    blocks = blocks + eye * prune_list.to(W.dtype).view(-1, 1, 1)
    Wg = W.view(n_units, kk, -1)
    scores = 0.5 * (Wg * torch.linalg.solve(blocks, Wg)).sum(dim=(1, 2))
    scores[prune_list] = torch.inf
    return scores


def _remove_block(
    idx: torch.Tensor, W: torch.Tensor, XtX_inv: torch.Tensor, prune_list: torch.Tensor, kk: int
) -> None:
    """Zero the row-groups of units `idx` with the optimal update of the
    remaining weights, and downdate the embedded support inverse accordingly
    (block inversion). In-place."""
    rows = idx if kk == 1 else (idx[:, None] * kk + torch.arange(kk, device=idx.device)).flatten()
    inv_block = torch.linalg.inv(XtX_inv[rows[:, None], rows])
    W -= XtX_inv[:, rows] @ inv_block @ W[rows]
    W[rows] = 0
    XtX_inv -= XtX_inv[:, rows] @ inv_block @ XtX_inv[rows]
    XtX_inv[rows, :] = 0
    XtX_inv[:, rows] = 0
    prune_list[idx] = True


@register_pruning_method("osscar")
class OSSCAR(PruningMethod):
    """One-shot structured pruning via combinatorial optimization (OSSCAR).

    Exactly one of `prune_fraction` / `n_remove` must be set (the method is
    budgeted; it has no automatic cutoff). Neurons dead on the calibration
    data count toward the budget and are pruned first.

    prune_fraction: fraction of the layer's neurons/channels to remove.
    n_remove:       exact number of neurons/channels to remove.
    lambda2:        ridge damping of H, relative to mean(diag(H)). Official
                    default 1e-2.
    update_iter:    units removed per greedy step in fastprune. Official
                    default 1 (one at a time, most accurate).
    local_search:   run the swap refinement after fastprune.
    local_iter:     max local-search iterations (official default 20; stops
                    at the first non-improving iteration).
    local_swap:     units swapped per local-search iteration (defaults to
                    update_iter, as in the official FC-layer pipeline).
    chunk_size:     calibration inputs are forwarded in chunks of this many
                    samples when accumulating H. Conv consumers additionally
                    accumulate their unfolded patches in sub-chunks of
                    `conv_chunk` images to bound memory.
    """

    def __init__(
        self,
        prune_fraction: float | None = None,
        n_remove: int | None = None,
        lambda2: float = 1e-2,
        update_iter: int = 1,
        local_search: bool = True,
        local_iter: int = 20,
        local_swap: int | None = None,
        chunk_size: int = 4096,
        conv_chunk: int = 8,
    ):
        if (prune_fraction is None) == (n_remove is None):
            raise ValueError("osscar needs exactly one of prune_fraction / n_remove")
        self.prune_fraction = prune_fraction
        self.n_remove = n_remove
        self.lambda2 = lambda2
        self.update_iter = max(1, update_iter)
        self.local_search = local_search
        self.local_iter = local_iter
        self.local_swap = local_swap if local_swap is not None else self.update_iter
        self.chunk_size = chunk_size
        self.conv_chunk = conv_chunk

    # ── calibration statistics ───────────────────────────────────────────

    def _collect_XtX(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> torch.Tensor:
        """H = X^T X of the consumer's inputs over the calibration data,
        captured with a forward pre-hook and accumulated in float64.
        Linear consumers: sequence activations ([N, T, H]) are flattened to
        rows like the official add_batch. Conv2d consumers: inputs are
        unfolded to im2col patch rows [N*L, Cin*kH*kW] (official get_hessian
        for conv layers), in sub-chunks of conv_chunk images to bound memory."""
        consumer = model.outgoing_module(layer_idx)
        if isinstance(consumer, nn.Conv2d):
            R = consumer.in_channels * consumer.kernel_size[0] * consumer.kernel_size[1]
            XtX = torch.zeros((R, R), dtype=torch.float64, device=ctx.device)

            def hook(_module, args):
                x = args[0].detach()
                for sub in x.split(self.conv_chunk):
                    u = F.unfold(sub, consumer.kernel_size, dilation=consumer.dilation,
                                 padding=consumer.padding, stride=consumer.stride)
                    u = u.transpose(1, 2).reshape(-1, R).to(torch.float64)
                    XtX.add_(u.t() @ u)
        else:
            R = consumer.weight.shape[1]
            XtX = torch.zeros((R, R), dtype=torch.float64, device=ctx.device)

            def hook(_module, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float64)
                XtX.add_(x.t() @ x)

        handle = consumer.register_forward_pre_hook(hook)
        model.eval()
        with torch.no_grad():
            for chunk in ctx.train_inputs.split(self.chunk_size):
                model(chunk)
        handle.remove()
        return XtX

    # ── the two optimization stages (official prune_algo.py) ─────────────

    def _fastprune(
        self, W: torch.Tensor, XtX: torch.Tensor, prune_list: torch.Tensor, n_to_remove: int, kk: int
    ) -> None:
        """Greedy stage (OSSCAR_fastprune): remove n_to_remove units in
        chunks of update_iter, lowest removal score first. In-place on
        W / prune_list."""
        if n_to_remove <= 0:
            return
        XtX_inv = _support_inverse(XtX, _expand(prune_list, kk))
        n_steps = max(1, n_to_remove // self.update_iter)
        quo, rem = divmod(n_to_remove, n_steps)
        for step in range(n_steps):
            chunk = quo + (1 if step < rem else 0)
            scores = _removal_scores(W, XtX_inv, prune_list, kk)
            idx = torch.argsort(scores)[:chunk]
            _remove_block(idx, W, XtX_inv, prune_list, kk)

    def _reentry_gains(
        self, XtX: torch.Tensor, XtY: torch.Tensor, XtX_inv: torch.Tensor,
        prune_list: torch.Tensor, kk: int
    ) -> torch.Tensor:
        """Gain of re-adding each pruned unit to the current support:
        obj_in_j = 1/2 g_j^T C_j g_j per output, with C_j the inverse Schur
        complement of the unit's block and g_j = G_j - H_j,supp @ W_supp."""
        supp = (~_expand(prune_list, kk)).nonzero(as_tuple=True)[0]
        H_inv = XtX_inv[supp[:, None], supp]
        H_invG = H_inv @ XtY[supp]
        if kk == 1:
            C = 1.0 / (
                torch.diag(XtX)
                - (XtX[supp, :] * (H_inv @ XtX[supp, :])).sum(dim=0)
                + (~prune_list).to(XtX.dtype) * 1e-8
            )
            gt = XtY - XtX[:, supp] @ H_invG
            gains = (gt ** 2).sum(dim=1) * C / 2.0
            gains[~prune_list] = -torch.inf
            return gains
        n_units = prune_list.shape[0]
        rows = _group_rows(n_units, kk, XtX.device)
        Xs = XtX[:, supp]                                   # [R, S]
        gt = (XtY - Xs @ H_invG).view(n_units, kk, -1)      # [H, kk, F]
        T = (Xs @ H_inv).view(n_units, kk, -1)              # rows grouped: [H, kk, S]
        Xs_g = Xs.view(n_units, kk, -1)
        S_blocks = XtX[rows[:, :, None], rows[:, None, :]] - T @ Xs_g.transpose(1, 2)
        eye = torch.eye(kk, dtype=XtX.dtype, device=XtX.device)
        S_blocks = S_blocks + eye * (~prune_list).to(XtX.dtype).view(-1, 1, 1) * 1e-8
        gains = 0.5 * (gt * torch.linalg.solve(S_blocks, gt)).sum(dim=(1, 2))
        gains[~prune_list] = -torch.inf
        return gains

    def _local_search_stage(
        self, XtX: torch.Tensor, XtY: torch.Tensor, prune_list: torch.Tensor, kk: int
    ) -> torch.Tensor:
        """Swap refinement (OSSCAR_local_search, switch_lb=1 behavior):
        returns the best prune_list found."""
        best_prune = prune_list.clone()
        W = _solve_support(XtX, XtY, _expand(prune_list, kk))
        obj_cur = _objective(W, XtX, XtY)

        for _ in range(self.local_iter):
            # 1. Remove the local_swap kept units with the lowest scores.
            XtX_inv = _support_inverse(XtX, _expand(prune_list, kk))
            scores = _removal_scores(W, XtX_inv, prune_list, kk)
            idx = torch.argsort(scores)[: self.local_swap]
            _remove_block(idx, W, XtX_inv, prune_list, kk)

            # 2. Re-add the local_swap pruned units with the highest
            #    re-entry gains (Schur-complement form).
            gains = self._reentry_gains(XtX, XtY, XtX_inv, prune_list, kk)
            idx2 = torch.argsort(-gains)[: self.local_swap]
            prune_list[idx2] = False

            # 3. Exact recompute; accept only on improvement, else restore
            #    the best support and stop (official defaults).
            W = _solve_support(XtX, XtY, _expand(prune_list, kk))
            obj_new = _objective(W, XtX, XtY)
            if obj_new < obj_cur * (1 + 1e-9):
                best_prune = prune_list.clone()
                obj_cur = obj_new
            else:
                break

        return best_prune

    # ── selection ────────────────────────────────────────────────────────

    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> PruneDecision:
        layer = model.prunable_layer(layer_idx)
        consumer = model.outgoing_module(layer_idx)
        if not isinstance(consumer, (nn.Linear, nn.Conv2d)):
            raise TypeError(
                f"osscar supports Linear and Conv2d consumers; layer {layer_idx} of "
                f"{type(model).__name__} feeds a {type(consumer).__name__}"
            )

        XtX = self._collect_XtX(model, layer_idx, ctx)
        B = model.outgoing_weights(layer_idx).detach().to(torch.float64).clone()  # [H*kk, fan_out]
        n_units = layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features
        kk, rem = divmod(B.shape[0], n_units)
        assert rem == 0, f"consumer rows {B.shape[0]} not a multiple of {n_units} prunable units"

        already = torch.zeros(n_units, dtype=torch.bool, device=XtX.device)
        for k in ctx.already_selected:
            already[k] = True

        # Dead inputs: units silent (over their whole patch group) on all
        # calibration data. Their rows can't be reconstructed from (and
        # contribute nothing) -- zero them and prune them first, like the
        # official pipeline.
        dead = (torch.diag(XtX).view(n_units, kk).sum(dim=1) == 0) & ~already
        B[_expand(dead | already, kk)] = 0

        # Ridge damping, and the target G from the damped H (official get_XTY).
        R = XtX.shape[0]
        XtX += torch.eye(R, dtype=XtX.dtype, device=XtX.device) * self.lambda2 * torch.diag(XtX).mean()
        XtY = XtX @ B

        pool = int((~already).sum())
        target = self.n_remove if self.n_remove is not None else int(round(self.prune_fraction * pool))
        target = min(target, pool - 1)  # keep at least one unit
        if target <= 0:
            return PruneDecision(remove=[])

        prune_list = already.clone()
        prune_list[dead] = True
        # Optimal dense-support solution as the starting point (equals B when
        # nothing is pre-pruned, since G = H @ B).
        W = _solve_support(XtX, XtY, _expand(prune_list, kk))

        remaining = target - int(dead.sum())
        self._fastprune(W, XtX, prune_list, remaining, kk)
        if self.local_search and remaining > 0:
            prune_list = self._local_search_stage(XtX, XtY, prune_list, kk)

        W_sol = _solve_support(XtX, XtY, _expand(prune_list, kk))

        remove = (prune_list & ~already).nonzero(as_tuple=True)[0].tolist()
        model_dtype = model.outgoing_weights(layer_idx).dtype
        return PruneDecision(remove=remove, new_outgoing=W_sol.to(model_dtype))
