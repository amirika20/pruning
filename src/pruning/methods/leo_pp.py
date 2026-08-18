"""LEO++ -- Serra, Yu, Kumar & Ramalingam, "Scaling Up Exact Neural Network
Compression by ReLU Stability", NeurIPS 2021 (arXiv:2102.07804).

Lossless compression: a hidden ReLU neuron whose pre-activation never changes
sign over the whole input domain X can be removed without changing the
network function at all.

* Stably INACTIVE (y_i <= 0 for every x in X): the neuron always outputs 0,
  so it is simply deleted -- no consumer change needed.
* Stably ACTIVE (y_i >= 0 for every x in X): ReLU is the identity there, so
  the neuron's output is affine in the layer's input. If the weight rows of
  the stably-active set have rank r < |active|, every dependent row
  w_i = sum_j alpha_j w_j (over a basis Q of independent stably-active rows)
  can be folded into the consumer (paper Alg. 2, generic case):

      W_out[:, j] += alpha_j * W_out[:, i]           for each basis j
      b_out       += W_out[:, i] * (b_i - alpha . b_Q)

  expressed here as MergeOps plus a PruneDecision.bias_delta; the basis
  neurons stay. (The paper's whole-layer folding / constant-network collapse
  need layer deletion, which neither this repo's surgery nor the official
  released code -- github.com/yuxwind/ExactCompression, prune_network.py --
  implements; like the official code we apply the generic per-layer case.)

Stability certification follows the paper's two-stage pipeline:

1. Empirical pre-pass (their train_fcnn.py --eval-stable): forward the
   training set; only neurons active on ALL samples (stably-active
   candidates, Q) or on NONE (stably-inactive candidates, P) remain.
2. ISA -- Identifying Stable Activations (paper Alg. 1): one MILP over the
   sub-network up to this layer that maximizes the number of candidate
   activation states it can flip, C(P,Q) = max sum(p) + sum(q) subject to a
   big-M ReLU encoding (their eq. 1-4 / 7-11; interval-arithmetic Ms floored
   at 1 and doubled, and p,q continuous, exactly as the official code).
   Every flipped candidate is disproven; when the optimum reaches 0 the
   survivors are certified stable over ALL of X (their Prop. 1). The
   official Gurobi lazy-constraint callback becomes iterative re-solves with
   the flipped candidates' p/q fixed to 0 -- their Corollary 2 bounds this
   by |candidates| solves -- so the method runs on scipy's HiGHS backend
   with no Gurobi dependency. Neurons the interval bounds already certify
   are settled without (and linearized inside) the MILP, and for the first
   hidden layer interval bounds over a box are exact, so no MILP is needed.

The certification domain X is an axis-aligned box (the paper uses [0,1]^d
raw-pixel inputs; this repo normalizes inputs, so pass the normalized bounds
or use the training data's per-feature range). Lossless means: the pruned
network computes the same function as the original ON X (up to float
round-off) -- pair with finetune.epochs: 0. The paper trains with L1
regularization to induce stability (training.l1 here); without it, few
neurons of a trained net are typically stable and the method may remove
little or nothing. It is unbudgeted by design.

certify: "milp"       full ISA certification (the paper's method; exact on X).
         "interval"   interval-arithmetic bounds only (sound but incomplete;
                      exact for the first hidden layer, loose deeper).
         "empirical"  trust the training-set pass (what the official released
                      pruning script actually consumes: lossless w.r.t. the
                      training data, not certified over all of X).
"milp"/"interval" need the exact sub-network up to the layer as an
input->Linear->ReLU chain, i.e. the plain `mlp` model; "empirical" works on
any fully-connected-style layer (mlp, resmlp, transformer FFN).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.mlp import MLP
from src.models.registry import MergeOp, PrunableModel
from src.pruning.registry import PruneContext, PruneDecision, PruningMethod, register_pruning_method

_CERTIFY_MODES = ("milp", "interval", "empirical")


@register_pruning_method("leo_pp")
class LEOPlusPlus(PruningMethod):
    """Exact (lossless) compression via ReLU stability (Serra et al. 2021).

    certify:     "milp" (paper, default) | "interval" | "empirical" -- see
                 module docstring.
    input_box:   the certification domain X. "data" (default) uses the
                 per-feature min/max of the training inputs; [lo, hi] is a
                 uniform box in the model's (normalized) input space, e.g.
                 [-0.4243, 2.8216] for pixel range [0, 1] under this repo's
                 MNIST normalization. Ignored by certify: "empirical".
    time_limit:  total MILP seconds per layer (certify: "milp"). Candidates
                 still unresolved at the limit are conservatively kept.
    fold_active: also fold rank-deficient stably-active neurons (paper
                 Alg. 2 lines 16-24). False removes stably-inactive only.
    dep_tol:     relative residual below which a stably-active weight row
                 counts as linearly dependent on the basis.
    chunk_size:  forward-pass chunk for the empirical candidate pass.
    """

    def __init__(
        self,
        certify: str = "milp",
        input_box: str | list[float] = "data",
        time_limit: float = 300.0,
        fold_active: bool = True,
        dep_tol: float = 1e-6,
        chunk_size: int = 4096,
    ):
        if certify not in _CERTIFY_MODES:
            raise ValueError(f"leo_pp certify must be one of {_CERTIFY_MODES}, got {certify!r}")
        if input_box != "data":
            if not (isinstance(input_box, (list, tuple)) and len(input_box) == 2):
                raise ValueError("leo_pp input_box must be 'data' or [lo, hi]")
            if float(input_box[0]) > float(input_box[1]):
                raise ValueError("leo_pp input_box has lo > hi")
        self.certify = certify
        self.input_box = input_box
        self.time_limit = time_limit
        self.fold_active = fold_active
        self.dep_tol = dep_tol
        self.chunk_size = chunk_size

    # ── stage 1: empirical candidates (their --eval-stable pre-pass) ──────

    def _empirical_candidates(
        self, model: PrunableModel, layer_idx: int, ctx: PruneContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """(ever_active, ever_inactive) bool [H] over the training inputs:
        pre-activation > 0 on some sample / <= 0 on some sample."""
        layer = model.prunable_layer(layer_idx)
        H = layer.out_features
        ever_active = torch.zeros(H, dtype=torch.bool, device=ctx.device)
        ever_inactive = torch.zeros(H, dtype=torch.bool, device=ctx.device)

        def hook(_module, _input, output):
            y = output.detach().reshape(-1, output.shape[-1])  # [N(*T), H]
            ever_active.logical_or_((y > 0).any(dim=0))
            ever_inactive.logical_or_((y <= 0).any(dim=0))

        handle = layer.register_forward_hook(hook)
        model.eval()
        with torch.no_grad():
            for chunk in ctx.train_inputs.split(self.chunk_size):
                model(chunk)
        handle.remove()
        return ever_active.cpu().numpy(), ever_inactive.cpu().numpy()

    # ── stage 2: certification over the input box ─────────────────────────

    def _chain(self, model: PrunableModel, layer_idx: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Weights/biases (float64) of hidden layers 0..layer_idx -- the
        sub-network that determines this layer's pre-activations."""
        Ws, bs = [], []
        for m in range(layer_idx + 1):
            lin = model.prunable_layer(m)
            Ws.append(lin.weight.detach().cpu().numpy().astype(np.float64))
            bs.append(lin.bias.detach().cpu().numpy().astype(np.float64))
        return Ws, bs

    def _box(self, ctx: PruneContext, n0: int) -> tuple[np.ndarray, np.ndarray]:
        if self.input_box == "data":
            flat = ctx.train_inputs.reshape(-1, ctx.train_inputs.shape[-1]).to(torch.float64)
            return flat.min(dim=0).values.cpu().numpy(), flat.max(dim=0).values.cpu().numpy()
        lo, hi = float(self.input_box[0]), float(self.input_box[1])
        return np.full(n0, lo), np.full(n0, hi)

    @staticmethod
    def _interval_bounds(
        Ws: list[np.ndarray], bs: list[np.ndarray], lo: np.ndarray, hi: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per layer, (lb, ub) on the PRE-activations over the box, by
        interval arithmetic (paper A6). Exact for the first hidden layer."""
        bounds = []
        lo_x, hi_x = lo, hi
        for W, b in zip(Ws, bs):
            Wp, Wn = np.clip(W, 0, None), np.clip(W, None, 0)
            ub = Wp @ hi_x + Wn @ lo_x + b
            lb = Wp @ lo_x + Wn @ hi_x + b
            bounds.append((lb, ub))
            lo_x, hi_x = np.maximum(lb, 0.0), np.maximum(ub, 0.0)
        return bounds

    def _isa_milp(
        self,
        Ws: list[np.ndarray],
        bs: list[np.ndarray],
        lo: np.ndarray,
        hi: np.ndarray,
        bounds: list[tuple[np.ndarray, np.ndarray]],
        P_rem: list[int],
        Q_rem: list[int],
    ) -> tuple[set[int], set[int]]:
        """ISA (paper Alg. 1) for the last layer of the chain: returns the
        (stably_inactive, stably_active) neuron indices certified over the
        box. Candidates disproven or unresolved at the time limit are
        excluded (sound either way)."""
        import scipy.sparse as sp
        from scipy.optimize import Bounds, LinearConstraint, milp

        t = len(Ws) - 1
        cand = sorted(set(P_rem) | set(Q_rem))
        c_pos = {i: k for k, i in enumerate(cand)}
        n0 = Ws[0].shape[1]
        widths = [W.shape[0] for W in Ws[:t]]

        # Variable layout: x0 | per below-layer m: x_m, chi_m, z_m | target
        # candidates: x_c, chi_c, z_c | p | q.
        pos = 0

        def alloc(n: int) -> np.ndarray:
            nonlocal pos
            out = np.arange(pos, pos + n)
            pos += n
            return out

        v_x0 = alloc(n0)
        v_x, v_chi, v_z = [], [], []
        for n_m in widths:
            v_x.append(alloc(n_m)); v_chi.append(alloc(n_m)); v_z.append(alloc(n_m))
        v_xc, v_chic, v_zc = alloc(len(cand)), alloc(len(cand)), alloc(len(cand))
        v_p, v_q = alloc(len(P_rem)), alloc(len(Q_rem))
        n_var = pos

        var_lb = np.full(n_var, 0.0)
        var_ub = np.full(n_var, np.inf)
        var_lb[v_x0], var_ub[v_x0] = lo, hi
        integrality = np.zeros(n_var)
        for zz in [*v_z, v_zc]:
            var_ub[zz] = 1.0
            integrality[zz] = 1
        var_ub[v_p] = 1.0
        var_ub[v_q] = 1.0

        rows, cols, vals = [], [], []
        con_lb, con_ub = [], []
        n_con = 0

        def add_row(coeffs: list[tuple[int, float]], lb_val: float, ub_val: float) -> None:
            nonlocal n_con
            for col, val in coeffs:
                rows.append(n_con); cols.append(col); vals.append(val)
            con_lb.append(lb_val); con_ub.append(ub_val)
            n_con += 1

        def encode_neuron(w: np.ndarray, b: float, prev: np.ndarray,
                          x: int, chi: int, z: int, lb_pre: float, ub_pre: float) -> None:
            """Big-M ReLU block (paper eq. 1-4), Ms as in the official code:
            M = 2*max(interval_ub, 1), mu = 2*max(-interval_lb, 1)."""
            M = 2.0 * max(ub_pre, 1.0)
            mu = 2.0 * max(-lb_pre, 1.0)
            add_row([*((prev[k], w[k]) for k in np.nonzero(w)[0]), (x, -1.0), (chi, 1.0)], -b, -b)
            add_row([(x, 1.0), (z, -M)], -np.inf, 0.0)
            add_row([(chi, 1.0), (z, mu)], -np.inf, mu)

        for m, n_m in enumerate(widths):
            prev = v_x0 if m == 0 else v_x[m - 1]
            lb_pre, ub_pre = bounds[m]
            for i in range(n_m):
                encode_neuron(Ws[m][i], bs[m][i], prev, v_x[m][i], v_chi[m][i], v_z[m][i],
                              lb_pre[i], ub_pre[i])
                # Interval-stable neurons below the target: pin z, which
                # linearizes their block (the official code's z-fixing).
                if ub_pre[i] <= 0:
                    var_ub[v_z[m][i]] = 0.0
                elif lb_pre[i] >= 0:
                    var_lb[v_z[m][i]] = 1.0

        prev = v_x0 if t == 0 else v_x[t - 1]
        lb_pre, ub_pre = bounds[t]
        for i in cand:
            k = c_pos[i]
            encode_neuron(Ws[t][i], bs[t][i], prev, v_xc[k], v_chic[k], v_zc[k],
                          lb_pre[i], ub_pre[i])
        for j, i in enumerate(P_rem):     # p <= z: activation witness
            add_row([(v_p[j], 1.0), (v_zc[c_pos[i]], -1.0)], -np.inf, 0.0)
        for j, i in enumerate(Q_rem):     # q <= 1 - z: inactivation witness
            add_row([(v_q[j], 1.0), (v_zc[c_pos[i]], 1.0)], -np.inf, 1.0)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(n_con, n_var))
        constraint = LinearConstraint(A, con_lb, con_ub)
        objective = np.zeros(n_var)
        objective[v_p] = -1.0
        objective[v_q] = -1.0

        alive_p = {j: i for j, i in enumerate(P_rem)}
        alive_q = {j: i for j, i in enumerate(Q_rem)}
        deadline = time.monotonic() + self.time_limit
        n_solves = 0
        while alive_p or alive_q:
            budget = deadline - time.monotonic()
            if budget <= 0:
                logging.warning(
                    f"  leo_pp: MILP time limit hit after {n_solves} solve(s); "
                    f"{len(alive_p) + len(alive_q)} candidate(s) left uncertified"
                )
                return set(), set()
            res = milp(objective, constraints=constraint, integrality=integrality,
                       bounds=Bounds(var_lb, var_ub), options={"time_limit": budget})
            n_solves += 1
            if res.status != 0:
                logging.warning(
                    f"  leo_pp: MILP unresolved (status={res.status}: {res.message}); "
                    f"{len(alive_p) + len(alive_q)} candidate(s) left uncertified"
                )
                return set(), set()
            if -res.fun < 0.5:
                break  # C(P,Q) = 0: every remaining candidate is stable (Prop. 1)
            # Positive objective: the solution witnesses flipped states.
            # Disprove those candidates (lazy p/q = 0 cuts become bound fixes).
            flipped = False
            for j in list(alive_p):
                if res.x[v_p[j]] > 0.5:
                    var_ub[v_p[j]] = 0.0
                    del alive_p[j]
                    flipped = True
            for j in list(alive_q):
                if res.x[v_q[j]] > 0.5:
                    var_ub[v_q[j]] = 0.0
                    del alive_q[j]
                    flipped = True
            if not flipped:  # defensive: should be impossible at optimality
                logging.warning("  leo_pp: MILP made no progress; stopping certification")
                return set(), set()
        return set(alive_p.values()), set(alive_q.values())

    # ── surgery math: rank-based folding of stably-active neurons ─────────

    def _fold_active(
        self, W: np.ndarray, b: np.ndarray, A: np.ndarray, active: list[int]
    ) -> tuple[list[int], list[MergeOp], np.ndarray]:
        """Paper Alg. 2 lines 16-24 (= official prune_active_per_layer).
        Greedy basis over the stably-active rows of W in index order (their
        find_independ_rows2); each dependent row w_i = alpha . W_basis folds
        into the consumer as MergeOps + a bias delta. Returns
        (dependents_to_remove, merges, bias_delta [fan_out])."""
        basis: list[int] = []
        B: np.ndarray | None = None  # orthonormal rows spanning the basis
        # each dependent stores the alpha solved against the basis *prefix*
        # that existed when it was found (the basis only grows afterwards)
        deps: list[tuple[int, np.ndarray]] = []
        for i in active:
            w = W[i]
            resid = w if B is None else w - (w @ B.T) @ B
            if np.linalg.norm(resid) > self.dep_tol * max(1.0, np.linalg.norm(w)):
                B = resid[None] / np.linalg.norm(resid) if B is None else \
                    np.vstack([B, resid / np.linalg.norm(resid)])
                basis.append(i)
            else:
                alpha, *_ = np.linalg.lstsq(W[basis].T, w, rcond=None)
                deps.append((i, alpha))

        merges: list[MergeOp] = []
        bias_delta = np.zeros(A.shape[1])
        for i, alpha in deps:
            pre = basis[: len(alpha)]
            for j, a_j in zip(pre, alpha):
                if abs(a_j) > 1e-12:
                    merges.append(MergeOp(removed=i, survivor=j, scale=float(a_j)))
            bias_delta += A[i] * (b[i] - float(alpha @ b[pre]))
        return [i for i, _ in deps], merges, bias_delta

    # ── selection ──────────────────────────────────────────────────────────

    def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> PruneDecision:
        layer = model.prunable_layer(layer_idx)
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                "leo_pp supports fully-connected layers only; layer "
                f"{layer_idx} of {type(model).__name__} is a {type(layer).__name__}"
            )
        if self.certify != "empirical" and not isinstance(model, MLP):
            raise TypeError(
                f"leo_pp certify={self.certify!r} needs the exact input->Linear->ReLU chain "
                f"and supports the plain 'mlp' model only, not {type(model).__name__}; "
                "use certify: 'empirical' for other fully-connected architectures"
            )

        H = layer.out_features
        ever_active, ever_inactive = self._empirical_candidates(model, layer_idx, ctx)
        valid = np.ones(H, dtype=bool)
        for k in ctx.already_selected:
            valid[k] = False
        P_cand = valid & ~ever_active     # never active on data: inactive candidates
        Q_cand = valid & ~ever_inactive   # never inactive on data: active candidates

        if self.certify == "empirical":
            inactive = set(np.nonzero(P_cand)[0].tolist())
            active = set(np.nonzero(Q_cand)[0].tolist())
        else:
            Ws, bs = self._chain(model, layer_idx)
            lo, hi = self._box(ctx, Ws[0].shape[1])
            bounds = self._interval_bounds(Ws, bs, lo, hi)
            lb_pre, ub_pre = bounds[layer_idx]
            inactive = set(np.nonzero(P_cand & (ub_pre <= 0))[0].tolist())
            active = set(np.nonzero(Q_cand & (lb_pre >= 0))[0].tolist())
            P_rem = sorted(set(np.nonzero(P_cand)[0].tolist()) - inactive)
            Q_rem = sorted(set(np.nonzero(Q_cand)[0].tolist()) - active)
            # Interval bounds are exact for the first hidden layer (the
            # pre-activation is affine over the box), so its remaining
            # candidates are provably UNstable -- nothing left to solve.
            if self.certify == "milp" and layer_idx > 0 and (P_rem or Q_rem):
                milp_inactive, milp_active = self._isa_milp(Ws, bs, lo, hi, bounds, P_rem, Q_rem)
                inactive |= milp_inactive
                active |= milp_active

        remove = sorted(inactive)
        merges: list[MergeOp] = []
        bias_delta_t: torch.Tensor | None = None
        n_folded = 0
        if self.fold_active and active:
            W = layer.weight.detach().cpu().numpy().astype(np.float64)
            b = layer.bias.detach().cpu().numpy().astype(np.float64)
            A = model.outgoing_weights(layer_idx).detach().cpu().numpy().astype(np.float64)
            deps, merges, bias_delta = self._fold_active(W, b, A, sorted(active))
            n_folded = len(deps)
            if deps:
                remove = sorted(inactive | set(deps))
                bias_delta_t = torch.from_numpy(bias_delta)

        # A layer must keep at least one neuron (whole-layer collapse -- paper
        # Alg. 2 cases 1-2 -- needs layer deletion, which the surgery loop
        # doesn't support). Keeping one stably-inactive neuron is still
        # lossless: it contributes 0 everywhere.
        if len(remove) + len(ctx.already_selected) >= H and remove:
            kept = remove.pop(0)
            logging.info(f"  leo_pp: layer {layer_idx} fully stable; keeping neuron {kept}")
            merges = [op for op in merges if op.removed != kept]

        logging.info(
            f"  leo_pp: layer {layer_idx}: {len(inactive)} stably inactive, "
            f"{len(active)} stably active ({n_folded} folded as linearly dependent) "
            f"[certify={self.certify}]"
        )
        return PruneDecision(remove=remove, merges=merges, bias_delta=bias_delta_t)
