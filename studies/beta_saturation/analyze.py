"""Hypothesis tests + plots over the per-neuron records from collect.py.

Original hypothesis: neurons that are always active (act_freq = 1) or never
active (act_freq = 0) on the training set have the largest |beta| = |b|/||w||.

Deep-layer follow-up: |beta| is distance from the ORIGIN, but layer 1+ inputs
are post-ReLU and live in the nonnegative orthant, far from the origin. So we
compare a ladder of candidate saturation scores:

    abs_beta   |b| / ||w||           data-free, origin-centric (the hypothesis)
    ibp_score  interval-bound margin data-free given only the input box
    dist_mean  |w.mu + b| / ||w||    needs one statistic: the mean input mu
    z_margin   |E z| / std(z)        needs pre-activation moments (upper bound
                                     on what any such score can achieve)
    abs_b, w_norm                    naive baselines

Per layer (beta scales differ across layers, so layers are never pooled):
  - AUROC of each score separating saturated from mixed neurons
    (1.0 = score ranks every saturated neuron above every mixed one)
  - precision@k for abs_beta (k = #saturated): the literal "highest |beta|" claim
  - Spearman rho between saturation |2p - 1| and abs_beta
  - Mann-Whitney U p-value (saturated > mixed by abs_beta)
  - IBP certificate counts vs actual saturated counts (data-free lower bound)
  - a per-category profile of median features: what IS special about
    dead / always-on neurons in each layer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CATEGORY_COLORS = {"never": "#d62728", "always": "#1f77b4", "mixed": "#b0b0b0"}

# Candidate saturation scores, ordered data-free -> data-dependent.
CANDIDATE_SCORES = ["abs_beta", "ibp_score", "dist_mean", "z_margin", "abs_b", "w_norm"]
# Features profiled per category to see what saturated neurons look like.
PROFILE_FEATURES = ["act_freq", "abs_beta", "dist_mean", "z_margin", "cos_w_mu",
                    "frac_pos_w", "w_norm", "b"]


def categorize(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Adds `category` (never / always / mixed), `saturation` = |2p - 1| and
    the derived `abs_b` column. eps = 0 means strictly never/always."""
    df = df.copy()
    df["saturation"] = (2 * df["act_freq"] - 1).abs()
    df["abs_b"] = df["b"].abs()
    df["category"] = "mixed"
    df.loc[df["act_freq"] <= eps, "category"] = "never"
    df.loc[df["act_freq"] >= 1 - eps, "category"] = "always"
    return df


def _auroc(pos: pd.Series, neg: pd.Series) -> float | None:
    """P(score of a random saturated neuron > score of a random mixed one)."""
    pos, neg = pos.dropna(), neg.dropna()
    if not len(pos) or not len(neg):
        return None
    u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (len(pos) * len(neg)))


def layer_stats(layer_df: pd.DataFrame, eps: float) -> dict:
    """All hypothesis metrics for one layer (records may span seeds)."""
    valid = layer_df.dropna(subset=["abs_beta"])
    saturated = valid[valid["category"] != "mixed"]
    mixed = valid[valid["category"] == "mixed"]

    out = {
        "n_neurons": int(len(layer_df)),
        "n_nan_beta": int(len(layer_df) - len(valid)),
        "n_never": int((valid["category"] == "never").sum()),
        "n_always": int((valid["category"] == "always").sum()),
        "n_mixed": int(len(mixed)),
        "median_abs_beta_saturated": float(saturated["abs_beta"].median()) if len(saturated) else None,
        "median_abs_beta_mixed": float(mixed["abs_beta"].median()) if len(mixed) else None,
        "spearman_saturation_vs_abs_beta": None,
        "spearman_pvalue": None,
        "auroc_abs_beta_detects_saturated": None,
        "mannwhitney_pvalue": None,
        "precision_at_k": None,
        # AUROC of every candidate score for the same saturated-vs-mixed task.
        "score_auroc": {s: _auroc(saturated[s], mixed[s]) for s in CANDIDATE_SCORES
                        if s in valid.columns},
        # Data-free certificates vs reality.
        "ibp": _ibp_stats(valid),
        # Median feature values per category: the "what's special" profile.
        "profile": {
            cat: {f: float(g[f].median()) for f in PROFILE_FEATURES if f in g.columns}
            for cat, g in valid.groupby("category")
        },
        "profile_n": {cat: int(len(g)) for cat, g in valid.groupby("category")},
    }
    out["auroc_abs_beta_detects_saturated"] = out["score_auroc"].get("abs_beta")

    if valid["saturation"].nunique() > 1 and valid["abs_beta"].nunique() > 1:
        rho, p = stats.spearmanr(valid["saturation"], valid["abs_beta"])
        out["spearman_saturation_vs_abs_beta"] = float(rho)
        out["spearman_pvalue"] = float(p)

    if len(saturated) and len(mixed):
        _, p = stats.mannwhitneyu(saturated["abs_beta"], mixed["abs_beta"], alternative="greater")
        out["mannwhitney_pvalue"] = float(p)
        k = len(saturated)
        top_k = valid.nlargest(k, "abs_beta")
        out["precision_at_k"] = float((top_k["category"] != "mixed").mean())

    return out


def _ibp_stats(valid: pd.DataFrame) -> dict | None:
    """How many neurons the data-free interval bounds certify, and whether the
    certificates are sound (a certified neuron must really be saturated)."""
    if "ibp_dead" not in valid.columns or valid["ibp_dead"].isna().all():
        return None
    dead_cert = valid["ibp_dead"].fillna(False).astype(bool)
    always_cert = valid["ibp_always"].fillna(False).astype(bool)
    n_cert = int((dead_cert | always_cert).sum())
    correct = int(((dead_cert & (valid["category"] == "never"))
                   | (always_cert & (valid["category"] == "always"))).sum())
    return {
        "n_certified_dead": int(dead_cert.sum()),
        "n_certified_always": int(always_cert.sum()),
        "n_certified": n_cert,
        "certificate_precision": (correct / n_cert) if n_cert else None,
    }


def analyze(df: pd.DataFrame, eps: float) -> dict:
    df = categorize(df, eps)
    return {
        "eps": eps,
        "layers": {int(l): layer_stats(g, eps) for l, g in df.groupby("layer")},
    }


# ── plots ────────────────────────────────────────────────────────────────────

def _layer_grid(n_layers: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = min(n_layers, 3)
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    return fig, axes.ravel()


def _finish(fig, axes, n_used: int, title: str, save_path: Path) -> None:
    for ax in axes[n_used:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, save_path: Path, title: str = "",
                 score: str = "abs_beta", ylabel: str = r"$|\beta| = |b| / \|w\|$") -> None:
    """act_freq vs a score per layer; saturated neurons colored."""
    layers = sorted(df["layer"].unique())
    fig, axes = _layer_grid(len(layers))
    for ax, l in zip(axes, layers):
        g = df[df["layer"] == l].dropna(subset=[score])
        for cat, color in CATEGORY_COLORS.items():
            sub = g[g["category"] == cat]
            ax.scatter(sub["act_freq"], sub[score], s=14, alpha=0.6,
                       color=color, label=f"{cat} (n={len(sub)})")
        ax.set_yscale("log")
        ax.set_xlabel("activation frequency p")
        ax.set_ylabel(ylabel)
        ax.set_title(f"layer {l}")
        ax.legend(fontsize=8)
    _finish(fig, axes, len(layers), title or f"Activation frequency vs {score}", save_path)


def plot_rank(df: pd.DataFrame, save_path: Path, title: str = "") -> None:
    """Neurons sorted by |beta| (descending); saturated ones marked. If the
    hypothesis holds, the colored points cluster on the left."""
    layers = sorted(df["layer"].unique())
    fig, axes = _layer_grid(len(layers))
    for ax, l in zip(axes, layers):
        g = df[df["layer"] == l].dropna(subset=["abs_beta"]).sort_values("abs_beta", ascending=False)
        ranks = np.arange(len(g))
        for cat, color in CATEGORY_COLORS.items():
            mask = (g["category"] == cat).to_numpy()
            ax.scatter(ranks[mask], g["abs_beta"].to_numpy()[mask], s=14, alpha=0.7,
                       color=color, label=cat)
        ax.set_yscale("log")
        ax.set_xlabel(r"rank by $|\beta|$ (0 = largest)")
        ax.set_ylabel(r"$|\beta|$")
        ax.set_title(f"layer {l}")
        ax.legend(fontsize=8)
    _finish(fig, axes, len(layers), title or r"Are saturated neurons the top-$|\beta|$ ones?", save_path)


def plot_group_box(df: pd.DataFrame, save_path: Path, title: str = "") -> None:
    """|beta| distribution: never / always / mixed, per layer."""
    layers = sorted(df["layer"].unique())
    fig, axes = _layer_grid(len(layers))
    order = ["never", "always", "mixed"]
    for ax, l in zip(axes, layers):
        g = df[df["layer"] == l].dropna(subset=["abs_beta"])
        data, labels = [], []
        for cat in order:
            vals = g.loc[g["category"] == cat, "abs_beta"]
            if len(vals):
                data.append(vals)
                labels.append(f"{cat}\n(n={len(vals)})")
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, lbl in zip(bp["boxes"], labels):
                patch.set_facecolor(CATEGORY_COLORS[lbl.split("\n")[0]])
                patch.set_alpha(0.5)
        ax.set_yscale("log")
        ax.set_ylabel(r"$|\beta|$")
        ax.set_title(f"layer {l}")
    _finish(fig, axes, len(layers), title or r"$|\beta|$ by activation category", save_path)


def plot_score_auroc(pooled: dict, save_path: Path, title: str = "") -> None:
    """Grouped bars: AUROC of each candidate score, per layer."""
    layers = sorted(pooled["layers"])
    scores = CANDIDATE_SCORES
    fig, ax = plt.subplots(figsize=(1.8 * len(layers) * len(scores) / 6 + 4, 4.5))
    width = 0.8 / len(scores)
    cmap = plt.get_cmap("viridis")
    for si, score in enumerate(scores):
        vals = [pooled["layers"][l]["score_auroc"].get(score) for l in layers]
        xs = [li + si * width for li in range(len(layers))]
        ax.bar(xs, [v if v is not None else np.nan for v in vals], width=width,
               label=score, color=cmap(si / max(len(scores) - 1, 1)))
    ax.axhline(0.5, color="k", lw=1, ls="--", alpha=0.6)
    ax.set_xticks([li + 0.4 - width / 2 for li in range(len(layers))])
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("AUROC (saturated vs mixed)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title(title or "Which score finds saturated neurons, per layer?")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_deep_profile(df: pd.DataFrame, save_path: Path, title: str = "") -> None:
    """cos(w, mu) vs z_margin per layer -- the deep-layer 'what is special'
    picture: dead neurons should sit at cos < 0, always-on at cos > 0."""
    layers = sorted(df["layer"].unique())
    fig, axes = _layer_grid(len(layers))
    for ax, l in zip(axes, layers):
        g = df[df["layer"] == l].dropna(subset=["cos_w_mu", "z_margin"])
        for cat, color in CATEGORY_COLORS.items():
            sub = g[g["category"] == cat]
            ax.scatter(sub["cos_w_mu"], sub["z_margin"], s=14, alpha=0.6,
                       color=color, label=f"{cat} (n={len(sub)})")
        ax.axvline(0, color="k", lw=1, alpha=0.4)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\cos(w, \mu)$  (alignment with mean input)")
        ax.set_ylabel(r"$|E[z]| / \mathrm{std}(z)$")
        ax.set_title(f"layer {l}")
        ax.legend(fontsize=8)
    _finish(fig, axes, len(layers),
            title or "Deep-neuron profile: alignment vs standardized margin", save_path)


# ── report ───────────────────────────────────────────────────────────────────

def _fmt(v, spec):
    return format(v, spec) if v is not None and not (isinstance(v, float) and np.isnan(v)) \
        else "-".rjust(len(format(0, spec)))


def format_report(name: str, per_seed: dict[int, dict], pooled: dict) -> str:
    """Readable summary: original-hypothesis table, candidate-score AUROC
    ladder, IBP certificates, per-category profile, per-seed stability."""
    lines = [f"beta-saturation study: {name}", "=" * 78, ""]
    lines.append(f"eps = {pooled['eps']} (act_freq <= eps -> 'never', >= 1-eps -> 'always')")
    layers = sorted(pooled["layers"])

    lines += ["", "[1] Original hypothesis -- |beta| = |b|/||w|| (pooled over seeds):"]
    header = (f"{'layer':>5} {'n':>6} {'never':>6} {'always':>6} {'mixed':>6} "
              f"{'med|B|sat':>10} {'med|B|mix':>10} {'AUROC':>7} {'prec@k':>7} "
              f"{'spearman':>9} {'p(MWU)':>9}")
    lines += [header, "-" * len(header)]
    for l in layers:
        s = pooled["layers"][l]
        lines.append(
            f"{l:>5} {s['n_neurons']:>6} {s['n_never']:>6} {s['n_always']:>6} {s['n_mixed']:>6} "
            f"{_fmt(s['median_abs_beta_saturated'], '>10.4f')} {_fmt(s['median_abs_beta_mixed'], '>10.4f')} "
            f"{_fmt(s['auroc_abs_beta_detects_saturated'], '>7.3f')} {_fmt(s['precision_at_k'], '>7.3f')} "
            f"{_fmt(s['spearman_saturation_vs_abs_beta'], '>9.3f')} {_fmt(s['mannwhitney_pvalue'], '>9.2e')}"
        )

    lines += ["", "[2] Candidate scores, AUROC per layer (data-free -> data-dependent):"]
    header = f"{'layer':>5} {'n_sat':>6} " + " ".join(f"{s:>10}" for s in CANDIDATE_SCORES)
    lines += [header, "-" * len(header)]
    for l in layers:
        s = pooled["layers"][l]
        n_sat = s["n_never"] + s["n_always"]
        row = f"{l:>5} {n_sat:>6} " + " ".join(
            _fmt(s["score_auroc"].get(sc), ">10.3f") for sc in CANDIDATE_SCORES)
        lines.append(row)

    if any(pooled["layers"][l].get("ibp") for l in layers):
        lines += ["", "[3] Data-free IBP certificates (provably dead / always-on over the input box):"]
        header = (f"{'layer':>5} {'cert_dead':>10} {'cert_always':>12} "
                  f"{'actual_never':>13} {'actual_always':>14} {'precision':>10}")
        lines += [header, "-" * len(header)]
        for l in layers:
            s = pooled["layers"][l]
            ibp = s.get("ibp")
            if ibp is None:
                continue
            lines.append(
                f"{l:>5} {ibp['n_certified_dead']:>10} {ibp['n_certified_always']:>12} "
                f"{s['n_never']:>13} {s['n_always']:>14} "
                f"{_fmt(ibp['certificate_precision'], '>10.3f')}"
            )

    lines += ["", "[4] What is special about saturated neurons (median per category):"]
    prof_feats = ["abs_beta", "dist_mean", "z_margin", "cos_w_mu", "frac_pos_w", "w_norm", "b"]
    header = (f"{'layer':>5} {'category':>9} {'n':>6} "
              + " ".join(f"{f:>10}" for f in prof_feats))
    lines += [header, "-" * len(header)]
    for l in layers:
        s = pooled["layers"][l]
        for cat in ["never", "always", "mixed"]:
            if cat not in s["profile"]:
                continue
            p = s["profile"][cat]
            lines.append(
                f"{l:>5} {cat:>9} {s['profile_n'][cat]:>6} "
                + " ".join(_fmt(p.get(f), ">10.4f") for f in prof_feats)
            )

    lines += ["", "[5] abs_beta AUROC per seed (stability check):"]
    for seed, res in sorted(per_seed.items()):
        vals = ", ".join(
            f"L{l}: {_fmt(r['auroc_abs_beta_detects_saturated'], '.3f')}"
            for l, r in sorted(res["layers"].items())
        )
        lines.append(f"  seed {seed}: {vals}")

    lines += ["",
              "Reading guide: [1] tests the original |beta| hypothesis (AUROC/prec@k near 1",
              "= supported; ~0.5 = no signal; < 0.5 = inverted). [2] compares fixes for deep",
              "layers -- dist_mean re-centers beta on the layer-input mean, z_margin is the",
              "statistical upper bound. [3] counts provable saturations. [4] profiles the",
              "saturated neurons: dead ones should show cos(w,mu) < 0 / negative b,",
              "always-on ones cos(w,mu) > 0 / positive b."]
    return "\n".join(lines) + "\n"
