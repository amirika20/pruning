# TODO — road to the benchmark sweep

Written 2026-08-21. Companion to `studies/DESIGN_SPACE.md` (option registry +
evidence ledger E1–E17) and `studies/CLUSTER_HANDOFF.md`. Evidence tags below
refer to that ledger; read it before re-deciding anything it already settled.

**Status in one line:** P0 is clear — every method runs on Conv/BN, the sweep
harness produces accuracy-versus-width curves with first-crossing capacity, and
the baselines exist. What remains is protocol decisions (P2), the paper work
(P3), and housekeeping (P4) before the real run.

**One early measurement worth knowing before writing the cost paragraph:** on a
small MLP the sweep's `plan_seconds` (0.15–0.19s) is much smaller than its
`solve_seconds` (1.2–2.2s), because the repair and realization are per-width no
matter what — only the greedy pass amortizes. MASH's whole-sweep total came out
ABOVE OSSCAR's there (2.41s vs 0.41s). The anytime advantage is real but it
amortizes the dendrogram, not the repair, so the claim needs to be stated that
way and measured at a scale where the O(H^2) pass actually dominates.

---

## P0 — blocks a *meaningful* benchmark

- [x] **DONE (8cb0dc6).** ~~Conv/BN support for MASH.~~ `mash._prepare` raises `NotImplementedError`
      on `Conv2d`, and `data_free_merge` is FC-only, `saturated`'s always-on path
      is Linear-only, HOPE's merge disables itself under BN. So on the paper's
      suite (ResNet-20/56, VGG-16-BN, MobileNetV2, ResNet-50) the only arms that
      can run are `osscar` and `saturated(mode=dead)` — i.e. our method is absent
      from every conv row.
      Needs: im2col patch moments, patch-space box, BN folded into
      `(w_eff, b_eff)` (read access already exists via `prunable_bn`, and
      `saturated`/`mash.extract_units` already fold it).
      Reference implementations: `studies/gram_stability/resnet_phase_a.py`
      (exact BN folding, patch treatment, exact realization back into the block)
      and `phase_d_cnn.py` (`conv_units`, `sample_patches`).
      **Recommendation:** make the conv path `dictionary="medoid"` +
      `repair="empirical"`. It needs no BN write-back (survivors keep their
      original filters and BN slots) and E12/E13 say it is the better arm on
      BN-trained filters anyway. Merged filters on conv can stay unsupported.

- [x] **DONE (this commit).** ~~Width-sweep harness with dendrogram reuse.~~ Capacity at −δ needs an
      accuracy-vs-width curve, and the runner does one prune per invocation.
      Worse, `MASH.select()` rebuilds the dendrogram on every call, so sweeping
      12 widths through the runner would rebuild it 12 times — paying 12× the
      cost *and* destroying the "one pass serves every target width" claim the
      paper leads with. The measurement and the headline would both be wrong.
      Needs: a `plan()`/`apply_at(k)` split on MASH so the trajectory is built
      once and cut many times, plus `src/experiments/sweep.py` to assemble the
      curve. `studies/gram_stability/phase_b.py` and `phase_c2.py` have the
      shape of this (`dendrogram` + `partition_at` + `apply_cuts_*`).

- [x] **DONE (this commit).** ~~First-crossing capacity metric in `src/`.~~ There is no capacity metric
      in the promoted path at all, and the studies version has a known bug:
      `capacity()` takes the MAX passing width rather than the first crossing, so
      a denser grid reads systematically higher — E15 measured 0.772 vs 0.484 on
      one fashion cell from grid density alone. Fix the definition when
      promoting, not after.

## P0 — clear

- [x] **DONE.** ~~Make the greedy pass subquadratic.~~ `MashEngine.step()` cached
      per-row minima instead of rescanning the H x H matrix, so a pass went from
      O(H^3) to O(H^2) -- measured x4.0 per doubling of H instead of x8. Whole-model
      plan time: opt-1.3b 2.0h -> 4.5min, 2.7b 5.2h -> 9.3min, 6.7b 21.1h ->
      23.9min, 13b 51.3h -> 46.6min (27-66x). 6.7b and 13b are re-enabled in
      suite.yaml; 13b still needs an 80GB card for its 48.4 GiB of fp32 weights.
      Equivalence is a permanent self-test: identical merge sequences and costs
      against the brute-force argmin for all three scores.

## P1 — needed before numbers are publishable

- [x] **DONE (d163063).** ~~Random and magnitude baselines.~~ Table 2 has both columns; neither is
      registered. (`silent`/`redundant` were removed and were never these.)
- [ ] **Trained-checkpoint cache.** Determinism makes retraining correct, not
      cheap; for M methods × N widths × S seeds training dominates. Key it on
      `config_digest(model, data, training) + seed` — `src/reproducibility.py`
      already exposes exactly that.
- [x] **DONE.** ~~Confirm every dataset is local.~~ ImageNet-1k is the Kempner
      shared testbed copy at
      `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k`
      (baked into suite.yaml). Everything else downloads; `scripts/warm_caches.py`
      fetches the checkpoints from a login node.
- [ ] **Promote OSSCAR's swap refinement into the conv recipe.** E13: swap closes
      the *residual* gap to OSSCAR completely on CNNs. Currently only reachable
      by composing `OSSCAR._local_search_stage` from
      `studies/gram_stability/phase_d_cnn_hybrid.py`.
- [ ] **Per-layer relative output error in the report.** The studies harness
      measures it; it is the natural quality column beside the per-layer counts,
      and the certified tier needs it to report tightness. Costs one forward pass
      per layer, so make it opt-in.

## P2 — protocol decisions to settle *before* burning compute

- [ ] **Seeds / tolerances / grid.** Studies used 3 seeds and −0.5/−1/−2pt. E9
      warns the 12-point grid (±0.08) is too coarse to quote — pick the final
      resolution now, since it changes the run count.
- [ ] **Cross-layer allocation.** E7: equal fractions ≈ greedy ≈ grid oracle on
      MLPs, so allocation barely matters there — but its own caveat says per-slot
      capacity on ResNet spreads 19–76%, so re-test allocation on conv rather
      than inheriting "equal fractions is enough".
- [ ] **Decide the headline metric set.** Opinion, worth arguing about: tables
      that are capacity-only will mostly show ties, because E14 and E15 together
      say the dictionary *and* the selection criterion stop mattering under
      global repair. What does not converge is (a) the cost of producing a whole
      width sweep — one dendrogram versus OSSCAR re-solving per width — and (b)
      the certified tier, which nothing else offers. Report those alongside
      capacity or the contribution reads thinner than it is.

## P3 — paper work owed (`studies/paper/main.tex`)

- [ ] **Rework §4 per E17.** Make `delta_f` the primary functional criterion and
      demote the exact-damage form. This *reverses* an edit I made earlier in the
      session: I wrote that the fan-out Gram carries information "no pairwise
      distance between responses can carry", and E17 measures its contribution as
      ~nothing (Δ_F reproduces `func_matched`'s error *and* its cluster
      structure, 6 clusters vs 6). The claim as written is empirically false.
- [ ] **Rename loudness → mass** (11 occurrences in `main.tex`, plus
      `studies/DESIGN_SPACE.md`, `CLUSTER_HANDOFF.md`, and the
      `studies/gram_stability/*.py` docstrings). Decided this session.
- [ ] **Rename ward → cylinder** in the paper's prose. Ward is the *linkage*, and
      Δ_F is equally Ward-form; the axis that differs is pre- vs post-ReLU.
- [ ] **Add the ward+global column to Table 2.** E15: domain-only selection with
      data-light repair ties the winner — a legitimate mixed tier.
- [ ] **Frame `Ours_cert` as a certificate, not a capacity result.** E15 measures
      it at 0.33–0.63 against OSSCAR's 0.79. Selling it on capacity invites the
      obvious rejection.
- [ ] **State the pre-ReLU structural limit as a result.** Any weights-only
      criterion is necessarily pre-ReLU, because knowing what the ReLU clips
      requires the measure — which is why `ward_l2` and `ward_whitened` both
      failed to help (E17). This is a strong negative result, not a caveat.
- [ ] **Write Related Work.** Still empty. HOPE (arXiv:2607.21366) belongs there
      as a *concurrent, independent* derivation of our kernel — its eq. 74 is our
      factorization, its capacity is our mass, its eq. 85 is our arc-cosine
      kernel. Srinivas & Babu (BMVC 2015) is the ancestor: same gauge argument,
      keep-survivor dictionary, no calibration of the angular-vs-offset scale.
- [ ] **Decide the fate of the unrun table rows** (OPT-125M…13B, CIFAR-10,
      VGG/MobileNet/ResNet-56). All cells are `\tbd`. Either commit to running
      them or trim; leaving them implies results that do not exist, and E5 argues
      the pretrained rows will be unflattering to the certified tier.
- [ ] **Abstract, Introduction, Discussion** are still `TODO`.

## P4 — housekeeping

- [ ] **Commit the studies Ward work.** `WardL2Merge`/`WardWhitenedMerge`
      (`merge.py`), `DeltaFMerge` (`functional.py`), the `make_engine` wiring,
      `overlap_ward.py`, `why_ward_groups.py`, and the E15–E17 entries in
      `DESIGN_SPACE.md` are all uncommitted — the code behind three evidence
      entries is not in git.
- [ ] **`.gitignore` for LaTeX artifacts** (`*.aux`, `*.fls`, `*.log`, `*.out`,
      `*.synctex.gz`, `*.toc`, `*.fdb_latexmk`) — `studies/method/` and
      `studies/paper/` both leak build noise into `git status`.
- [ ] **`studies/finetune_tier/`** is untracked; decide whether it ships.
- [ ] **Optional refactor:** `saturated.py` imports `extract_units` and the
      rectified-Gaussian Grams from `mash.py`. Cleaner would be
      `src/pruning/geometry.py`. Cosmetic; do it only if the coupling bites.
- [ ] **Wide-layer caveat in the similarity analysis.** `PRc/H` is computed from
      256 calibration inputs, so for layers wider than that the covariance is
      rank-deficient and the effective dimension is capped by the sample, not the
      layer. Raise the input budget before reading it on wide models.

---

## Already done this session (for context)

`hope` ported (a129d82) · MASH promoted and named, mass/cylinder terminology
fixed (518ea57) · `saturated` promoted with the measured-region certificate,
split sample sizes and zero-energy guard (d039b5e) · `silent`/`redundant`
removed, 11 configs migrated (8426a62) · determinism + fingerprints + per-unit
pruning analysis (2fbd884) · before/after geometry battery (aae0340) ·
neuron-similarity matrix analysis (d842d96).
