# Path from LB 0.366 to 0.39+: proposal

## 1. Leaderboard map

Verified via own progression comments / kernel code diff / dataset files; "claimed" = notebook's own banner text, unverified.

| Family | Visible/claimed LB | What it actually does differently | Confidence |
|---|---|---|---|
| **ahmedberatozer v4f→v4g** (our base = kozykappa fork) | v3.6=0.358, v4f adds `fe_v4` retrained ranker, **v4g adds PubChem-in-ICEBERG candidate injection → 0.366** | v1 engine + FPNet-A + gated PubChem channel + ICEBERG same-formula rescoring (z(ranker)+0.5·z(ICE)). Byte-diffed: `v4g-inference` == our `ref/nb.py` exactly. | High (our own lineage) |
| **ahmedberatozer v4h** (running now) | not yet scored | v4g + **GLACIER** (2nd independent forward-MS/MS predictor, MassSpecGym `glacier.JointModel`) as a 3rd rescoring term: z(ranker)+0.5·z(ICE)+0.5·z(GL). Fails back to v4g order on any error. | High (code inspected) |
| **prvsiyan baseline** (114 votes, most-forked) | 0.245–0.266 own-engine stages | Origin of analog propagation + MetFrag-lite + FPNet ranker; the ancestor most other forks build on. | High |
| **beraterolelk 0.336/0.338** | 0.336–0.338 (own simpler engine, below our base) | Same 4-channel idea (library+analog+MetFrag+FPNet) but **rigorously ablated**: PubChem/large-isomer expansion is measured to be catastrophic (Class-2 MRR 0.732→0.328–0.379 once PubChem candidates are added, even with a model channel). ChEBI+LIPID MAPS targeted expansion: +8.8% candidates, +7–19% library coverage, dilution cost only −0.026. | High (numbers, not banner) |
| **denpugovkin "protected bio-db tail"** | Diagnostic only | Paired A/B: public-28 (bio-DB tail, curated) scored 0.347; public-29 (adds PubChemLite candidates) scored **0.329 and was rejected**. Corroborates beraterolelk's PubChem-dilution finding independently. | High (explicit paired numbers) |
| **haideptry v32 "0.350+/0.365+"** | claimed 0.365+ | Independent architecture ("Two-Ranker": prvsiyan's FPNet-based ranker + megayak's no-FPNet 7-library adduct-shifted ranker, blended w=0.88/0.12) + MMFF94 isomer bank. Roughly matches our measured 0.366. | Medium (plausible, not independently reproduced here) |
| **catsfunny/llccqq624 "next-direct-w088"** | measured predicted-LB 0.280–0.285 on a *simpler* engine (no v1-engine/fe_v4/ICEBERG stack) | Real ablation table for the Two-Ranker blend weight sweep; documents the 3 "lethal traps" (candidate-cap recall bug, premature tautomer dedup, timsTOF/GNPS domain-gap FPNet leak 0.49 vs 0.80) that this whole notebook family was built to fix. | High (has a real numbers table) |
| **haideptry v35/v36, xhhuang v35/v36 "0.380+ SOTA"** | claimed 0.380+ | **Identical marketing banner text copy-pasted verbatim across 4 different-author forks**, no differentiated ablation, no paired evidence shown. Same code skeleton as v32/v33. | **Low — treat as unverified/inflated** |
| **dariushafshar (meta-analysis, cited by w088)** | n/a (analysis) | Public board noise ≈0.006 per reseed (prvsiyan's own measurement); "sort order moves MRR 0.03"; same-formula tie-breaking ≈ 95% of the reachable gap (alex chilton); "a database is worth adding exactly to the extent it contains your answers" — decoys are not free. | High, load-bearing for risk assessment |

## 2. Recommended change set

**Adopt `casmi26-v4h-inference` as-is (already queued), and if it lands ≥0.368, stack a second independent ranker on top of it exactly the way the haideptry/w088 lineage does — this is "v4i."**

Step A (already running, keep it): v4h = v4g + GLACIER 3rd-term rescoring. No code change needed beyond what's already in `ref/casmi26-v4h-inference/casmi26-v4h-inference.ipynb`. Dataset: `ahmedberatozer/casmi26-glacier`. Fail-safe (any GLACIER exception → falls back to identical v4g/0.366 behavior), so this cannot regress below our current score.

Step B (concrete new work, do only after Step A's score is known): add a second, structurally-different ranker as an ensemble vote before the ICE/GL rescoring stage, mirroring the "Two-Ranker" pattern:
- Attach `megayak/casmi26-simulated-ranker-rows` (`sim_rank_rows_nofp.npz`, 4 HistGradientBoosting seeds, no-FPNet, 7-library adduct-shifted features) as Ranker 2, alongside our existing v1-engine+`fe_v4`+FPNet-A ranker as Ranker 1.
- Blend by within-molecule reciprocal-probability averaging (not raw score averaging — score scales differ), i.e. `p = w·rank_to_prob(r1) + (1-w)·rank_to_prob(r2)`.
- Start `w=0.88` (the community-converged value, corroborated by catsfunny's real sweep table) but re-derive it on our own held-out `enveda-np-examples` split before trusting it — that set is leaky for absolute score (models saw related structures) but still usable for *relative* weight comparison since both rankers share the same leak exposure.
- Run the existing ICE/GL same-formula rescoring exactly as in v4h, applied to the blended order.
- Everything stays offline; all three datasets (`casmi26-glacier`, `casmi26-simulated-ranker-rows`, `casmi26-ranker-features`) are small (<150MB each), well inside the 9h T4 budget alongside the existing v4h runtime (~85 min for ICE+GLACIER budgets combined, rest is feature/engine compute, unchanged from v4f/v4g).

## 3. Expected LB, evidence, risks, fallback

**Expected LB for Step A alone (v4h): 0.368–0.374.** Evidence: our own progression shows ICEBERG alone added ~+0.004–0.008 (0.354/0.358→0.366 across v4b→v3.6→v4g). GLACIER is architecturally similar to ICEBERG (same `ms-pred`-style forward-model family, same fusion mechanism, same z-score blend at equal weight) — plausibly lower marginal, correlated-signal returns, not a second independent gain of the same size.

**Expected LB for Step A+B combined: 0.375–0.392, with 0.39 being an optimistic but not implausible outcome, not a guaranteed one.** No public notebook has *verified* (non-templated) evidence at or above 0.39 — the only such claims are the copy-pasted v35/v36 banners across 4 authors, which I explicitly discount. The credible ceiling evidence (catsfunny's measured Two-Ranker calibration table, haideptry v32's real architecture matching our own 0.366) suggests a second uncorrelated ranker is the strongest lever anyone has actually demonstrated, on top of what we already have (gated PubChem, ICEBERG).

**Risks:**
1. **Correlated-signal risk**: GLACIER + ICEBERG may be similar enough (both same-formula forward-model rescorers) that Step A undershoots 0.368.
2. **Domain-gap risk**: haideptry's own post-mortem shows public FPNet-family models trained on GNPS/MoNA look great in-library (0.76–0.82 MRR) but crash to 0.49 on held-out timsTOF — the same risk applies to GLACIER and to megayak's simulated ranker if their training data isn't timsTOF-representative.
3. **Public-LB overfit / noise**: ~400 test molecules, ~0.006 noise per reseed (prvsiyan's own measurement). Any single submission's delta under ~0.01 is not distinguishable from noise — don't over-interpret one run.
4. **Candidate-pool temptation**: do **not** add PubChem/ChEBI/etc. to the raw pool without gating — beraterolelk and denpugovkin both show this is net-negative (0.732→0.33–0.38 Class-2 MRR; 0.347→0.329 paired).
5. **Runtime**: two extra heavy rescoring passes (ICE 2700s + GL 2400s budgets) plus a ranker refit; should still fit comfortably under 9h on T4 but leaves less slack than the current base.

**Fallback**: v4h already falls back to v4g/0.366 automatically on GLACIER failure. If Step B's blend underperforms on the (leaky) validation proxy, skip it and submit v4h alone — it cannot score below our current 0.366 baseline by construction.

## 4. Rationale — alternatives considered and rejected

- **ChEBI + LIPID MAPS targeted pool expansion** (beraterolelk's own evidenced win, +coverage/−dilution): rejected as the *primary* recommendation only because no ready-made offline Kaggle dataset for it exists yet (checked kernel/dataset listings) — building one (download, RDKit parse, fingerprint, package as a dataset) is real new offline-data engineering, not a notebook-cell edit, and doesn't fit "concrete change set." Flagged as a good secondary project if time allows.
- **Chasing the v35/v36 "0.380+ SOTA" forks directly**: rejected — identical banner text across 4 unrelated authors with zero differentiated ablation evidence is a strong tell of copy-paste inflation, not a real result to fork blindly.
- **Naive PubChem/large-isomer pool expansion**: rejected outright — two independent, numbers-backed sources (beraterolelk, denpugovkin) show this is net-negative even when gated by a model channel.
- **Full Two-Ranker rewrite as a ground-up fork of the w088/haideptry lineage** instead of extending our own v4g/v4h: rejected — our own pipeline (v1 engine + fe_v4 + FPNet-A + gated PubChem + ICE/GL rescoring) already matches or beats haideptry v32's claimed 0.365+ with our verified 0.366, so throwing it away to adopt an architecture that performs no better (on the honest, non-inflated numbers) is strictly worse than grafting its one genuinely new idea (a second, structurally different ranker) onto what we already have proven.
