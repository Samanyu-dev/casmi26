# Candidate 1: from LB 0.366 to 0.39+

Surveyed 24 public notebooks (pulled into `candidate-1/pulled/`, flattened code in `pulled/_flat/`), the public LB, and
dataset listings. Short version: **no public notebook scores above 0.366**. There are 15 teams at 0.390 or higher
(top: pikachu 0.440) and all of them are private. Any path to 0.39 therefore stacks changes that no one has scored
publicly. What I propose is the one extra channel that is independent of our pipeline, measured on the LB by others,
and fits the 9 h budget.

Correction to TASK.md: `ref/enveda-0-37-lb-score.ipynb` is **byte-identical to v4g** (`ICE_PC=True`, which injects
PubChem candidates into ICEBERG), not to v4f. So our 0.366 is the v4g score, and v4h = our base + GLACIER.

## 1. Leaderboard map

| Family (refs) | Visible / claimed LB | What differs |
|---|---|---|
| **ahmedberatozer** v3.6 → v4b → v4f → **v4g** (= our base) | 0.358 → 0.354 → 0.362 → **0.366** (header comments; the flexonafft ablation confirms v4g−v4f = +0.004) | v1 engine + FPNet A + fe_v4 families (derivation/frag2/analog_struct/model_views/fragnet) + a 4-booster LGBM, then ICEBERG same-formula re-scoring (λ=0.5), then a gated PubChem A+B channel merged at slots [2..10] or [4..20]. v4g also sends the PubChem list through ICEBERG. |
| ahmedberatozer **v4h** (`casmi26-glacier`) | unscored (running) | v4g + GLACIER third term: z(rank) + 0.5 z(ICE) + 0.5 z(GL). Budget 2400 s. Falls back to v4g on failure. |
| ahmedberatozer `casmi26-ice-ft` (dataset only) | none | fine-tuned ICEBERG `inten.pt`. Its own author never shipped it, and no notebook uses it. |
| dmitriigluzdov from-spectra-to-structures | v17 = 0.366 (v4g repro); v13 = 0.341 (two-ranker + PubChemLite); v14 = 0.326 | Current version is a v4h repro (unscored). Useful because it shows the two-ranker family and the v4 family scored separately. |
| flexonafft confidence-gated-analog-gen | unscored | Now just a v4g ablation that turns off ICE_PC. |
| prvsiyan analog-propagation (heon29, beraterolelk "0.336 SOTA", wyananwyanan channel-1 are clones) | 0.335 peak; family mean **0.329 ± 0.004**; the same code rerun gave 0.328 | Library + mass-shifted analogs + MetFrag-lite + FPNet (single+merged) + HGB ranker. Key measurements: full PubChem expansion **0.335 → 0.205**; pairing the per-spectrum and merged FPNet views **+0.019**; seed noise about 0.006. |
| megayak two-rankers / llccqq624 **W080/W088** (catsfunny copy), raunakdey | 0.341 → 0.342 (8 seeds) | Two HGB rankers: prvsiyan 31-feature FPNet ranker (w=0.88) + megayak 51-feature no-FP simulated ranker (0.12). |
| denpugovkin protected-bio-db-tail | public-28 **0.347** (W088 + curated PubChemLite tail), public-29 0.329 (rejected) | Current version is a diagnostic probe with no submission. |
| haideptry v32 / v33 / v36 (xhhuang, nursrijan, evgendvorkin, uninhibitedscholar replicas) | v32 "0.350+" title; v33/v36 "0.360+/0.365+/0.380+" are *targets*, not scores | W088 + deeper GBM (d8, 160 it) + "dual shield" (protects ranks 1–2) + MMFF isomer bank. That bank is **precomputed on the visible test**, so it does not transfer to the hidden rerun. v35 was wiped (`sys.exit(0)`). |
| senanuretin pool study | train+COCONUT 0.258, +PubChemLite 0.273, +ChEBI/LM 0.268, +PubChem-lit 0.252 | More decoys cost more than the recall they add. |
| matweyisupov v8 gated-hybrid, franciscoangulo V5x–V6x | ≤0.342 | Static public-test caches with hash fallback. The caches never help the hidden rerun. |
| lucifer19 IONWRAITH, dedquoc, cacinie, tamerlan | none / low | Heuristic or baseline pipelines. |

**What the LB says about the test mix** (prvsiyan's decomposition): Class 1 is about 16% and saturated (library-only
scores 0.151). Class 2 is about 27–30%, and the pool already contains nearly all of it (ρ≈1). Class 3 is about 55%.
A perfect ranker over the pool would score about 0.43, so from 0.366 **the remaining gain is almost entirely ranking
of Class 2**. Adding candidates does not help. It also means the teams above 0.43 are getting some Class 3 right.

## 2. Recommended change set: "v4i" = v4h + gated rank fusion with the W088 two-ranker

**Step 0 (no cost).** Wait for the v4h score. Use v4h as the base if it scores ≥ 0.366, otherwise v4g. Everything
below sits on top of that base.

**Step 1: add W088 as an independent second system.** It has different FPNet weights, a different ranker, and a
different feature code. Run it as a subprocess, the same way the PubChem channel is isolated.
- Add these inputs to our fork (all small and public): `prvsiyan/casmi26-fp-models-v2`, `prvsiyan/casmi26-ranker-features`,
  `megayak/casmi26-simulated-ranker-rows`, `prvsiyan/coconut-casmi26-candidates`, `prvsiyan/chebi-lipidmaps-casmi26`,
  `thedevastator/open-source-natural-product-annotations`. RDKit comes from our existing
  `metric/rdkit-2026-3-3-wheel`, so drop `aidensong123/...` and check that the imports resolve.
- New cell 1b, placed after the PubChem channel and before the engine: write the code cells of
  `llccqq624/casmi26-next-direct-w088` (v1, Apache-2.0; its `pv.py`, `pv_fp.py`, `casmi_engine.py` and run cell)
  to `/kaggle/working/w088/`. Run it with
  `subprocess.run([sys.executable, 'run_w088.py'], cwd='/kaggle/working/w088', timeout=150*60)` and read
  `submission.csv` into `W88[mid] = list_of_smiles`. **Any exception or timeout sets `W88 = {}`**, which gives
  byte-identical v4h output.
- Order the GPU work so W088 runs *before* ICE and GL (it frees its memory on exit). Budget: PubChem about 1 h +
  W088 ≤ 2.5 h + engine about 45 min (harness: about 6 s/mol) + ICE 45 min + GL 40 min, about 5.5–6 h. That leaves
  3 h of headroom. If the first run is over 7 h, cut W088 seeds from 4 to 2 (`RANK.SEEDS`).

**Step 2: gated fusion.** New cell 7, after the gated PubChem merge. It applies to `final` for each molecule:
```python
FUSE_W, FUSE_K = 0.5, 5
def key14(s): ...  # tautomer-canonical InChIKey[:14] with RDKit 2026.03.3 (same as scorer); cache it
def fuse(v4, w88):
    if not w88: return v4
    sc = {}
    for lst, w in ((v4, 1.0), (w88[:25], FUSE_W)):
        seen = set()
        for r, s in enumerate(lst):
            k = key14(s)
            if k in seen: continue
            seen.add(k); d = sc.setdefault(k, [0.0, s]); d[0] += w / (FUSE_K + r + 1)
    k1 = key14(v4[0])                      # rank-1 lock: v4h top-1 never moves
    rest = sorted((v for k, v in sc.items() if k != k1), key=lambda t: -t[0])
    return [v4[0]] + [s for _, s in rest][:24]
if lib_max < LIB_TAU: final = fuse(final, W88.get(mid, []))   # class-1 (lib_max >= 0.9) untouched
```
Why these values: with k=5 and w=0.5, a W088-only candidate ranked 1st scores 0.083. That is the same as v4h rank
6–7. It can move up to rank 2 only when both lists agree, and consensus is where the gain from independent views is
largest (prvsiyan's +0.019 came from pairing views). The class-1 gate and the rank-1 lock copy the two protections
that have already held up on the LB: the v3.6 PubChem gate and haideptry's dual shield.

**Submission plan (3 submissions):** (a) v4i with `FUSE_W=0.5`; (b) `FUSE_W=0.35`, only if (a) gains ≥ 0.006 (the
seed noise floor); (c) keep the best of v4h, v4i and v4i-b as the final selection.

## 3. Expected LB, evidence, risks, fallback

- **v4h**: +0.000 to +0.008. ICEBERG-type re-scoring has scored +0.004 (the v4g PubChem-ICE increment). v4b → v4f
  (+0.008) bundles ICE with other changes, so that step alone does not isolate ICE.
- **Fusion**: +0.006 to +0.018. The evidence is indirect but consistent. Combining views and channels built on
  different models is the only change type that has moved this LB repeatedly: +0.019 from paired FPNet views, +0.004
  from the v3.6 PubChem gate, and +0.006 from denpugovkin's gated tail. W088 (0.342) is weaker than v4 but its
  errors are uncorrelated with v4's.
- **Point estimate: 0.375–0.385. My probability of reaching ≥ 0.39 is about 20–25%.** No public evidence supports a
  sure path to 0.39. Everything that has scored above 0.37 is private.

Risks:
1. **Public-LB overfit.** The public LB is about 33% of about 400 molecules, around 130, and seed noise is ±0.006.
   Only two knobs are tuned, and only if (a) clears the noise floor.
2. **Dilution** of Class-2 answers that v4h already ranks at 2–5. The rank-1 lock, w<1 and the class-1 gate bound
   this, and the worst case is small reorders.
3. **Runtime/OOM.** W088 builds a 712k pool with multiprocessing. Limit `workers=2` to protect the 13 GB RAM, run it
   in a subprocess with a timeout, and let the fail-closed path drop back to v4h.
4. **Dependency drift.** W088 expects RDKit 2026.03.3, which our wheel provides. Check the key14 function against
   the scorer on 20 train SMILES in a smoke run.

**Fallback:** v4h (or v4g) unchanged. The fused notebook drops back to it automatically on any W088 failure.

**Before submitting:** run the local `exp/` harness with the fusion on c1/c2 to confirm c1 does not regress (fusion
must not touch c1, and c2 should not drop more than 0.01). The harness is leaky, so use it only as a check that
nothing got worse. It says nothing about gains.

## 4. Alternatives considered and rejected

| Idea | Why rejected |
|---|---|
| Bigger pool (full PubChem, PubChemLite, ChEBI/LM, PubChem-lit) | Measured: −0.130, +0.015/−0.005, −0.021. v4 already has a gated PubChem tier, and ρ≈1 means recall is not the bottleneck. |
| Tune ICE_LAM/GL_LAM, LIB_TAU, SLOTS | Single-knob moves fall inside the ±0.006 noise band. This is how you overfit the public LB. |
| haideptry v33/v36 isomer banks, v8 static caches | Computed on the visible format-example test, so they do not transfer to the hidden rerun. v35 was deleted. |
| `casmi26-ice-ft` (fine-tuned ICEBERG) | Unused by its author, with no score anywhere. A reasonable 4th submission after v4i, not the main bet. |
| Class-3 derivative enumeration | v4 already generates (`EngineCfg(generate=True)`), and starkhushi measured −0.002 on the prvsiyan pipeline. The ceiling is large (55% of test), but no public component moves it. It needs a real de novo model and more than one day. |
| Seed bagging / deeper GBM (raunakdey, haideptry) | +0.001 measured, and deeper GBM did not transfer (prvsiyan d10: −0.014). |
| Swap base to the two-ranker family | 0.342–0.350 < 0.366. It is useful only as a second, independent view, which is what Step 1 does. |
| Train a new ranker or FPNet on timsTOF-only data | The largest possible upside, but it cannot be validated honestly in the time left (local sims are leaky). |
