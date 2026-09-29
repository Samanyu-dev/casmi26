# Arena synthesis (2026-09-29)

Candidates: 1 = Opus (arena/candidate-1/proposal.md), 2 = Sonnet (arena/candidate-2/proposal.md),
3 = Fable (dropped: monthly spend limit hit before any output).

**Convergence.** Both surviving candidates independently recommend the same shape:
v4h (our 0.366 v4g base + GLACIER re-scoring) + an independent second system, the W088 two-ranker
(prvsiyan FPNet ranker + megayak no-FP simulated ranker), blended within molecule.
Both also reject pool expansion (measured large drops) and the "0.380+" haideptry/xhhuang banners (targets copied
across forks, with isomer banks built on the visible test). Per arena rules, when candidates agree we ship the
shared plan, so no separate cross-judge was run.

**Base picked:** candidate 1. It gives concrete cell edits, a runtime budget, a fail-closed subprocess and gating.
**Graft from candidate 2:** the calibration that seed noise is about 0.006. It sets the bar for keeping a tuning step.

**Built:** runs/v4i/casmi-v4i.ipynb = v4h + cell 2 (W088 in a subprocess, 150 min timeout; any failure gives W88={},
which reproduces v4h exactly) + a final cell that fuses the two lists. The fusion uses weighted reciprocal rank (k=5,
W088 weight 0.5), keys on the scorer's InChIKey14, leaves molecules with lib_max >= 0.9 untouched, and locks
v4h's rank-1. Fix applied: W088's `find("fp_bits.npy")` would have picked up casmi26-v2-pool's file, so it now
reads the copy that sits next to coco_fp.npy.

**Expectation:** 0.375–0.385; chance of 0.39+ is about 20–25% (no public notebook has a verified score above 0.366).
**Verification:** private Kaggle runs of v4h and v4i, then submit v4i. Tune FUSE_W only if v4i beats v4h by >= 0.006.
