# CASMI 2026 (Kaggle: enveda-CASMI26-molecule-id-mass-spectra)

| Entry | Public LB |
|---|---|
| `solution.py` — own baseline: 10 ppm mass filter + binned cosine library search | 0.140 |
| Private fork of the public `v4f` pipeline (kozykappa/enveda-0-37-lb-score) | 0.366 |

- `solution.py` — baseline; `python solution.py val` scores MRR@25 on enveda-np-examples held out.
- `exp/` — private Kaggle experiment notebook (`casmi-harness`) simulating classes 1/2/3 on the v4f engine; `exp/out/harness.csv` = results
  (c1 0.954, c2 0.973, c3 0.361 — c2/c3 leaky: public models were trained on those structures).
- `runs/` — Kaggle notebook bundles pushed with `kaggle kernels push -p runs/<name>`.
- `arena/` — parallel research proposals for reaching 0.39+.

Setup: `python3.12 -m venv .venv && . .venv/bin/activate && pip install kaggle pandas pyarrow scipy`, token in `~/.kaggle/access_token`,
then `kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data && unzip data/*.zip -d data`.
