"""CASMI 2026 baseline: precursor-mass filter over train structures + binned cosine library search.

python solution.py        -> writes submission.csv from test.parquet
python solution.py val    -> MRR@25 using enveda-np-examples as queries (their own spectra removed from the library)
Works unchanged in a Kaggle notebook (no rdkit/matchms needed).
"""
import os, re, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from scipy import sparse

D = next(p for p in ['/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra', 'data'] if os.path.exists(p))
PPM, BIN, TOPK, NCOL, K = 10, 0.01, 64, 200_000, 25
P = 1.007276
ADDUCT = {'[M+H]+': P, '[M+NH4]+': 18.033823, '[M-H2O+H]+': P - 18.010565, '[M-2H2O+H]+': P - 36.021129,
          '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -P, '[M-H2O-H]-': -P - 18.010565,
          '[M+CH2O2-H]-': 46.005479 - P, '[M+Cl]-': 34.969402}
EL = {'H': 1.00782503207, 'D': 2.0141017778, 'C': 12.0, 'N': 14.0030740048, 'O': 15.99491461956, 'S': 31.97207100,
      'P': 30.97376163, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473, 'Si': 27.9769265325,
      'B': 11.0093054, 'Se': 79.9165213, 'Na': 22.9897692809, 'K': 38.96370668, 'As': 74.9215965, 'Fe': 55.9349375,
      'Mg': 23.9850417, 'Ca': 39.96259098, 'Al': 26.98153863, 'Li': 7.01600455, 'Cu': 62.9295975, 'Zn': 63.9291422}


def formula_mass(f):
    try:
        return sum(EL[e] * int(n or 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', f))
    except (KeyError, TypeError):
        return np.inf  # unknown element / missing formula -> never a candidate


def read_rows(rows, cols):
    """Take sorted global row indices from train.parquet, streaming so the full peak arrays never sit in RAM."""
    out, start = [], 0
    for b in pq.ParquetFile(f'{D}/train.parquet').iter_batches(batch_size=200_000, columns=cols):
        sel = rows[(rows >= start) & (rows < start + b.num_rows)] - start
        if len(sel):
            out.append(b.take(pa.array(sel)))
        start += b.num_rows
    return pa.Table.from_batches(out)


def to_csr(t, prec, spread=False):
    """Drop precursor region + <1% peaks, keep top-K, sqrt, L2-normalise, bin at 0.01 Da."""
    lens = pc.list_value_length(t['ms2_mzs']).fill_null(0).to_numpy()
    mz = pc.list_flatten(t['ms2_mzs']).to_numpy()
    it = pc.list_flatten(t['ms2_normalized_intensities']).to_numpy()
    row = np.repeat(np.arange(len(lens)), lens)
    k = (mz < prec[row] - 1) & (it >= 0.01) & (mz > 0)
    row, mz, it = row[k], mz[k], it[k]
    o = np.lexsort((-it, row))
    row, mz, it = row[o], mz[o], it[o]
    k = np.arange(len(row)) - np.searchsorted(row, row) < TOPK
    row, mz, it = row[k], mz[k], np.sqrt(it[k])
    col = np.minimum(np.round(mz / BIN).astype(np.int64), NCOL - 1)
    X = sparse.csr_matrix((it, (row, col)), shape=(len(lens), NCOL))
    n = np.sqrt(X.multiply(X).sum(1)).A1
    X = sparse.diags(1 / np.where(n == 0, 1, n)) @ X
    if spread:  # +-1 bin tolerance on the query side only
        X = X @ (sparse.eye(NCOL, k=-1) + sparse.eye(NCOL) + sparse.eye(NCOL, k=1))
    return X.tocsr()


def main(val):
    cols = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode', 'adduct', 'precursor_mz', 'ingest_lib']
    meta = pq.read_table(f'{D}/train.parquet', columns=cols).to_pandas()
    meta['sid'], keys = pd.factorize(meta.inchikey14)
    first = meta.drop_duplicates('sid').sort_values('sid')
    S = pd.DataFrame({'ik14': keys, 'smiles': first.normalized_smiles.values,
                      'mass': [formula_mass(f) for f in first.molecular_formula]})
    S['pop'] = np.bincount(meta.sid, minlength=len(S))
    smi, iks = S.smiles.to_numpy(str), S.ik14.to_numpy(str)
    print(f'{len(meta):,} spectra, {len(S):,} structures', flush=True)

    if val:
        is_q = (meta.ingest_lib == 'enveda-np-examples') & meta.adduct.isin(ADDUCT)
        qrows = np.flatnonzero(is_q)
        q = meta.iloc[qrows].reset_index(drop=True).assign(molecule_id=lambda d: d.inchikey14)
        qpeaks = read_rows(qrows, ['ms2_mzs', 'ms2_normalized_intensities'])
        lib_ok = ~is_q.values
    else:
        qt = pq.read_table(f'{D}/test.parquet')
        q = qt.drop(['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
        qpeaks, lib_ok = qt, np.ones(len(meta), bool)
    q['M'] = q.precursor_mz - q.adduct.map(ADDUCT)
    qmode = q.ionization_mode.to_numpy(str)
    Q = to_csr(qpeaks, q.precursor_mz.values, spread=True)

    # candidate structures per molecule by neutral mass
    o = np.argsort(S.mass.values)
    sm = S.mass.values[o]
    groups = q.groupby('molecule_id').indices
    cands = {}
    for mid, qi in groups.items():
        M = np.median(q.M.values[qi])
        lo, hi = np.searchsorted(sm, [M * (1 - PPM / 1e6), M * (1 + PPM / 1e6)])
        if lo == hi:  # ponytail: nothing in window -> nearest masses; PubChem/COCONUT + de novo is the real fix
            lo, hi = max(0, lo - K // 2), lo + K // 2
        cands[mid] = o[lo:hi]

    # library = train spectra of any candidate structure, grouped by sid
    need = np.zeros(len(S), bool)
    need[np.concatenate(list(cands.values()))] = True
    lrows = np.flatnonzero(need[meta.sid.values] & lib_ok)
    lrows = lrows[np.argsort(meta.sid.values[lrows], kind='stable')]
    lsid, lmode = meta.sid.values[lrows], meta.ionization_mode.to_numpy(str)[lrows]
    srt = np.argsort(lrows)  # read_rows returns rows in file order
    L = to_csr(read_rows(lrows[srt], ['ms2_mzs', 'ms2_normalized_intensities']),
               meta.precursor_mz.values[lrows[srt]])[np.argsort(srt)]
    print(f'library: {len(lrows):,} spectra', flush=True)

    out, rr, inwin = [], [], []
    for mid, qi in groups.items():
        c = cands[mid]
        a, b = np.searchsorted(lsid, c, 'left'), np.searchsorted(lsid, c, 'right')
        rows = np.concatenate([np.arange(x, y) for x, y in zip(a, b)] + [np.array([], int)])
        score = np.zeros(len(c))
        if len(rows):
            sim = (Q[qi] @ L[rows].T).toarray()
            sim[qmode[qi][:, None] != lmode[rows][None, :]] = 0
            best = pd.DataFrame(sim.T).groupby(np.repeat(np.arange(len(c)), b - a)).max()
            score[best.index] = best.mean(axis=1).values  # mean over the molecule's spectra of best match
        top = c[np.lexsort((-S['pop'].values[c], -score))][:K]
        out.append((mid, ';'.join(smi[top])))
        if val:
            hit = np.flatnonzero(iks[top] == mid)
            rr.append(1 / (hit[0] + 1) if len(hit) else 0)
            inwin.append(mid in set(iks[c]))

    if val:
        print(f'val molecules {len(rr)}  MRR@25 {np.mean(rr):.4f}  top1 {np.mean(np.array(rr) == 1):.3f}  '
              f'truth in mass window {np.mean(inwin):.3f}')
    else:
        pd.DataFrame(out, columns=['molecule_id', 'smiles']).to_csv('submission.csv', index=False)
        print('wrote submission.csv', len(out))


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'val')
