# ---- EXPERIMENT HARNESS (not a submission) ----
# Held-out timsTOF natural products (enveda-np-examples) run through the v1 engine + v4b ranker under three simulations:
#   c1: only this molecule's enveda-np-examples spectra hidden (other libraries still have it)   -> class 1
#   c2: every spectrum of the structure hidden, structure stays in the pool                    -> class 2
#   c3: c2 + structure dropped from the pool (can only be recovered by generation)              -> class 3
# Caveat: FPNet / ranker / fe_v4 resources were likely trained on these structures, so c2/c3 are optimistic.
import time, pickle
NPLIB, MAXMOL = 9, int(os.environ.get('MAXMOL', 250))
meta = L.meta
ok = (L.lib == NPLIB) & meta.adduct.isin(chem.TEST_ADDUCTS).values & (L.sid >= 0) & (L.n_clean > 0)
rows_all = np.flatnonzero(ok)
sids = np.unique(L.sid[rows_all])[:MAXMOL]
print('val structures', len(sids), 'spectra', len(rows_all), flush=True)

def spectra_of(rows):
    out = []
    for r in rows[:16]:
        a, b = L.off[r], L.off[r + 1]
        ce = float(meta.ce_mean.values[r])
        out.append(dict(mz=L.mz[a:b].astype(np.float64), it=L.it[a:b].astype(np.float64), mode=int(L.mode[r]),
                        adduct=meta.adduct.values[r], prec=float(L.prec[r]), ce=ce if np.isfinite(ce) else np.nan,
                        ce_n=int(max(1, meta.ce_n.values[r]))))
    return out

res, lists = [], {}
t1 = time.time()
for i, sid in enumerate(sids):
    rows = rows_all[L.sid[rows_all] == sid]
    spectra = spectra_of(rows)
    nm = meta.nm.values[rows]; nm = nm[np.isfinite(nm)]
    if not len(nm): continue
    target = float(np.median(nm)); truth = L.struct_key[sid]
    for scen in ('c1', 'c2', 'c3'):
        ex = np.zeros(len(L.sid), bool)
        if scen == 'c1':
            ex[rows] = True; kw = dict(exclude=ex, exclude_sid=int(sid), exclude_lib=NPLIB)
        else:
            ex[L.sid == sid] = True; kw = dict(exclude=ex, exclude_sid=int(sid))
            if scen == 'c3': kw['drop_pid'] = int(P.key2pid.get(truth, -1))
        t2 = time.time()
        try:
            C, X, F, names, info = V.run(spectra, target, **kw)
            keys, scs = [], []
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                seen = set()
                for j in np.argsort(-score, kind='stable'):
                    if C.key[j] in seen: continue
                    seen.add(C.key[j]); keys.append(C.key[j]); scs.append(float(score[j]))
                    if len(keys) >= TOPN: break
            rank = keys.index(truth) + 1 if truth in keys else 0
            res.append(dict(sid=int(sid), scen=scen, rank=rank, rr=(1 / rank if 0 < rank <= 25 else 0.0),
                            in_cands=truth in set(C.key), n_cand=len(C.pid), lib_max=float(info.get('lib_max', 0.0)),
                            secs=time.time() - t2))
            lists[(int(sid), scen)] = (keys, scs, float(info.get('lib_max', 0.0)))
        except Exception as e:
            print('ERROR', sid, scen, repr(e)); res.append(dict(sid=int(sid), scen=scen, rank=0, rr=0.0, err=repr(e)))
    if (i + 1) % 10 == 0:
        R = pd.DataFrame(res)
        print(f'{i+1}/{len(sids)} {time.time()-t1:.0f}s', R.groupby('scen').rr.mean().round(3).to_dict(), flush=True)

R = pd.DataFrame(res)
R.to_csv('/kaggle/working/harness.csv', index=False)
pickle.dump(lists, open('/kaggle/working/harness_lists.pkl', 'wb'))
print('\n==== MRR@25 by simulated class ====')
print(R.groupby('scen').agg(mrr=('rr', 'mean'), top1=('rank', lambda r: (r == 1).mean()),
                            recall25=('rank', lambda r: ((r > 0) & (r <= 25)).mean()),
                            in_cands=('in_cands', 'mean'), secs=('secs', 'mean'), n=('rr', 'size')).round(3))
print('total', f'{time.time()-T0:.0f}s')
