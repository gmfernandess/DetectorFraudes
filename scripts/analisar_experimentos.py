"""Análise dos experimentos (tabelas do artigo) a partir dos .npz gerados por experimentos_professor.py.

Uso:
  python scripts/analisar_experimentos.py results/experimentos_professor results/analise 200

Gera: resumo.csv (PR-AUC com IC, Brier, threshold de máximo F2, Precision/Recall com IC),
custo_relativo.csv, custo_monetario.csv, calibracao.csv e comparacao_modelos.csv.
"""
import sys, glob, os, re, json
import numpy as np, pandas as pd
from math import sqrt
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

D = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 200
GRID = np.round(np.r_[[0.001, 0.005, 0.01, 0.02, 0.03, 0.04], np.arange(0.05, 0.951, 0.05),
                      [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]], 3)
RATIOS = [1, 10, 100, 1000]
CFP_MONEY = [100, 1_000, 10_000, 100_000]

def counts(y, p, amount=None):
    pos = p[:, None] >= GRID[None, :]
    tp = (pos & (y[:, None] == 1)).sum(0); fp = (pos & (y[:, None] == 0)).sum(0)
    fn = (~pos & (y[:, None] == 1)).sum(0)
    fn_amt = None
    if amount is not None:
        fr = y == 1
        fn_amt = np.array([amount[fr & (p < t)].sum() for t in GRID])
    return tp, fp, fn, fn_amt

def prf(tp, fp, fn):
    P = np.divide(tp, tp + fp, out=np.zeros(len(tp), float), where=(tp + fp) > 0)
    R = tp / (tp + fn)
    F2 = np.divide(5 * P * R, 4 * P + R, out=np.zeros(len(tp), float), where=(4 * P + R) > 0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros(len(tp), float), where=(P + R) > 0)
    return P, R, F1, F2

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, bins - 1)
    e = 0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

i05 = int(np.where(np.isclose(GRID, 0.5))[0][0])
def boot_iter(y):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for b in range(NB):
        r = np.random.default_rng(1000 + b)
        yield np.r_[r.choice(pos, len(pos)), r.choice(neg, len(neg))]
rng = np.random.default_rng(42)
rows, cost_rows, money_rows, calib_rows = [], [], [], []
splits = {}
boot_idx_cache = {}
test_probs = {}
for f in sorted(glob.glob(os.path.join(D, 'probs__*.npz'))):
    _, etapa, split, cols, bal, model = os.path.basename(f)[:-4].split('__')
    if split not in splits:
        splits[split] = np.load(os.path.join(D, f'split__{split}.npz'))
    S = splits[split]; Z = np.load(f)
    yv, yt = S['y_val'].astype(int), S['y_test'].astype(int)
    av, at = S['amount_val'], S['amount_test']
    pv, pt = Z['prob_val'].astype(float), Z['prob_test'].astype(float)
    key = (etapa, split, cols, bal, model); test_probs[key] = pt.astype(np.float32)
    tpv, fpv, fnv, fnav = counts(yv, pv, av)
    tpt, fpt, fnt, fnat = counts(yt, pt, at)
    Pv, Rv, F1v, F2v = prf(tpv, fpv, fnv); Pt, Rt, F1t, F2t = prf(tpt, fpt, fnt)
    j = int(np.argmax(F2v))  # max F2 on validation (ties -> lowest threshold)
    # bootstrap PR-AUC (stratified)
    aps = [average_precision_score(yt[i], pt[i]) for i in boot_iter(yt)] if etapa == 'principal' else [np.nan]
    wP, wR = wilson(tpt[j], tpt[j] + fpt[j]), wilson(tpt[j], tpt[j] + fnt[j])
    rows.append(dict(etapa=etapa, split=split, colunas=cols, balanceamento=bal, modelo=model,
        val_pr_auc=average_precision_score(yv, pv), test_pr_auc=average_precision_score(yt, pt),
        test_pr_auc_lo=np.percentile(aps, 2.5), test_pr_auc_hi=np.percentile(aps, 97.5),
        test_roc_auc=roc_auc_score(yt, pt), test_brier=brier_score_loss(yt, pt), test_ece=ece(yt, pt),
        thr_f2=GRID[j], val_f2=F2v[j], test_P=Pt[j], test_P_lo=wP[0], test_P_hi=wP[1],
        test_R=Rt[j], test_R_lo=wR[0], test_R_hi=wR[1], test_F2=F2t[j], test_TP=tpt[j], test_FP=fpt[j], test_FN=fnt[j],
        test_P05=Pt[i05], test_R05=Rt[i05], test_F2_05=F2t[i05], test_FP05=fpt[i05], test_FN05=fnt[i05],
        thr_f2_at_edge=bool(j == len(GRID) - 1 or j == 0)))
    for k in RATIOS:
        c = 1 / k; jv = int(np.argmin(fnv + c * fpv))
        cs = fnt[jv] + c * fpt[jv]; c5 = fnt[i05] + c * fpt[i05]
        cost_rows.append(dict(etapa=etapa, split=split, colunas=cols, balanceamento=bal, modelo=model, razao=k,
            thr=GRID[jv], test_cost=cs, test_cost_05=c5, reducao=1 - cs / c5))
    for cfp in CFP_MONEY:
        jv = int(np.argmin(fnav + cfp * fpv))
        cs = fnat[jv] + cfp * fpt[jv]; c5 = fnat[i05] + cfp * fpt[i05]
        money_rows.append(dict(etapa=etapa, split=split, colunas=cols, balanceamento=bal, modelo=model, c_fp=cfp,
            thr=GRID[jv], test_cost=cs, test_cost_05=c5, reducao=1 - cs / c5,
            fraude_total=at[yt == 1].sum(), fn_amt=fnat[jv], fp=fpt[jv]))
    # calibration table (10 quantile bins) for test
    q = np.quantile(pt, np.linspace(0, 1, 11)); q[-1] += 1e-9
    idx = np.clip(np.searchsorted(q, pt, side='right') - 1, 0, 9)
    for b in range(10):
        m = idx == b
        calib_rows.append(dict(etapa=etapa, colunas=cols, balanceamento=bal, modelo=model, bin=b,
                               prob_media=pt[m].mean(), freq_obs=yt[m].mean(), n=int(m.sum())))
    print('ok', key, flush=True)

R = pd.DataFrame(rows); R.to_csv(f'{OUT}/resumo.csv', index=False)
pd.DataFrame(cost_rows).to_csv(f'{OUT}/custo_relativo.csv', index=False)
pd.DataFrame(money_rows).to_csv(f'{OUT}/custo_monetario.csv', index=False)
pd.DataFrame(calib_rows).to_csv(f'{OUT}/calibracao.csv', index=False)

# paired comparisons between models on test (same split/cols/bal)
comp = []
for (etapa, split, cols, bal, model), pt in test_probs.items():
    if model != 'xgboost' or etapa != 'principal': continue
    yt = splits[split]['y_test'].astype(int)
    for other in ['random_forest', 'logistic_regression']:
        po = test_probs.get((etapa, split, cols, bal, other))
        if po is None: continue
        d = [average_precision_score(yt[i], pt[i]) - average_precision_score(yt[i], po[i]) for i in boot_iter(yt)]
        comp.append(dict(etapa=etapa, split=split, colunas=cols, balanceamento=bal, comparacao=f'xgboost - {other}',
                         dif_pr_auc=average_precision_score(yt, pt) - average_precision_score(yt, po),
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p_boot=2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))
pd.DataFrame(comp).to_csv(f'{OUT}/comparacao_modelos.csv', index=False)
print('FIM')
