import os as _os
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
SERGIO_DIR = _os.path.join(WORK, 'SERGIO')
"""
Test de l'hypothese EIV (erreur sur la variable explicative) sur les donnees SERGIO.

Hypothese H1 : la sous-estimation de n pour les activations vient de l'imprecision de x = U_regulateur observe
(la cible integre le regulateur sur ~1/lambda ; x instantane est une mesure bruitee du regulateur "effectif").
Prediction : debruiter x (sans utiliser la cible ni le regulateur pour definir les voisins) fait remonter n_hat vers 2
pour les activations, sans sur-corriger les repressions, et sans gain equivalent sur des paires non liees.

Pre-enregistre avant execution :
  - toutes les cibles a regulateur unique (pas seulement les 14 premieres) ;
  - type de regulation connu (on isole l'estimation des parametres) ;
  - succes de H1 : mediane de n_hat des activations dans [1.6, 2.4] apres correction ET mediane des repressions
    restant dans [1.6, 2.4] ET gain de R2 des paires reelles > gain des paires de controle.
"""
import json, sys, warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, WORK)
from validate import load_truth, LAM, f_act, f_rep, true_curve

warnings.filterwarnings("ignore")
OUT = sys.argv[1] if len(sys.argv) > 1 else WORK + '/result_eiv_sergio.json'
npz = np.load(WORK + '/sergio_clean.npz'); U = npz['U'].astype(float)
G, N = U.shape; BIN = np.repeat(np.arange(6), 300)
KS = [5, 15, 50, 150]
truth = load_truth()
rng = np.random.default_rng(0)


def pca_embed(exclude, ncomp=10):
    keep = [g for g in range(G) if g not in exclude]
    X = U[keep].T
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return u[:, :ncomp] * s[:ncomp]


def fit_hill(x, y, typ):
    f = f_act if typ == 'act' else f_rep
    xm, ym = max(x.mean(), 1e-6), max(np.abs(y).max(), 1e-6)
    best = None
    for n0 in (1.0, 2.0, 3.0):                      # 3 departs, on garde le meilleur RSS
        try:
            p, _ = curve_fit(f, x, y, p0=[ym, xm, n0], bounds=([0, 1e-6, 0.5], [np.inf, np.inf, 6.0]), maxfev=20000)
            rss = np.sum((y - f(x, *p)) ** 2)
            if best is None or rss < best[0]: best = (rss, p)
        except Exception:
            pass
    if best is None: return None
    rss, p = best
    return dict(a=p[0], h=p[1], n=p[2], r2=1 - rss / np.sum((y - y.mean()) ** 2))


rows = []
for tr in truth:
    t, r = tr['t'], tr['r']; typ = 'act' if tr['K'] > 0 else 'rep'
    x, y = U[r], LAM * U[t]; h_true = x.mean()
    emb = pca_embed({t, r})
    nn = NearestNeighbors(n_neighbors=max(KS)).fit(emb); idx = nn.kneighbors(emb, return_distance=False)
    conds = {'raw': x}
    for k in KS: conds[f'knn{k}'] = x[idx[:, :k]].mean(1)
    conds['binmean'] = np.array([x[BIN == b].mean() for b in range(6)])[BIN]
    # paire de controle : regulateur non lie, meme cible
    rc = int(rng.choice([g for g in range(G) if g not in (t, r)]))
    xc = U[rc]; embc = pca_embed({t, rc}); idc = NearestNeighbors(n_neighbors=max(KS)).fit(embc).kneighbors(embc, return_distance=False)
    grid = np.linspace(*np.percentile(x, [2, 98]), 120)
    f_true = true_curve(grid, tr['K'], h_true, 2.0)
    row = dict(t=t, r=r, type=typ, K=abs(tr['K']), h_true=float(h_true), cv_within_bin=float(np.mean([x[BIN == b].std() / max(x[BIN == b].mean(), 1e-9) for b in range(6)])))
    for name, xs in conds.items():
        fit = fit_hill(xs, y, typ)
        if fit is None: row[name] = None; continue
        fn = f_act if typ == 'act' else f_rep
        rng_t = f_true.max() - f_true.min()
        row[name] = dict(n=float(fit['n']), K_ratio=float(fit['a'] / abs(tr['K'])), h_ratio=float(fit['h'] / h_true),
                         nrmse=float(np.sqrt(np.mean((f_true - fn(grid, fit['a'], fit['h'], fit['n'])) ** 2)) / rng_t),
                         r2=float(fit['r2']), at_bound=bool(fit['n'] <= 0.51 or fit['n'] >= 5.99))
    # controle : gain de R2 par lissage, paires reelles vs non liees (meme type de modele : meilleur des deux signes)
    def best_r2(xv):
        fa, fr = fit_hill(xv, y, 'act'), fit_hill(xv, y, 'rep')
        return max(f['r2'] for f in (fa, fr) if f is not None)
    for k in (15, 50):
        row[f'gain_r2_real_knn{k}'] = float(best_r2(x[idx[:, :k]].mean(1)) - best_r2(x))
        row[f'gain_r2_ctrl_knn{k}'] = float(best_r2(xc[idc[:, :k]].mean(1)) - best_r2(xc))
    rows.append(row)
    print(f"t={t:3d} r={r:3d} {typ} n_raw={row['raw']['n']:.2f}  knn15={row['knn15']['n']:.2f}  knn50={row['knn50']['n']:.2f}  bin={row['binmean']['n']:.2f}", flush=True)

json.dump(rows, open(OUT, 'w'), indent=1)
print('ecrit', OUT, len(rows), 'cibles')
