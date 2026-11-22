import os as _os
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
SERGIO_DIR = _os.path.join(WORK, 'SERGIO')
"""Diagnostic : l'approximation quasi-stationnaire P ~ lambda*U_t est-elle acceptable ?
Residu = lambda*U_t - P_vrai(U_r) avec la vraie equation SERGIO (h = moyenne du regulateur sur toutes les cellules)."""
import numpy as np, json
from validate import load_truth, LAM
U = np.load(WORK + '/sergio_clean.npz')['U']; B = np.repeat(np.arange(6), 300)
hill = lambda x, h, n: x**n / (h**n + x**n)
out = {}
for typ, sel in (("activation", lambda t: t['K'] > 0), ("repression", lambda t: t['K'] < 0)):
    r2, bias = [], []
    for t in load_truth()[:14]:
        if not sel(t): continue
        x, y = U[t['r']], LAM * U[t['t']]; h = x.mean()
        P = abs(t['K']) * (hill(x, h, 2) if t['K'] > 0 else 1 - hill(x, h, 2)); res = y - P
        r2.append(1 - np.sum(res**2) / np.sum((y - y.mean())**2))
        bias.append([res[B == b].mean() / abs(t['K']) for b in range(6)])
    out[typ] = dict(n=len(r2), R2_median=float(np.median(r2)), biais_relatif_moyen_par_type=np.round(np.mean(bias, 0), 3).tolist(),
                    biais_relatif_max_abs=float(np.abs(bias).max()))
print(json.dumps(out, indent=1)); json.dump(out, open(WORK + '/result_steady_state_diag.json', 'w'), indent=1)
