"""Familles candidates pour l'identification d'une equation (dx_i/dt = production(x_reg) - g*x_i) et selection par BIC.
 hill  : a*h(x;K,n) - g x            (h = repression 1/(1+(x/K)^n), ou activation 1-h)
 leaky : a*h(x;K,n) + c - g x        (Hill avec production basale c)
 gauss : a*exp(-(x/K)^n) - g x       (extinction en exponentielle etiree)"""
import numpy as np
from scipy.optimize import least_squares
from identify import hill_rep

BOUNDS = {'hill': ([0, .05, 1, 0], [50, 20, 10, 10]), 'leaky': ([0, 0, .05, 1, 0], [50, 50, 20, 10, 10]), 'gauss': ([0, .05, 1, 0], [50, 20, 10, 10])}
NAMES = {'hill': ['a', 'K', 'n', 'g'], 'leaky': ['a', 'c', 'K', 'n', 'g'], 'gauss': ['a', 'K', 'n', 'g']}
FLOOR = 1e-8      # resolution numerique du champ : evite qu'un RSS ~1e-25 (derivees exactes) rende le BIC arbitraire


def _h(x, K, n, typ):
    h = hill_rep(x, K, n); return h if typ == 'rep' else 1 - h


def model_F(fam, p, xr, xo, typ='rep'):
    if fam == 'hill':  a, K, n, g = p; return a * _h(xr, K, n, typ) - g * xo
    if fam == 'leaky': a, c, K, n, g = p; return a * _h(xr, K, n, typ) + c - g * xo
    if fam == 'gauss': a, K, n, g = p; return a * np.exp(-(np.maximum(xr, 0) / K) ** n) - g * xo


def fit_family(fam, X, F, i, reg, typ, a0, K0, g0):
    if fam != 'hill' and typ != 'rep': return None
    y = F[:, i]; best = None
    for n0 in (1.0, 2.0, 4.0):
        x0 = {'hill': [a0, K0, n0, g0], 'leaky': [0.9 * a0, 0.1 * a0, K0, n0, g0], 'gauss': [a0, 1.3 * K0, n0, g0]}[fam]
        lo, hi = BOUNDS[fam]; x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            r = least_squares(lambda p: model_F(fam, p, X[:, reg], X[:, i], typ) - y, x0, bounds=(lo, hi), method='trf', x_scale='jac')
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best['rss']: best = dict(fam=fam, p=[float(v) for v in r.x], rss=rss)
    if best is None: return None
    N = len(y); k = len(best['p']); best['k'] = k; best['bic'] = float(N * np.log(best['rss'] / N + FLOOR) + k * np.log(N)); return best


def select_family(X, F, i, search_best, fams=('hill', 'leaky', 'gauss')):
    reg, typ = search_best['reg'], search_best['type']
    out = {f: fit_family(f, X, F, i, reg, typ, search_best['a'], search_best['K'], search_best['g']) for f in fams}
    out = {f: v for f, v in out.items() if v is not None}
    chosen = min(out, key=lambda f: out[f]['bic']); return chosen, out, reg, typ


def make_rhs_fam(eqs):
    """eqs: liste de dict(fam, reg, typ, p) pour i=0,1."""
    def rhs(X):
        X = np.asarray(X, float); o = np.zeros_like(X)
        for i, e in enumerate(eqs): o[..., i] = model_F(e['fam'], e['p'], X[..., e['reg']], X[..., i], e['typ'])
        return o
    return rhs
