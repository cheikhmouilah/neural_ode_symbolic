"""Modele C - identification symbolique d'un champ (distillation du Neural ODE ou derivees exactes).
Trois outils : (1) STLSQ de PySINDy sur bibliotheque polynomiale + grille de Hill ; (2) recherche structuree
(un terme de Hill + degradation propre) ; (3) raffinement des constantes par moindres carres a region de confiance (trf)."""
import numpy as np
from scipy.optimize import least_squares
import pysindy as ps


def hill_rep(x, K, n):
    return 1.0 / (1.0 + (np.maximum(x, 0) / K) ** n)


def structured_search(X, F, i, Kgrid, ngrid, top=5):
    """Equation i : F_i ~ a*h_j(x_j;K,n) - g*x_i, avec h = repression ou activation, j in {0,1}. a>=0, g>=0."""
    y = F[:, i]; cands = []
    for j in (0, 1):
        for typ in ('rep', 'act'):
            for K in Kgrid:
                for n in ngrid:
                    h = hill_rep(X[:, j], K, n); h = h if typ == 'rep' else 1 - h
                    A = np.stack([h, -X[:, i]], 1)
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    if coef[0] < 0 or coef[1] < 0: continue
                    rss = float(np.sum((y - A @ coef) ** 2))
                    cands.append(dict(reg=j, type=typ, K=K, n=n, a=float(coef[0]), g=float(coef[1]), rss=rss))
    cands.sort(key=lambda d: d['rss'])
    return cands[0], cands[:top]


def refine(X, F, i, s):
    y = F[:, i]; j, typ = s['reg'], s['type']

    def model(p):
        a, K, n, g = p; h = hill_rep(X[:, j], K, n); h = h if typ == 'rep' else 1 - h
        return a * h - g * X[:, i]
    res = least_squares(lambda p: model(p) - y, x0=[s['a'], s['K'], s['n'], s['g']],
                        bounds=([0, 0.05, 1.0, 0], [50, 20, 6.0, 10]), method='trf', x_scale='jac')
    return dict(reg=j, type=typ, a=float(res.x[0]), K=float(res.x[1]), n=float(res.x[2]), g=float(res.x[3]), rss=float(2 * res.cost))


def sindy_support(X, F, Kgrid, ngrid, threshold=0.05, alpha=1e-6):
    """STLSQ (PySINDy) sur [1, x1, x2, x1^2, x1x2, x2^2, Hill-rep_j(K,n)...]; renvoie les termes actifs par equation."""
    cols, names = [np.ones(len(X)), X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2], ['1', 'x1', 'x2', 'x1^2', 'x1x2', 'x2^2']
    for j in (0, 1):
        for K in Kgrid:
            for n in ngrid:
                cols.append(hill_rep(X[:, j], K, n)); names.append(f'rep(x{j+1};K={K},n={n})')
    Th = np.stack(cols, 1); nrm = np.linalg.norm(Th, axis=0); Ths = Th / nrm
    out = []
    for i in (0, 1):
        opt = ps.STLSQ(threshold=threshold, alpha=alpha)
        opt.fit(Ths, F[:, i:i + 1])
        c = np.asarray(opt.coef_).reshape(-1) / nrm
        out.append({names[k]: float(c[k]) for k in np.nonzero(c)[0]})
    return out


def make_rhs(params):
    def rhs(X):
        X = np.asarray(X, float); out = np.zeros_like(X)
        for i, s in enumerate(params):
            h = hill_rep(X[..., s['reg']], s['K'], s['n']); h = h if s['type'] == 'rep' else 1 - h
            out[..., i] = s['a'] * h - s['g'] * X[..., i]
        return out
    return rhs
