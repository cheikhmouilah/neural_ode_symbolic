"""Systemes generiques pour le test preenregistre des regles de detection (nouveaux systemes).
p : dict(a=(a1,a2), K=(K1,K2), n, g=(g1,g2), [eps], [w], [b], [Ks]). eq1 est reprime par x2 (K[0]) ; eq2 par x1 (K[1])."""
import numpy as np
from scipy.integrate import solve_ivp


def rep(x, K, kind, p):
    x = np.maximum(x, 0); n = p['n']
    if kind in ('hill', 'selfact'): return 1.0 / (1.0 + (x / K) ** n)
    if kind == 'leaky': e = p['eps']; return e + (1 - e) / (1.0 + (x / K) ** n)
    if kind == 'gauss': return np.exp(-(x / K) ** n)
    if kind == 'fermi': return 1.0 / (1.0 + np.exp(np.clip((x - K) / p['w'], -50, 50)))
    raise ValueError(kind)


def field_gen(X, p, kind):
    X = np.asarray(X, float); x1, x2 = X[..., 0], X[..., 1]
    d1 = p['a'][0] * rep(x2, p['K'][0], kind, p) - p['g'][0] * x1
    d2 = p['a'][1] * rep(x1, p['K'][1], kind, p) - p['g'][1] * x2
    if kind == 'selfact':
        d1 = d1 + p['b'][0] * np.maximum(x1, 0) ** 2 / (p['Ks'] ** 2 + np.maximum(x1, 0) ** 2)
        d2 = d2 + p['b'][1] * np.maximum(x2, 0) ** 2 / (p['Ks'] ** 2 + np.maximum(x2, 0) ** 2)
    return np.stack([d1, d2], -1)


def make_dataset_gen(seed, n_traj, t, lo, hi, p, kind):
    X0 = np.random.default_rng(seed).uniform(lo, hi, size=(n_traj, 2))
    X = np.stack([solve_ivp(lambda s, x: field_gen(x, p, kind), (t[0], t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12, method='LSODA').y.T for x0 in X0])
    return X0, X, field_gen(X, p, kind)
