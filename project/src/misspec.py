"""Systemes alternatifs pour le test de mauvaise specification : meme topologie (toggle), non-linearite de repression differente."""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(a=(4.0, 3.6), K=(0.8, 1.0), n=2.0, g=(1.0, 0.9))


def repression(x, K, n, kind, eps):
    x = np.maximum(x, 0)
    if kind == 'hill':  return 1.0 / (1.0 + (x / K) ** n)
    if kind == 'leaky': return eps + (1 - eps) / (1.0 + (x / K) ** n)        # Hill avec production basale eps*a
    if kind == 'gauss': return np.exp(-(x / K) ** n)                          # extinction en exp(-(x/K)^n)
    raise ValueError(kind)


def field_alt(X, p, kind, eps=0.15):
    X = np.asarray(X, float); x1, x2 = X[..., 0], X[..., 1]
    d1 = p['a'][0] * repression(x2, p['K'][0], p['n'], kind, eps) - p['g'][0] * x1
    d2 = p['a'][1] * repression(x1, p['K'][1], p['n'], kind, eps) - p['g'][1] * x2
    return np.stack([d1, d2], -1)


def simulate_alt(x0, t, p, kind, eps=0.15):
    return solve_ivp(lambda s, x: field_alt(x, p, kind, eps), (t[0], t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12, method='LSODA').y.T


def make_dataset_alt(seed, n_traj, t, lo, hi, p, kind, eps=0.15):
    X0 = np.random.default_rng(seed).uniform(lo, hi, size=(n_traj, 2))
    X = np.stack([simulate_alt(x0, t, p, kind, eps) for x0 in X0]); return X0, X, field_alt(X, p, kind, eps)
