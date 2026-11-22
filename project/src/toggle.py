"""Modele A - dynamique de reference : toggle switch (verite terrain connue)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

P_TRUE = dict(a=(4.0, 3.6), K=(0.8, 1.0), n=2.0, g=(1.0, 0.9))


def field(X, p=P_TRUE):
    """X: (...,2) -> (...,2). x1 est reprime par x2 (seuil K[0]); x2 est reprime par x1 (seuil K[1])."""
    X = np.asarray(X, float); x1, x2 = X[..., 0], X[..., 1]
    d1 = p['a'][0] / (1 + (np.maximum(x2, 0) / p['K'][0]) ** p['n']) - p['g'][0] * x1
    d2 = p['a'][1] / (1 + (np.maximum(x1, 0) / p['K'][1]) ** p['n']) - p['g'][1] * x2
    return np.stack([d1, d2], -1)


def simulate(x0, t, p=P_TRUE):
    sol = solve_ivp(lambda s, x: field(x, p), (t[0], t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12, method='LSODA')
    return sol.y.T


def make_dataset(seed, n_traj, t, lo, hi, p=P_TRUE):
    rng = np.random.default_rng(seed)
    X0 = rng.uniform(lo, hi, size=(n_traj, 2))
    X = np.stack([simulate(x0, t, p) for x0 in X0])          # (n_traj, nt, 2)
    return X0, X, field(X, p)                                  # vitesses exactes


def jacobian(fun, x, eps=1e-6):
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        J[:, j] = (fun(x + d) - fun(x - d)) / (2 * eps)
    return J


def fixed_points(fun, starts=None, tol=1e-6):
    """Points fixes de fun: R^2->R^2, classes par valeurs propres du jacobien numerique."""
    if starts is None:
        g = np.linspace(0.0, 4.0, 9); starts = [(a, b) for a in g for b in g]
    found = []
    for s in starts:
        x, info, ier, _ = fsolve(fun, s, full_output=True, xtol=1e-12)
        if ier == 1 and np.linalg.norm(fun(x)) < tol and (x > -1e-6).all():
            if not any(np.linalg.norm(x - y['x']) < 1e-4 for y in found):
                ev = np.linalg.eigvals(jacobian(fun, x))
                found.append(dict(x=x, stable=bool((ev.real < 0).all()), eig=ev.real.tolist()))
    return sorted(found, key=lambda d: tuple(d['x']))
