"""Equilibres, stabilite (jacobien, eq. 3), correspondance aux clusters annotes (V-G, VII-C), integration depuis les progeniteurs."""
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp


def jacobian(fun, x, eps=1e-5):
    d = len(x); J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps; J[:, j] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return J


def equilibria(fun, starts, lo, hi, tol, dedup=2e-2):
    found = []
    for s in starts:
        r = root(fun, s, method='hybr', tol=1e-10)
        x = r.x
        if r.success and np.linalg.norm(fun(x)) < tol and np.all(x >= lo) and np.all(x <= hi):
            if not any(np.linalg.norm(x - y['x']) < dedup for y in found):
                ev = np.linalg.eigvals(jacobian(fun, x)); found.append(dict(x=x, stable=bool((ev.real < 0).all()), max_eig_real=float(ev.real.max())))
    return found


def cluster_correspondence(eqs, Z, clusters):
    names = sorted(set(clusters)); cen = {c: Z[clusters == c].mean(0) for c in names}
    rad = {c: float(np.sqrt(((Z[clusters == c] - cen[c]) ** 2).sum(1).mean())) for c in names}
    st = [e for e in eqs if e['stable']]
    rows = []
    for c in names:
        if st:
            dist = [np.linalg.norm(e['x'] - cen[c]) for e in st]; k = int(np.argmin(dist)); rows.append(dict(cluster=c, nearest_stable_dist=float(dist[k]), dist_over_radius=float(dist[k] / rad[c]), radius=rad[c]))
        else: rows.append(dict(cluster=c, nearest_stable_dist=None, dist_over_radius=None, radius=rad[c]))
    return rows, cen


def endpoints(fun_np, z0, T, cen, rtol=1e-6):
    names = list(cen); C = np.stack([cen[n] for n in names]); res = []; speed = []
    for z in z0:
        s = solve_ivp(lambda t, x: fun_np(x), (0, T), z, method='LSODA', rtol=rtol, atol=1e-8)
        xe = s.y[:, -1]
        if not np.all(np.isfinite(xe)):
            xe = np.nan_to_num(xe, nan=0.0, posinf=1e3, neginf=-1e3)
        res.append(names[int(np.argmin(((C - xe) ** 2).sum(1)))])
        f_val = fun_np(xe)
        sp = float(np.linalg.norm(f_val)) if np.all(np.isfinite(f_val)) else 0.0
        speed.append(sp)
    return res, speed
