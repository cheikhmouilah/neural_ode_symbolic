"""Generateur de systemes synthetiques de type reseau regulateur (equations 4-5 de la partie theorique), donnees bruitees, dropout, pseudotemps, couverture."""
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'real_data', 'src'))
from real_identify import predict_all


def gen_system(d, seed):
    rng = np.random.default_rng(10_000 + seed); eqs = []
    for i in range(d):
        nreg = 1 if rng.random() < 0.5 else 2
        regs = rng.choice(d, size=min(nreg, d), replace=False)
        terms = [dict(kind='hill', j=int(j), typ=str(rng.choice(['rep', 'act'])), K=float(rng.uniform(0.4, 1.5)), n=float(rng.choice([1.5, 2.0, 3.0])), a=float(rng.uniform(1.0, 3.0))) for j in regs]
        eqs.append(dict(i=i, terms=terms, g=float(rng.uniform(0.8, 1.4))))
    return eqs


def field_true(eqs, Z): return predict_all(eqs, np.atleast_2d(Z))


def zhi(eqs): return max(sum(t['a'] for t in e['terms']) / e['g'] for e in eqs)


def simulate(eqs, Z0, times):
    out = []
    for z0 in Z0:
        s = solve_ivp(lambda t, x: field_true(eqs, x)[0], (times[0], times[-1]), z0, t_eval=times, method='LSODA', rtol=1e-8, atol=1e-10)
        out.append(s.y.T)
    return np.stack(out)


def sample_ics(eqs, n, rng, mode):
    d = len(eqs); hi = zhi(eqs)
    if mode == 'manifold':
        p0, p1 = rng.uniform(0, hi, d), rng.uniform(0, hi, d); s = rng.uniform(0, 1, (n, 1))
        return np.clip(p0 + s * (p1 - p0) + 0.01 * hi * rng.normal(size=(n, d)), 0, None)
    return rng.uniform(0, hi, (n, d))


def make_data(eqs, cfg, seed):
    rng = np.random.default_rng(seed); d = len(eqs); nt, T = cfg['n_times'], cfg['horizon']
    t_true = np.linspace(0, T, nt)
    Z0 = sample_ics(eqs, cfg['n_traj'], rng, cfg['ic_mode']); Ztrue = simulate(eqs, Z0, t_true)
    sd = Ztrue.reshape(-1, d).std(0)
    extra = {}
    if cfg.get('obs_model', 'gaussian') == 'nb':       # comptages binomiale negative (eq. 2) : moyenne kappa * s * z, dispersion r ; dropout = zero-inflation
        kap, r = cfg['count_scale'], cfg['nb_dispersion']; S = np.exp(0.3 * rng.normal(size=Ztrue.shape[:2]))
        Yc = rng.poisson(rng.gamma(shape=r, scale=kap * S[..., None] * Ztrue / r + 1e-12))
        if cfg['dropout'] > 0: Yc = np.where(rng.random(Yc.shape) < cfg['dropout'], 0, Yc)
        X = np.clip(Yc / (kap * S[..., None]), 0, None); extra = dict(Yc=Yc, S=S, kappa=kap)
    else:
        X = Ztrue + cfg['sigma_rel'] * sd * rng.normal(size=Ztrue.shape)
        if cfg['dropout'] > 0: X = np.where(rng.random(X.shape) < cfg['dropout'], 0.0, X)
        X = np.clip(X, 0, None)
    t_obs = T * (t_true / T) ** cfg['pseudo_gamma'] if cfg['time_mode'] == 'pseudo' else t_true.copy()    # temps vu par l'observateur
    return dict(Ztrue=Ztrue, X=X, t_true=t_true, t_obs=t_obs, Z0=Z0, **extra)
