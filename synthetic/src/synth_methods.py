"""Neural ODE sur trajectoires (perte L_obs, eq. 7), distillation, bases de reference (identification directe, forme faible, oracle, forme libre), metriques et verification de la proposition 1."""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..', 'real_data', 'src'))
from torchdiffeq import odeint, odeint_adjoint
from scipy.optimize import nnls
from scipy.integrate import solve_ivp
from real_node import VF, field_np
from real_identify import build_candidates, select_structure, predict_all, refine, structure_set, n_params
from sklearn.neighbors import NearestNeighbors


def train_traj_node(Xtr, Xva, times, cfg, seed, nb=None):
    """Perte L_obs gaussienne (eq. 7) ou binomiale negative sur comptages (nb = dict(Ytr, Str, Yva, Sva, kappa)) ; solveur rk4 pas fixe ; retropropagation directe ou methode adjointe (cfg['adjoint'])."""
    torch.manual_seed(seed); d = Xtr.shape[2]; f = VF(d, cfg['hidden']); log_r = torch.nn.Parameter(torch.full((d,), float(np.log(5.0))))
    opt = torch.optim.Adam(list(f.parameters()) + ([log_r] if nb else []), lr=cfg['lr'])
    tt = torch.as_tensor(times, dtype=torch.float32); Xt, Xv = [torch.as_tensor(a, dtype=torch.float32) for a in (Xtr, Xva)]
    scale = float(Xt.reshape(-1, d).var(0).sum()) + 1e-9; step = float(min(cfg['step_size'], np.diff(times).min() / 2)); opts = dict(step_size=step)
    if nb: Yt, St, Yv, Sv = [torch.as_tensor(nb[k], dtype=torch.float32) for k in ('Ytr', 'Str', 'Yva', 'Sva')]; kap = float(nb['kappa'])
    def traj(f_, X0):
        if cfg.get('adjoint'): return odeint_adjoint(f_, X0, tt, method='rk4', options=opts, adjoint_params=tuple(f_.parameters())).permute(1, 0, 2)
        return odeint(f_, X0, tt, method='rk4', options=opts).permute(1, 0, 2)
    def nll(pred, Y, S):
        mu = (kap * S[..., None] * pred).clamp_min(1e-6); r = log_r.exp(); return -(torch.lgamma(Y + r) - torch.lgamma(r) - torch.lgamma(Y + 1) + r * torch.log(r / (r + mu)) + Y * torch.log(mu / (r + mu))).mean()
    best = (np.inf, None); bad = 0
    for ep in range(cfg['epochs']):
        opt.zero_grad(); pred = traj(f, Xt[:, 0])
        l_obs = nll(pred[:, 1:], Yt[:, 1:], St[:, 1:]) if nb else ((pred[:, 1:] - Xt[:, 1:]) ** 2).sum(-1).mean() / scale          # L_obs (eq. 7) ou L_NB (eq. 9)
        reg = (f(0.0, Xt.reshape(-1, d)) ** 2).sum(1).mean() / scale
        loss = l_obs + cfg['lam_reg'] * reg; loss.backward(); torch.nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step()
        if ep % 5 == 0:
            with torch.no_grad():
                pv = traj(f, Xv[:, 0]); lv = float(nll(pv[:, 1:], Yv[:, 1:], Sv[:, 1:]) if nb else ((pv[:, 1:] - Xv[:, 1:]) ** 2).sum(-1).mean() / scale)
            if lv < best[0] - 1e-7: best = (lv, {k: v.clone() for k, v in f.state_dict().items()}); bad = 0
            else:
                bad += 1
                if bad > 25: break
    f.load_state_dict(best[1]); return f, dict(best_val=best[0], epochs_run=ep + 1, loss='nb' if nb else 'gaussian', adjoint=bool(cfg.get('adjoint')))


def integrate_node(f, Z0, times):
    with torch.no_grad(): return odeint(f, torch.as_tensor(np.asarray(Z0), dtype=torch.float32), torch.as_tensor(times, dtype=torch.float32), rtol=1e-6, atol=1e-8).permute(1, 0, 2).numpy().astype(float)


def sample_Df(f, Z0tr, t_end, rng, dist):
    nf = dist['n_fine']; rec = integrate_node(f, Z0tr, np.linspace(0, t_end, nf)).reshape(-1, Z0tr.shape[1]); M = dist['M']
    nn = NearestNeighbors(n_neighbors=dist['knn'] + 1).fit(rec); dk = nn.kneighbors(rec)[0][:, -1]; thr = np.quantile(dk, dist['reject_quantile'])
    i = rng.choice(len(rec), M); cand = np.maximum(rec[i] + rng.normal(size=(M, rec.shape[1])) * (dist['jitter_frac'] * dk[i] / np.sqrt(rec.shape[1]))[:, None], 0)
    keep = nn.kneighbors(cand, n_neighbors=1)[0][:, 0] <= thr; obs = rec[rng.choice(len(rec), M // 2)]
    return np.vstack([obs, cand[keep][:M - len(obs)]])


def fd_targets(X, t):
    return np.gradient(X, t, axis=1)


def weak_arrays(X, t, w, stride, Phi_cols):
    """Forme faible : int phi x_dot = - int phi' x ; phi = ((t-a)(b-t))^2 normalise. Retourne A (fenetres, C), D (fenetres, d), Y (fenetres, d)."""
    R, nt, d = X.shape; Ls = list(range(0, nt - w + 1, stride)); A, D, Y = [], [], []
    for s in Ls:
        tw = t[s:s + w]; a, b = tw[0], tw[-1]; u = (tw - a) * (b - tw); nrm = ((b - a) / 2) ** 4
        phi = u ** 2 / nrm; dphi = 2 * u * (a + b - 2 * tw) / nrm; q = np.zeros(w); dt = np.diff(tw); q[:-1] += dt / 2; q[1:] += dt / 2
        B, Bp = phi * q, dphi * q
        A.append(np.einsum('k,rkc->rc', B, Phi_cols[:, s:s + w, :])); D.append(np.einsum('k,rkd->rd', B, X[:, s:s + w, :])); Y.append(-np.einsum('k,rkd->rd', Bp, X[:, s:s + w, :]))
    return np.concatenate(A), np.concatenate(D), np.concatenate(Y)


def weak_select(Atr, Dtr, ytr, Ava, Dva, yva, i, meta, kmax, tol):
    sel = []; pareto = []
    def fit(S, A, D, y):
        M = np.column_stack([A[:, S], -D]) if S else (-D)[:, None]; c, _ = nnls(M, y, maxiter=2000); return c, M
    def to_model(c, S):
        terms = []
        for cc, idx in zip(c[:-1], S): t = dict(meta[idx]); t['a'] = float(cc); terms.append(t)
        return dict(i=i, terms=terms, g=float(c[-1]))
    c, _ = fit([], Atr, Dtr, ytr); mv = float(np.mean((np.column_stack([Ava[:, []], -Dva]) @ c - yva) ** 2)) if True else 0
    pareto.append(dict(k=0, omega=1, mse_val=mv, model=to_model(c, [])))
    for k in range(1, kmax + 1):
        best = (np.inf, None); used = {(meta[s].get('j'), meta[s].get('typ')) if meta[s]['kind'] == 'hill' else ('const',) for s in sel}
        for cnd in range(len(meta)):
            key = (meta[cnd].get('j'), meta[cnd].get('typ')) if meta[cnd]['kind'] == 'hill' else ('const',)
            if key in used: continue
            c, M = fit(sel + [cnd], Atr, Dtr, ytr); rss = float(np.sum((M @ c - ytr) ** 2))
            if rss < best[0]: best = (rss, cnd)
        if best[1] is None: break
        sel.append(best[1]); c, _ = fit(sel, Atr, Dtr, ytr); Mv = np.column_stack([Ava[:, sel], -Dva])
        pareto.append(dict(k=k, omega=n_params(to_model(c, sel)['terms']), mse_val=float(np.mean((Mv @ c - yva) ** 2)), model=to_model(c, sel)))
    mn = min(p['mse_val'] for p in pareto)
    return min([p for p in pareto if p['mse_val'] <= (1 + tol) * mn], key=lambda p: (p['omega'], p['mse_val']))['model']


def cosine(Fp, Ft, eps=1e-9):
    a, b = np.linalg.norm(Fp, axis=1), np.linalg.norm(Ft, axis=1); m = (a > eps) & (b > eps); return float(np.mean(np.sum(Fp[m] * Ft[m], 1) / (a[m] * b[m])))


def nrmse(Fp, Ft): return float(np.sqrt(np.mean((Fp - Ft) ** 2)) / (Ft.max() - Ft.min()))


def structure_metrics(models, eqs):
    true = {(e['i'], t['j'], t['typ']) for e in eqs for t in e['terms']}
    est = {(m['i'], t['j'], t['typ']) for m in models for t in m['terms'] if t['kind'] == 'hill'}
    tp = true & est; p = len(tp) / max(len(est), 1); r = len(tp) / len(true); f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    regs_t = {(i, j) for i, j, _ in true}; regs_e = {(i, j) for i, j, _ in est}
    exact = np.mean([{(t['j'], t['typ']) for t in e['terms']} == {(t['j'], t['typ']) for t in m['terms'] if t['kind'] == 'hill'} for e, m in zip(eqs, models)])
    errs = {'a': [], 'K': [], 'n': []}
    for e in eqs:
        m = next(m for m in models if m['i'] == e['i'])
        for t in e['terms']:
            for u in m['terms']:
                if u['kind'] == 'hill' and u['j'] == t['j'] and u['typ'] == t['typ']:
                    for k in errs: errs[k].append(abs(u[k] - t[k]) / t[k])
    return dict(precision=p, recall=r, F1=f1, edge_recall_ignoring_type=len(regs_t & regs_e) / len(regs_t), exact_equation_rate=float(exact), n_terms_est=len(est), n_terms_true=len(true),
                n_matched=len(tp), param_relerr_median={k: (float(np.median(v)) if v else None) for k, v in errs.items()}, param_relerr_all_median=(float(np.median(sum(errs.values(), []))) if tp else None))


def lipschitz(F, pts, h=1e-4):
    d = pts.shape[1]; J = np.zeros((len(pts), d, d))
    for k in range(d):
        e = np.zeros(d); e[k] = h; J[:, :, k] = (F(pts + e) - F(pts - e)) / (2 * h)
    return float(np.linalg.norm(J, 2, axis=(1, 2)).max())


def prop1_check(F_true, F_hat, Ztraj, t, Z0):
    """Verifie la borne de Groenwall (proposition 1) sur chaque trajectoire de test : E = erreur de trajectoire, B = (eps/L)(exp(L t)-1)."""
    rows = []
    for r in range(len(Ztraj)):
        Zt = Ztraj[r]; eps = float(np.linalg.norm(F_true(Zt) - F_hat(Zt), axis=1).max())
        s = solve_ivp(lambda tt, x: F_hat(x[None])[0], (t[0], t[-1]), Z0[r], t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
        if s.y.shape[1] != len(t): continue
        W = s.y.T; E = float(np.linalg.norm(Zt - W, axis=1).max()); L = lipschitz(F_hat, np.vstack([Zt, W]))
        if L < 1e-9: continue
        B = eps / L * (np.exp(L * (t[-1] - t[0])) - 1); rows.append(dict(eps=eps, L=L, E=E, B=B, holds=bool(E <= B * (1 + 1e-6) + 1e-12)))
    if not rows: return None
    return dict(n=len(rows), holds_fraction=float(np.mean([r['holds'] for r in rows])), median_B_over_E=float(np.median([r['B'] / max(r['E'], 1e-12) for r in rows])), median_E=float(np.median([r['E'] for r in rows])), median_eps=float(np.median([r['eps'] for r in rows])), median_L=float(np.median([r['L'] for r in rows])))


def free_form_gplearn(Ztr, Ftr, Zte, Fnode_te, eqs, fcfg, seed):
    from gplearn.genetic import SymbolicRegressor
    d = Ztr.shape[1]; out = []; used_sets = []
    for i in range(d):
        sr = SymbolicRegressor(population_size=fcfg['population'], generations=fcfg['generations'], function_set=('add', 'sub', 'mul', 'div'), parsimony_coefficient=0.01, random_state=seed + i, n_jobs=1, verbose=0)
        sr.fit(Ztr[:2000], Ftr[:2000, i]); pred = sr.predict(Zte); prog = str(sr._program); used = {int(x) for x in __import__('re').findall(r'X(\d+)', prog)}
        true_set = {t['j'] for t in eqs[i]['terms']} | {i}
        out.append(dict(i=i, program=prog, length=int(sr._program.length_), r2_vs_node=float(1 - np.sum((pred - Fnode_te[:, i]) ** 2) / np.sum((Fnode_te[:, i] - Fnode_te[:, i].mean()) ** 2)), used=sorted(used),
                        var_recall=len(used & true_set) / len(true_set), var_precision=(len(used & true_set) / len(used)) if used else 0.0))
    return dict(equations=out, mean_var_recall=float(np.mean([o['var_recall'] for o in out])), mean_var_precision=float(np.mean([o['var_precision'] for o in out])), mean_length=float(np.mean([o['length'] for o in out])), mean_r2_vs_node=float(np.mean([o['r2_vs_node'] for o in out])))


def profile_ci(Z, y, model, ti, param, lib, ngrid=15, tol_var=0.01):
    """Profil (identifiabilite pratique, VI-A) d'un parametre (K ou n) du terme ti : le parametre est fixe sur une grille, les autres sont re-optimises.
    INTERVALLE DE TOLERANCE (pas un IC statistique : la cible est le champ deterministe du Neural ODE, dont l'erreur est un biais et non un bruit) :
    valeurs dont la perte d'ajustement reste <= RSS_min + tol_var * SST (par defaut 1 % de la variance totale de la cible)."""
    from scipy.optimize import least_squares
    from real_identify import predict_eq
    Kb, nb_ = lib['K_bounds'], lib['n_bounds']; terms = model['terms']; amax = 10 * (np.abs(y).max() + 1e-9)
    names = []; p0 = []; lo = []; hi = []
    for k, t in enumerate(terms):
        for nm, b in (('a', (0, amax)), ('K', Kb), ('n', nb_)):
            if t['kind'] != 'hill' and nm != 'a': continue
            names.append((k, nm)); p0.append(t[nm]); lo.append(b[0]); hi.append(b[1])
    names.append((None, 'g')); p0 += [model['g']]; lo += [0.0]; hi += [50.0]
    def build(p):
        m = dict(i=model['i'], g=float(p[-1]), terms=[dict(t) for t in terms])
        for (k, nm), v in zip(names[:-1], p[:-1]): m['terms'][k][nm] = float(v)
        return m
    ix = names.index((ti, param)); free = [j for j in range(len(names)) if j != ix]; p0 = np.array(p0, float); lo = np.array(lo, float); hi = np.array(hi, float)
    def rss_at(v):
        def res(q):
            p = p0.copy(); p[free] = q; p[ix] = v; return predict_eq(build(p), Z) - y
        r = least_squares(res, np.clip(p0[free], lo[free] + 1e-9, hi[free] - 1e-9), bounds=(lo[free], hi[free]), method='trf', x_scale='jac', max_nfev=60); return 2 * r.cost
    grid = np.geomspace(lo[ix], hi[ix], ngrid) if param == 'K' else np.linspace(lo[ix], hi[ix], ngrid)
    rss = np.array([rss_at(v) for v in grid]); rmin = min(rss.min(), float(np.sum((predict_eq(model, Z) - y) ** 2))); sst = float(np.sum((y - y.mean()) ** 2))
    ok = grid[(rss - rmin) <= tol_var * sst]
    return dict(lo=float(ok.min()) if len(ok) else None, hi=float(ok.max()) if len(ok) else None, at_bound=bool(len(ok) and (ok.min() <= grid[0] * 1.0001 or ok.max() >= grid[-1] * 0.9999)))


def profile_coverage(models, eqs, Zf, Ff, lib, rng_unused=None):
    rows = []
    for e in eqs:
        m = next(m for m in models if m['i'] == e['i'])
        for ti, u in enumerate(m['terms']):
            if u['kind'] != 'hill': continue
            tt = next((t for t in e['terms'] if t['j'] == u['j'] and t['typ'] == u['typ']), None)
            if tt is None: continue
            for pr in ('K', 'n'):
                ci = profile_ci(Zf, Ff[:, e['i']], m, ti, pr, lib)
                rows.append(dict(param=pr, true=tt[pr], est=u[pr], lo=ci['lo'], hi=ci['hi'], covered=bool(ci['lo'] is not None and ci['lo'] <= tt[pr] <= ci['hi']), at_bound=ci['at_bound'],
                                 log_width=(float(np.log10(ci['hi'] / ci['lo'])) if ci['lo'] and pr == 'K' else (float(ci['hi'] - ci['lo']) if ci['lo'] is not None else None))))
    if not rows: return None
    out = dict(n=len(rows))
    for pr in ('K', 'n'):
        R_ = [r for r in rows if r['param'] == pr]; out[pr] = dict(n=len(R_), coverage=float(np.mean([r['covered'] for r in R_])) if R_ else None, frac_at_bound=float(np.mean([r['at_bound'] for r in R_])) if R_ else None,
                                                                    median_width=float(np.median([r['log_width'] for r in R_ if r['log_width'] is not None])) if R_ else None)
    return out


def rk4_batch(F, Z0, T, dt=0.05):
    Z = np.array(Z0, float); n = int(round(T / dt))
    for _ in range(n):
        k1 = F(Z); k2 = F(Z + dt / 2 * k1); k3 = F(Z + dt / 2 * k2); k4 = F(Z + dt * k3); Z = Z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Z


def ko_eval(F_true, F_sym, F_node, Z0, T):
    """Validation interventionnelle in silico (VI-C) : delete du gene k (z_k = 0, dz_k = 0) ; decalage de l'etat final par rapport au sauvage ; cosinus et accord de signe avec le decalage vrai."""
    d = Z0.shape[1]; res = {n: [] for n in ('sym', 'node')}; sign = {n: [] for n in ('sym', 'node')}
    def ko(F, k):
        def G(Z):
            Z2 = Z.copy(); Z2[:, k] = 0.0; out = F(Z2); out[:, k] = 0.0; return out
        return G
    wt = {n: rk4_batch(F, Z0, T) for n, F in (('true', F_true), ('sym', F_sym), ('node', F_node))}
    for k in range(d):
        Zk = Z0.copy(); Zk[:, k] = 0.0; keep = [j for j in range(d) if j != k]
        sh = {n: (rk4_batch(ko(F, k), Zk, T) - wt[n])[:, keep] for n, F in (('true', F_true), ('sym', F_sym), ('node', F_node))}
        for n in ('sym', 'node'):
            a, b = sh[n].ravel(), sh['true'].ravel(); res[n].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            big = np.abs(b) > 0.05; sign[n].append(float(np.mean(np.sign(a[big]) == np.sign(b[big]))) if big.any() else np.nan)
    return dict(cos_sym=float(np.mean(res['sym'])), cos_node=float(np.mean(res['node'])), sign_agree_sym=float(np.nanmean(sign['sym'])), sign_agree_node=float(np.nanmean(sign['node'])))
