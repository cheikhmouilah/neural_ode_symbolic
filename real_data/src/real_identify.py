"""Identification symbolique en dimension d : bibliotheque de Hill (eq. 4-5, IV-A), selection de structure par selection avant
(moindres carres non negatifs), raffinement des constantes par moindres carres bornes multi-departs (trf, IV-A), front de Pareto
precision/complexite et regle de selection fixee a l'avance (IV-B). Equation i :  dz_i/dt = sum_k a_k H_k(z_j; K, n) + a_0 - delta z_i.
Les cibles sont soit le champ du Neural ODE (distillation, IV-C), soit les vitesses mesurees (identification directe, baseline)."""
import numpy as np
from scipy.optimize import nnls, least_squares


def hill(x, K, n, typ):
    h = 1.0 / (1.0 + (np.maximum(x, 0) / K) ** n)
    return h if typ == 'rep' else 1.0 - h


def build_candidates(Z, Kgrid, ngrid):
    d = Z.shape[1]; meta = [dict(kind='const')]; cols = [np.ones(len(Z))]
    for j in range(d):
        for typ in ('rep', 'act'):
            for K in Kgrid:
                for n in ngrid:
                    meta.append(dict(kind='hill', j=j, typ=typ, K=K, n=n)); cols.append(hill(Z[:, j], K, n, typ))
    return meta, np.stack(cols, 1)


def n_params(terms): return 1 + sum(3 if t['kind'] == 'hill' else 1 for t in terms)   # + delta


def predict_eq(model, Z):
    out = -model['g'] * Z[:, model['i']]
    for t in model['terms']:
        if t['kind'] == 'poly': out = out + t['a'] * (np.ones(len(Z)) if t['j'] is None else Z[:, t['j']] * (1.0 if t['k'] is None else Z[:, t['k']]))
        else: out = out + (t['a'] if t['kind'] == 'const' else t['a'] * hill(Z[:, t['j']], t['K'], t['n'], t['typ']))
    return out


def predict_all(models, Z): return np.stack([predict_eq(m, Z) for m in models], 1)


def _fit_nnls(Phi, Z, y, i, sel, meta):
    A = np.column_stack([Phi[:, sel], -Z[:, i]]) if sel else (-Z[:, i])[:, None]
    coef, _ = nnls(A, y, maxiter=2000)
    terms = []
    for c, idx in zip(coef[:-1], sel):
        t = dict(meta[idx]); t['a'] = float(c); terms.append(t)
    return dict(i=i, terms=terms, g=float(coef[-1])), float(np.sum((y - A @ coef) ** 2))


def refine(Z, y, model, lib, rng):
    """Raffinement borne multi-departs (trf). 
    Prend en charge le mode continu standard et le mode biophysique discret (n in {1, 2, 4})
    qui elimine la colinearite non identifiable entre K et n."""
    i = model['i']; Kb, nb = lib['K_bounds'], lib['n_bounds']; amax = 10 * (np.abs(y).max() + 1e-9)
    discrete_n_set = lib.get('discrete_n_set', (1.0, 2.0, 4.0)) if lib.get('discrete_n') else None

    # Mode 1 : Discretisation biophysique n in {1, 2, 4} (cooperativite monomere / dimere / tetramere)
    if discrete_n_set is not None and any(t['kind'] == 'hill' for t in model['terms']):
        hill_terms = [t for t in model['terms'] if t['kind'] == 'hill']
        const_terms = [t for t in model['terms'] if t['kind'] == 'const']
        n_candidates = [(n1,) for n1 in discrete_n_set] if len(hill_terms) == 1 else (
            [(n1, n2) for n1 in discrete_n_set for n2 in discrete_n_set] if len(hill_terms) == 2 else
            [tuple(t.get('n', 2.0) for t in hill_terms)]
        )
        best_overall = (np.inf, None, None)
        for n_tuple in n_candidates:
            p0, lo, hi = [], [], []
            for t in hill_terms: p0 += [t['a'], t['K']]; lo += [0.0, Kb[0]]; hi += [amax, Kb[1]]
            for t in const_terms: p0 += [t['a']]; lo += [0.0]; hi += [amax]
            p0 += [model['g']]; lo += [0.0]; hi += [50.0]
            
            def unpack_disc(p, n_vals):
                m = dict(i=i, terms=[], g=float(p[-1])); k = 0
                for idx, t in enumerate(hill_terms):
                    m['terms'].append(dict(kind='hill', j=t['j'], typ=t['typ'], a=float(p[k]), K=float(p[k+1]), n=float(n_vals[idx])))
                    k += 2
                for t in const_terms:
                    m['terms'].append(dict(kind='const', a=float(p[k]))); k += 1
                return m

            starts = [np.array(p0)]
            for _ in range(lib['n_starts'] - 1):
                p_s = np.array(p0).copy(); k = 0
                for _ in hill_terms:
                    p_s[k+1] = np.clip(p_s[k+1] * np.exp(rng.normal(0, 0.4)), Kb[0], Kb[1]); k += 2
                starts.append(p_s)

            for p_start in starts:
                p_start = np.clip(p_start, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
                r = least_squares(lambda q: predict_eq(unpack_disc(q, n_tuple), Z) - y, p_start,
                                  bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=150)
                cost = 2 * r.cost
                if cost < best_overall[0]: best_overall = (cost, r.x, n_tuple)

        return unpack_disc(best_overall[1], best_overall[2]), float(best_overall[0]), dict(discrete_n=best_overall[2])

    # Mode 2 : Standard continu (retrocompatible avec le protocole historique)
    def pack(m):
        p = []
        for t in m['terms']: p += [t['a']] if t['kind'] == 'const' else [t['a'], t['K'], t['n']]
        return np.array(p + [m['g']])
    def unpack(p):
        m = dict(i=i, terms=[], g=float(p[-1])); k = 0
        for t in model['terms']:
            if t['kind'] == 'const': m['terms'].append(dict(kind='const', a=float(p[k]))); k += 1
            else: m['terms'].append(dict(kind='hill', j=t['j'], typ=t['typ'], a=float(p[k]), K=float(p[k + 1]), n=float(p[k + 2]))); k += 3
        return m
    lo, hi = [], []
    for t in model['terms']:
        lo += [0.0] if t['kind'] == 'const' else [0.0, Kb[0], nb[0]]; hi += [amax] if t['kind'] == 'const' else [amax, Kb[1], nb[1]]
    lo += [0.0]; hi += [50.0]
    p0 = pack(model); starts = [p0]
    for _ in range(lib['n_starts'] - 1):
        p = p0.copy(); k = 0
        for t in model['terms']:
            if t['kind'] == 'const': k += 1
            else: p[k + 1] = np.clip(p[k + 1] * np.exp(rng.normal(0, 0.5)), Kb[0], Kb[1]); p[k + 2] = np.clip(p[k + 2] + rng.normal(0, 0.5), nb[0], nb[1]); k += 3
        starts.append(p)
    res = []
    for p in starts:
        p = np.clip(p, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        r = least_squares(lambda q: predict_eq(unpack(q), Z) - y, p, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=200)
        res.append((2 * r.cost, r.x))
    res.sort(key=lambda a: a[0]); best = res[0]
    near = [x for c, x in res if c <= 1.05 * best[0]]
    spread = None
    if any(t['kind'] == 'hill' for t in model['terms']):
        idxs = []; k = 0
        for t in model['terms']:
            if t['kind'] == 'const': k += 1
            else: idxs += [k + 1, k + 2]; k += 3
        sp = np.array(near)[:, idxs]; spread = dict(n_near=len(near), rel_std=(sp.std(0) / np.maximum(np.abs(sp.mean(0)), 1e-9)).tolist())
    return unpack(best[1]), float(best[0]), dict(costs=[float(c) for c, _ in res], spread=spread)


def select_structure(Ztr, ytr, Zva, yva, i, lib, rng, cand=None):
    """Selection avant jusqu'a k_max termes ; a chaque k : constantes raffinees ; front de Pareto (Omega, erreur de validation)."""
    meta, Phi = cand if cand is not None else build_candidates(Ztr, lib['K_grid'], lib['n_grid'])
    sel = []; pareto = []
    m0, _ = _fit_nnls(Phi, Ztr, ytr, i, [], meta)
    pareto.append(dict(k=0, omega=n_params([]), mse_val=float(np.mean((predict_eq(m0, Zva) - yva) ** 2)), model=m0, refine=None))
    for k in range(1, lib['k_max'] + 1):
        best = (np.inf, None)
        used = {(meta[s].get('j'), meta[s].get('typ')) if meta[s]['kind'] == 'hill' else ('const',) for s in sel}
        for c in range(len(meta)):
            key = (meta[c].get('j'), meta[c].get('typ')) if meta[c]['kind'] == 'hill' else ('const',)
            if key in used: continue
            m, rss = _fit_nnls(Phi, Ztr, ytr, i, sel + [c], meta)
            if rss < best[0]: best = (rss, c, m)
        if best[1] is None: break
        sel.append(best[1]); mref, rss, info = refine(Ztr, ytr, best[2], lib, rng)
        pareto.append(dict(k=k, omega=n_params(mref['terms']), mse_val=float(np.mean((predict_eq(mref, Zva) - yva) ** 2)), model=mref, refine=info))
    mn = min(p['mse_val'] for p in pareto)
    chosen = min([p for p in pareto if p['mse_val'] <= (1 + lib['selection_tolerance']) * mn], key=lambda p: (p['omega'], p['mse_val']))
    return chosen, pareto


def structure_set(models):
    out = []
    for m in models:
        out.append({('const',) if t['kind'] == 'const' else ('poly', t['j'], t['k']) if t['kind'] == 'poly' else (t['j'], t['typ']) for t in m['terms']})
    return out


def jaccard(A, B):
    js = []
    for a, b in zip(A, B):
        u = a | b; js.append(1.0 if not u else len(a & b) / len(u))
    return float(np.mean(js))


def describe(models, genes):
    lines = []
    for m in models:
        parts = []
        for t in m['terms']:
            if t['kind'] == 'const': parts.append(f"{t['a']:.3g}")
            elif t['kind'] == 'poly': parts.append(f"{t['a']:.3g}*" + ('1' if t['j'] is None else genes[t['j']] + ('' if t['k'] is None else '*' + genes[t['k']])))
            else: parts.append(f"{t['a']:.3g}*{'H-' if t['typ']=='rep' else 'H+'}({genes[t['j']]};K={t['K']:.3g},n={t['n']:.2g})")
        lines.append(f"d{genes[m['i']]}/dt = " + (' + '.join(parts) if parts else '0') + f" - {m['g']:.3g}*{genes[m['i']]}")
    return lines


def build_poly_candidates(Z):
    """Bibliotheque polynomiale (constante, lineaire, quadratique) : espace latent, ou les termes de Hill n'ont pas de lecture regulatrice (II-E)."""
    d = Z.shape[1]; meta = [dict(kind='poly', j=None, k=None)]; cols = [np.ones(len(Z))]
    for j in range(d): meta.append(dict(kind='poly', j=j, k=None)); cols.append(Z[:, j])
    for j in range(d):
        for k in range(j, d): meta.append(dict(kind='poly', j=j, k=k)); cols.append(Z[:, j] * Z[:, k])
    return meta, np.stack(cols, 1)


def select_structure_poly(Ztr, ytr, Zva, yva, i, lib, rng, cand=None):
    """Selection avant sur la bibliotheque polynomiale (moindres carres ordinaires, coefficients signes), meme regle de selection que select_structure."""
    meta, Phi = build_poly_candidates(Ztr); sel = []; pareto = []
    def model(S, c): return dict(i=i, g=0.0, terms=[dict(meta[idx], a=float(cc)) for idx, cc in zip(S, c)])
    pareto.append(dict(k=0, omega=1, mse_val=float(np.mean(yva ** 2)), model=model([], []), refine=None))
    for k in range(1, lib['k_max'] + 1):
        best = (np.inf, None)
        for cnd in range(len(meta)):
            if cnd in sel: continue
            c = np.linalg.lstsq(Phi[:, sel + [cnd]], ytr, rcond=None)[0]; rss = float(np.sum((Phi[:, sel + [cnd]] @ c - ytr) ** 2))
            if rss < best[0]: best = (rss, cnd)
        sel.append(best[1]); c = np.linalg.lstsq(Phi[:, sel], ytr, rcond=None)[0]; m = model(sel, c)
        pareto.append(dict(k=k, omega=1 + len(sel), mse_val=float(np.mean((predict_eq(m, Zva) - yva) ** 2)), model=m, refine=None))
    mn = min(p['mse_val'] for p in pareto)
    chosen = min([p for p in pareto if p['mse_val'] <= (1 + lib['selection_tolerance']) * mn], key=lambda p: (p['omega'], p['mse_val']))
    return chosen, pareto
