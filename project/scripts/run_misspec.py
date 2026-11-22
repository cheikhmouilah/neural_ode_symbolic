import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time, argparse, numpy as np
sys.path.insert(0, ROOT + '/src')
from misspec import field_alt, make_dataset_alt
from guards import assert_bistable
from toggle import fixed_points
from node import train_node, field_np, predict
from identify import structured_search
from families import select_family, fit_family, make_rhs_fam, NAMES
from scipy.integrate import solve_ivp
ap = argparse.ArgumentParser(); ap.add_argument('--kinds', nargs='+', default=['hill', 'leaky', 'gauss']); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); args = ap.parse_args()
P1 = json.load(open(ROOT + '/configs/phase1.json')); M = json.load(open(ROOT + '/configs/misspec.json'))
D, N, LIB = P1['data'], P1['node'], P1['library']; OUT = ROOT + '/results/phase1'; _os.makedirs(OUT, exist_ok=True)
rng_ = lambda a: float(a.max() - a.min()); nrmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / rng_(b))


def truth_params(kind, s, i):
    a, K, n, g = s['a'][i], s['K'][i], s['n'], s['g'][i]
    if kind == 'leaky': e = s['eps']; return dict(a=a * (1 - e), c=a * e, K=K, n=n, g=g)
    return dict(a=a, K=K, n=n, g=g)


def relerr(kind, p, s, i):
    tp = truth_params(kind, s, i); return {k: abs(v - tp[k]) / tp[k] for k, v in zip(NAMES[kind], p)}


def fpcmp(truth_fun, model_fun):
    st = [np.linspace(0, 5, 11)]; starts = [(a, b) for a in st[0] for b in st[0]]
    ref = [d for d in fixed_points(truth_fun, starts=starts) if d['stable']]; got = [d for d in fixed_points(model_fun, starts=starts) if d['stable']]
    if len(ref) != len(got): return dict(n_ref=len(ref), n_model=len(got), max_rel_err=None)
    return dict(n_ref=len(ref), n_model=len(got), max_rel_err=float(max(min(np.linalg.norm(g['x'] - r['x']) / max(np.linalg.norm(r['x']), 1e-9) for g in got) for r in ref)))


def traj(rhs, X0, t):
    return np.stack([solve_ivp(lambda _, x: rhs(x), (t[0], t[-1]), x0, t_eval=t, rtol=1e-8, atol=1e-10, method='LSODA').y.T for x0 in X0])


for kind in args.kinds:
    s = M['systems'][kind]; eps = s.get('eps', 0.0); p = dict(a=tuple(s['a']), K=tuple(s['K']), n=s['n'], g=tuple(s['g']))
    truth = lambda X: field_alt(X, p, kind, eps)
    GUARD = assert_bistable(lambda x: truth(x), f'systeme {kind}')   # arret avant entrainement
    for seed in args.seeds:
        t0 = time.time(); print(f'=== {kind} seed {seed} ===', flush=True)
        t = np.linspace(0, D['T'], D['n_times']); X0, X, V = make_dataset_alt(seed, 60, t, 0.0, s['ic_high'], p, kind, eps)
        tr, va, te = slice(0, 40), slice(40, 50), slice(50, None)
        f, info = train_node(t, X[tr], X[va], N, seed, log=lambda m: None)
        Xtr, Xte = X[tr].reshape(-1, 2), X[te].reshape(-1, 2); Fn_tr, Fn_te, Ft_te = field_np(f, Xtr), field_np(f, Xte), truth(Xte)
        r = dict(kind=kind, seed=seed, node_field_nrmse_vs_truth=nrmse(Fn_te, Ft_te), node_traj_nrmse=float(np.sqrt(np.mean((predict(f, X0[te], t, N) - X[te]) ** 2)) / rng_(X[te])))
        search = [structured_search(Xtr, Fn_tr, i, LIB['K_grid'], LIB['n_grid'])[0] for i in (0, 1)]
        r['search'] = [dict(reg=b['reg'], type=b['type']) for b in search]; r['reg_sign_ok'] = bool(all(search[i]['reg'] == 1 - i and search[i]['type'] == 'rep' for i in (0, 1)))
        # (a) Hill seul (famille imposee)
        hill = [dict(fam='hill', reg=search[i]['reg'], typ=search[i]['type'], p=fit_family('hill', Xtr, Fn_tr, i, search[i]['reg'], search[i]['type'], search[i]['a'], search[i]['K'], search[i]['g'])['p']) for i in (0, 1)]
        rh = make_rhs_fam(hill)
        r['hill_only'] = dict(nrmse_sym_to_node=nrmse(rh(Xte), Fn_te), nrmse_sym_to_truth=nrmse(rh(Xte), Ft_te), traj_nrmse=float(np.sqrt(np.mean((traj(rh, X0[te], t) - X[te]) ** 2)) / rng_(X[te])),
                              fixed_points=fpcmp(truth, lambda x: rh(np.asarray(x))), flagged_D1=bool(nrmse(rh(Xte), Fn_te) > 0.03))
        # (b) selection de famille par BIC
        sel = []; allfits = []
        for i in (0, 1):
            ch, out, reg, typ = select_family(Xtr, Fn_tr, i, search[i]); sel.append(dict(fam=ch, reg=reg, typ=typ, p=out[ch]['p'])); allfits.append({f_: dict(bic=v['bic'], rss=v['rss'], p=v['p']) for f_, v in out.items()})
        rs = make_rhs_fam(sel)
        r['selection'] = dict(chosen=[e['fam'] for e in sel], correct_both=bool(all(e['fam'] == kind for e in sel)), fits=allfits,
                              nrmse_sym_to_node=nrmse(rs(Xte), Fn_te), nrmse_sym_to_truth=nrmse(rs(Xte), Ft_te), traj_nrmse=float(np.sqrt(np.mean((traj(rs, X0[te], t) - X[te]) ** 2)) / rng_(X[te])),
                              fixed_points=fpcmp(truth, lambda x: rs(np.asarray(x))))
        # (c) parametres de la famille correcte (independamment de la selection)
        cf = [fit_family(kind, Xtr, Fn_tr, i, search[i]['reg'], search[i]['type'], search[i]['a'], search[i]['K'], search[i]['g']) for i in (0, 1)]
        r['true_family_params'] = [dict(p=c['p'], rel_err=relerr(kind, c['p'], s, i)) if c else None for i, c in enumerate(cf)]
        errs = [v for c in r['true_family_params'] if c for v in c['rel_err'].values()]; r['true_family_median_rel_err'] = float(np.median(errs)) if errs else None; r['true_family_max_rel_err'] = float(np.max(errs)) if errs else None
        r['seconds'] = time.time() - t0; json.dump(r, open(f'{OUT}/misspec_{kind}_seed{seed}.json', 'w'), indent=1, default=float)
        print(f"{kind} s{seed}: NODE champ={r['node_field_nrmse_vs_truth']:.4f} | Hill seul: sym->NODE={r['hill_only']['nrmse_sym_to_node']:.4f} flag={r['hill_only']['flagged_D1']} traj={r['hill_only']['traj_nrmse']:.4f} | choix={r['selection']['chosen']} | famille vraie: err med/max={r['true_family_median_rel_err']:.3f}/{r['true_family_max_rel_err']:.3f} | {r['seconds']:.0f}s", flush=True)
print('FIN')
