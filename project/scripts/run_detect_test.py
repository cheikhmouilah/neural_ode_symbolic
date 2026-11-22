import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time, argparse, numpy as np
sys.path.insert(0, ROOT + '/src')
from misspec2 import field_gen, make_dataset_gen
from toggle import fixed_points
from guards import assert_bistable
from node import train_node, field_np, predict
from identify import structured_search
from families import fit_family, make_rhs_fam
from scipy.integrate import solve_ivp
ap = argparse.ArgumentParser(); ap.add_argument('--systems', nargs='+', default=['C1', 'C2', 'C3', 'M1', 'M2', 'M3', 'M4']); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); args = ap.parse_args()
P1 = json.load(open(ROOT + '/configs/phase1.json')); C = json.load(open(ROOT + '/configs/detect_test.json'))
D, N, LIB = P1['data'], P1['node'], P1['library']; OUT = ROOT + '/results/phase1'; _os.makedirs(OUT, exist_ok=True); THETA = C['rules']['theta_G']
rng_ = lambda a: float(a.max() - a.min()); nrmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / rng_(b))
STARTS = [(a, b) for a in np.linspace(0, 7, 15) for b in np.linspace(0, 7, 15)]


def fpcmp(tf, mf):
    ref = [d for d in fixed_points(tf, starts=STARTS) if d['stable']]; got = [d for d in fixed_points(mf, starts=STARTS) if d['stable']]
    if len(ref) != len(got): return dict(n_ref=len(ref), n_model=len(got), max_rel_err=None)
    return dict(n_ref=len(ref), n_model=len(got), max_rel_err=float(max(min(np.linalg.norm(g['x'] - r['x']) / max(np.linalg.norm(r['x']), 1e-9) for g in got) for r in ref)))


def traj(rhs, X0, t):
    out = []
    for x0 in X0:
        s = solve_ivp(lambda _, x: rhs(x), (t[0], t[-1]), x0, t_eval=t, rtol=1e-8, atol=1e-10, method='LSODA'); out.append(s.y.T if s.y.shape[1] == len(t) else np.full((len(t), 2), 1e3))
    return np.stack(out)


for name in args.systems:
    s = C['systems'][name]; kind = s['kind']; p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ('role', 'kind', 'ic_high')}
    truth = lambda X: field_gen(X, p, kind)
    GUARD = assert_bistable(lambda x: truth(x), f'systeme {name}')   # arret avant entrainement
    for seed in args.seeds:
        t0 = time.time(); t = np.linspace(0, D['T'], D['n_times']); X0, X, V = make_dataset_gen(seed, 60, t, 0.0, s['ic_high'], p, kind)
        tr, va, te = slice(0, 40), slice(40, 50), slice(50, None)
        f, info = train_node(t, X[tr], X[va], N, seed, log=lambda m: None)
        Xtr, Xte = X[tr].reshape(-1, 2), X[te].reshape(-1, 2); Fn_tr, Fn_te, Ft_te = field_np(f, Xtr), field_np(f, Xte), truth(Xte)
        r = dict(system=name, kind=kind, seed=seed, node_field_nrmse=nrmse(Fn_te, Ft_te), node_traj_nrmse=float(np.sqrt(np.mean((predict(f, X0[te], t, N) - X[te]) ** 2)) / rng_(X[te])))
        r['node_ok'] = bool(r['node_traj_nrmse'] <= 0.10)
        search = [structured_search(Xtr, Fn_tr, i, LIB['K_grid'], LIB['n_grid'])[0] for i in (0, 1)]
        r['reg_sign_ok'] = bool(all(search[i]['reg'] == 1 - i and search[i]['type'] == 'rep' for i in (0, 1)))
        fits = []
        for i in (0, 1):
            q = {fm: fit_family(fm, Xtr, Fn_tr, i, search[i]['reg'], search[i]['type'], search[i]['a'], search[i]['K'], search[i]['g']) for fm in ('hill', 'leaky', 'gauss')}; fits.append({k: v for k, v in q.items() if v})
        hill = [dict(fam='hill', reg=search[i]['reg'], typ=search[i]['type'], p=fits[i]['hill']['p']) for i in (0, 1)]; rh = make_rhs_fam(hill)
        r['hill_only'] = dict(nrmse_sym_to_node=nrmse(rh(Xte), Fn_te), nrmse_sym_to_truth=nrmse(rh(Xte), Ft_te), traj_nrmse=float(np.sqrt(np.mean((traj(rh, X0[te], t) - X[te]) ** 2)) / rng_(X[te])), fixed_points=fpcmp(truth, lambda x: rh(np.asarray(x))))
        r['rule_T_flag'] = bool(r['hill_only']['traj_nrmse'] > 0.10); r['rule_F_flag'] = bool(r['hill_only']['nrmse_sym_to_node'] > 0.03)
        gsel, bsel, ratios = [], [], []
        for i in (0, 1):
            fh = fits[i]['hill']; gains = {fm: fh['rss'] / fits[i][fm]['rss'] for fm in ('leaky', 'gauss') if fm in fits[i]}
            best = max(gains, key=gains.get); ratios.append({k: float(v) for k, v in gains.items()})
            gsel.append(best if gains[best] > THETA else 'hill'); bsel.append(min(fits[i], key=lambda k: fits[i][k]['bic']))
        r['rule_G_choice'] = gsel; r['rule_B_choice'] = bsel; r['rss_gain_ratios'] = ratios
        eq = [dict(fam=gsel[i], reg=search[i]['reg'], typ=search[i]['type'], p=fits[i][gsel[i]]['p']) for i in (0, 1)]; rg = make_rhs_fam(eq)
        r['after_G'] = dict(nrmse_sym_to_truth=nrmse(rg(Xte), Ft_te), traj_nrmse=float(np.sqrt(np.mean((traj(rg, X0[te], t) - X[te]) ** 2)) / rng_(X[te])), fixed_points=fpcmp(truth, lambda x: rg(np.asarray(x))))
        r['seconds'] = time.time() - t0; json.dump(r, open(f'{OUT}/detect_{name}_seed{seed}.json', 'w'), indent=1, default=float)
        print(f"{name}({kind}) s{seed}: NODE traj={r['node_traj_nrmse']:.3f} | Hill seul traj={r['hill_only']['traj_nrmse']:.3f} T={r['rule_T_flag']} F={r['rule_F_flag']} | G={gsel} B={bsel} gains={[{k:round(v,2) for k,v in d.items()} for d in ratios]} | apres G traj={r['after_G']['traj_nrmse']:.3f} | {r['seconds']:.0f}s", flush=True)
print('FIN')
