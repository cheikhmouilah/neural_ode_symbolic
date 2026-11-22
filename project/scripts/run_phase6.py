import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
"""Phase 1 : toggle switch, Neural ODE -> distillation -> identification. Seuils preenregistres dans configs/phase1.json."""
import sys, json, time, argparse, numpy as np
sys.path.insert(0, ROOT + '/src')
from toggle import P_TRUE, field, make_dataset, fixed_points
from guards import assert_bistable
from node import train_node, field_np, predict
from identify import structured_search, refine, sindy_support, make_rhs
from scipy.integrate import solve_ivp

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); ap.add_argument('--velocity', type=float, default=0.0)
ap.add_argument('--tag', default='traj'); ap.add_argument('--noise', type=float, default=0.0); ap.add_argument('--ntrain', type=int, default=40); args = ap.parse_args()
cfg = json.load(open(ROOT + '/configs/phase1.json')); D, N, L = cfg['data'], cfg['node'], cfg['library']; N['velocity_weight'] = args.velocity
OUT = ROOT + '/results/phase1'; _os.makedirs(OUT, exist_ok=True)
TRUTH = [dict(reg=1, type='rep', a=P_TRUE['a'][0], K=P_TRUE['K'][0], n=P_TRUE['n'], g=P_TRUE['g'][0]),
         dict(reg=0, type='rep', a=P_TRUE['a'][1], K=P_TRUE['K'][1], n=P_TRUE['n'], g=P_TRUE['g'][1])]
rng_ = lambda a: float(a.max() - a.min())
nrmse_f = lambda Fp, Ft: float(np.sqrt(np.mean((Fp - Ft) ** 2)) / rng_(Ft))
cosine = lambda Fp, Ft: float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) for a, b in zip(Fp, Ft) if np.linalg.norm(b) > 1e-3 and np.linalg.norm(a) > 1e-12]))


def relerr(par):
    return [{k: abs(par[i][k] - TRUTH[i][k]) / TRUTH[i][k] for k in ('a', 'K', 'n', 'g')} for i in (0, 1)]


def identify(Xs, Fs):
    out = dict(search=[], refined=[], top=[])
    for i in (0, 1):
        b, top = structured_search(Xs, Fs, i, L['K_grid'], L['n_grid']); out['search'].append(b)
        out['refined'].append(refine(Xs, Fs, i, b)); out['top'].append([dict(reg=c['reg'], type=c['type'], K=c['K'], n=c['n'], rss=c['rss']) for c in top])
    out['structure_exact'] = bool(all(out['refined'][i]['reg'] == TRUTH[i]['reg'] and out['refined'][i]['type'] == 'rep' for i in (0, 1)))
    out['rel_err'] = relerr(out['refined']); out['max_rel_err'] = float(max(v for e in out['rel_err'] for v in e.values()))
    out['median_rel_err'] = float(np.median([v for e in out['rel_err'] for v in e.values()]))
    return out


def stlsq_report(Xs, Fs):
    rep = []
    for thr in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        sup = sindy_support(Xs, Fs, L['K_grid'], L['n_grid'], threshold=thr)
        row = dict(threshold=thr, n_terms=[len(s) for s in sup], exact=[], extra=[])
        for i, s in enumerate(sup):
            allowed = lambda k: k == f'x{i+1}' or (k.startswith(f'rep(x{TRUTH[i]["reg"]+1};'))
            row['exact'].append(bool(f'x{i+1}' in s and any(k.startswith('rep') for k in s) and all(allowed(k) for k in s)))
            row['extra'].append(sorted([k for k in s if not allowed(k)])[:4])
        rep.append(row)
    return rep


def sim_params(par, X0, t):
    rhs = make_rhs(par); out = []
    for x0 in X0:
        s = solve_ivp(lambda _, x: rhs(x), (t[0], t[-1]), x0, t_eval=t, rtol=1e-8, atol=1e-10, method='LSODA'); out.append(s.y.T)
    return np.stack(out)


def fp_compare(fun_model):
    ref = [d for d in fixed_points(lambda x: field(x)) if d['stable']]; got = [d for d in fixed_points(fun_model) if d['stable']]
    if len(ref) != len(got): return dict(n_ref=len(ref), n_model=len(got), max_rel_err=None, ok=False)
    errs = [min(np.linalg.norm(g['x'] - r['x']) / np.linalg.norm(r['x']) for g in got) for r in ref]
    return dict(n_ref=len(ref), n_model=len(got), max_rel_err=float(max(errs)), ok=bool(max(errs) < cfg['preregistered_thresholds']['P5_stable_fixed_points_rel_error']))


GUARD = assert_bistable(lambda x: field(x), 'toggle standard (P_TRUE)')
res = []
for seed in args.seeds:
    print(f"=== seed {seed} ({args.tag}) ===", flush=True); t0 = time.time()
    t = np.linspace(0, D['T'], D['n_times']); X0, X, V = make_dataset(seed, D['n_traj'], t, D['ic_low'], D['ic_high'])
    tr, va, te = slice(0, args.ntrain), slice(40, 50), slice(50, None)
    Xobs = X.copy(); Xobs[:50] += args.noise * np.random.default_rng(5000 + seed).normal(size=Xobs[:50].shape)
    f, info = train_node(t, Xobs[tr], Xobs[va], N, seed, Vtr=V[tr], log=lambda s: print(s, flush=True))
    Xtr, Xte = Xobs[tr].reshape(-1, 2), X[te].reshape(-1, 2)
    Ftr_node, Fte_node, Fte_true = field_np(f, Xtr), field_np(f, Xte), field(Xte)
    r = dict(seed=seed, tag=args.tag, node_seconds=info['seconds'], node_best_val=info['best_val'])
    r['P1_node_field_nrmse_test'] = nrmse_f(Fte_node, Fte_true); r['node_field_cosine_test'] = cosine(Fte_node, Fte_true)
    # trajectoires NODE
    Xp = predict(f, X0[te], t, N); r['node_traj_nrmse_test'] = float(np.sqrt(np.mean((Xp - X[te]) ** 2)) / rng_(X[te]))
    # distillation : D_f = (etats d'entrainement, champ du NODE)
    r['distilled'] = identify(Xtr, Ftr_node)
    # reference : identification directe sur derivees exactes (borne haute de l'identification)
    r['direct_exact_derivatives'] = identify(Xtr, field(Xtr))
    dt = t[1] - t[0]; Xi = Xobs[tr]; Xfd = Xi[:, 1:-1].reshape(-1, 2); Ffd = ((Xi[:, 2:] - Xi[:, :-2]) / (2 * dt)).reshape(-1, 2)
    r['direct_finite_differences'] = identify(Xfd, Ffd); r['noise'] = args.noise; r['n_train'] = args.ntrain
    r['stlsq_baseline_on_distilled'] = stlsq_report(Xtr, Ftr_node)
    par = r['distilled']['refined']; rhs = make_rhs(par)
    r['field_nrmse_symbolic_vs_truth_test'] = nrmse_f(rhs(Xte), Fte_true)
    r['field_nrmse_symbolic_vs_node_test'] = nrmse_f(rhs(Xte), Fte_node)          # objectif 1 : fidelite de distillation
    r['symbolic_cosine_test'] = cosine(rhs(Xte), Fte_true)
    Xs_ = sim_params(par, X0[te], t); r['P4_symbolic_traj_nrmse_test'] = float(np.sqrt(np.mean((Xs_ - X[te]) ** 2)) / rng_(X[te]))
    r['P5_fixed_points_symbolic'] = fp_compare(lambda x: rhs(np.asarray(x)))
    r['fixed_points_node'] = fp_compare(lambda x: field_np(f, np.asarray(x)[None])[0])
    # --- variante EXPLORATOIRE (post hoc, apres echec de P3b) : D_f echantillonne uniformement dans la boite supportee ---
    box = np.random.default_rng(1000 + seed).uniform(D['ic_low'], D['ic_high'], size=(2000, 2))
    r['distilled_box'] = identify(box, field_np(f, box)); parb = r['distilled_box']['refined']; rhsb = make_rhs(parb)
    r['box_field_nrmse_symbolic_vs_truth_test'] = nrmse_f(rhsb(Xte), Fte_true)
    Xb_ = sim_params(parb, X0[te], t); r['box_traj_nrmse_test'] = float(np.sqrt(np.mean((Xb_ - X[te]) ** 2)) / rng_(X[te]))
    r['box_fixed_points'] = fp_compare(lambda x: rhsb(np.asarray(x)))
    r['seconds_total'] = time.time() - t0; res.append(r)
    json.dump(r, open(f'{OUT}/phase1_{args.tag}_seed{seed}.json', 'w'), indent=1, default=float)
    print(f"seed {seed}: P1 field NRMSE={r['P1_node_field_nrmse_test']:.4f} | structure exacte={r['distilled']['structure_exact']} | param rel err med/max={r['distilled']['median_rel_err']:.3f}/{r['distilled']['max_rel_err']:.3f} | P4 traj={r['P4_symbolic_traj_nrmse_test']:.4f} | FD direct err med/max={r['direct_finite_differences']['median_rel_err']:.3f}/{r['direct_finite_differences']['max_rel_err']:.3f} struct={r['direct_finite_differences']['structure_exact']} | P5 ok={r['P5_fixed_points_symbolic']['ok']} | BOX err med/max={r['distilled_box']['median_rel_err']:.3f}/{r['distilled_box']['max_rel_err']:.3f} | {r['seconds_total']:.0f}s", flush=True)
print('FIN')
