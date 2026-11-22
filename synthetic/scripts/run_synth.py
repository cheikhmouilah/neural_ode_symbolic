"""Benchmark synthetique multi-genes. Un appel = une variante x une graine. Seuils fixes dans ../configs/synth.json avant execution."""
import os, sys, json, time, argparse, hashlib, copy, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT + '/src')
from synth_system import gen_system, field_true, simulate, make_data, zhi, sample_ics
from synth_methods import *
from real_identify import describe
ap = argparse.ArgumentParser(); ap.add_argument('--variant', default='base'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--free-form', choices=['none', 'gplearn', 'pysr'], default='none'); ap.add_argument('--tag', default='synth'); ap.add_argument('--quick', action='store_true', help='test de fonctionnement : peu d epoques, petite bibliotheque ; resultats dans results/synth_quick'); ap.add_argument('--force', action='store_true')
args = ap.parse_args()
CP = ROOT + '/configs/synth.json'; C = json.load(open(CP)); sha = hashlib.sha256(open(CP, 'rb').read()).hexdigest()[:16]
def dmerge(a, b):
    for k, v in b.items(): a[k] = dmerge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a
cfg = dmerge(copy.deepcopy(C['base']), C['variants'][args.variant]); lib = cfg['library']
if args.quick: cfg['node'].update(epochs=60); cfg['n_traj'] = min(cfg['n_traj'], 24); cfg['distillation'].update(M=1000, n_fine=20); lib.update(k_max=2, n_starts=2, K_grid=lib['K_grid'][::2], n_grid=[1.0, 2.0, 3.0]); cfg['free_form'].update(population=60, generations=4, pysr_iterations=2); cfg['d'] = min(cfg['d'], 4)
OUT = ROOT + ('/results/synth_quick' if args.quick else '/results/synth'); os.makedirs(OUT, exist_ok=True); name = f'{OUT}/{args.tag}_{args.variant}_seed{args.seed}.json'
if os.path.exists(name) and not args.force: print('[saute]', name); sys.exit(0)
t0 = time.time(); T_ = {}; seed = args.seed; rng = np.random.default_rng(seed)
eqs = gen_system(cfg['d'], seed); D = make_data(eqs, cfg, 1000 + seed); X, t_obs, t_true, Ztrue = D['X'], D['t_obs'], D['t_true'], D['Ztrue']; d = cfg['d']
perm = rng.permutation(len(X)); nv = max(2, int(cfg['node']['val_fraction'] * len(X))); va, tr = perm[:nv], perm[nv:]
tn = time.time(); nbd = dict(Ytr=D['Yc'][tr], Str=D['S'][tr], Yva=D['Yc'][va], Sva=D['S'][va], kappa=D['kappa']) if cfg.get('obs_model') == 'nb' else None
f, info = train_traj_node(X[tr], X[va], t_obs, cfg['node'], seed, nb=nbd); T_['node_train'] = time.time() - tn
R = dict(variant=args.variant, seed=seed, config_sha=('QUICK-' + sha if args.quick else sha), cfg=cfg, truth=[dict(i=e['i'], g=e['g'], terms=e['terms']) for e in eqs], node=info, thresholds_note='voir configs/synth.json')
# ---- jeu de test : nouvelles conditions initiales (meme distribution), trajectoires vraies sans bruit ----
Z0te = sample_ics(eqs, cfg['n_test'], np.random.default_rng(5000 + seed), cfg['ic_mode']); Zte = simulate(eqs, Z0te, t_true); Zsup = Zte.reshape(-1, d)
Zbox = np.random.default_rng(6000 + seed).uniform(0, zhi(eqs), (2000, d)); Ftrue_sup, Ftrue_box = field_true(eqs, Zsup), field_true(eqs, Zbox)
Fn_sup, Fn_box = field_np(f, Zsup), field_np(f, Zbox)
Pn = integrate_node(f, Z0te, t_obs); R['node_metrics'] = dict(traj_nrmse_vs_truth=nrmse(Pn, Zte), cos_vs_truth_support=cosine(Fn_sup, Ftrue_sup), cos_vs_truth_offsupport=cosine(Fn_box, Ftrue_box),
                                                              speed_ratio_median=float(np.median(np.linalg.norm(Fn_sup, axis=1) / np.maximum(np.linalg.norm(Ftrue_sup, axis=1), 1e-9))))
# ---- distillation ----
td = time.time()
Zf = sample_Df(f, X[tr][:, 0], t_obs[-1], rng, cfg['distillation']); Ff = field_np(f, Zf); p = rng.permutation(len(Zf)); nvf = int(cfg['distillation']['val_fraction'] * len(Zf)); sv, st = p[:nvf], p[nvf:]
cand = build_candidates(Zf[st], lib['K_grid'], lib['n_grid']); models = []
for i in range(d): ch, _ = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, lib, rng, cand=cand); models.append(ch['model'])
def Fsym(Z): return predict_all(models, np.atleast_2d(Z))
T_['distill_sample_select'] = time.time() - td
Fs_sup, Fs_box = Fsym(Zsup), Fsym(Zbox)
R['distilled'] = dict(equations=describe(models, [f'z{i}' for i in range(d)]), structure=structure_metrics(models, eqs), cos_vs_node_support=cosine(Fs_sup, Fn_sup), cos_vs_truth_support=cosine(Fs_sup, Ftrue_sup), cos_vs_truth_offsupport=cosine(Fs_box, Ftrue_box),
                      nrmse_vs_truth_support=nrmse(Fs_sup, Ftrue_sup), speed_ratio_median=float(np.median(np.linalg.norm(Fs_sup, axis=1) / np.maximum(np.linalg.norm(Ftrue_sup, axis=1), 1e-9))))
traj_err = []
for r in range(len(Z0te)):
    from scipy.integrate import solve_ivp
    s = solve_ivp(lambda tt, x: Fsym(x[None])[0], (t_obs[0], t_obs[-1]), Z0te[r], t_eval=t_obs, method='LSODA', rtol=1e-8, atol=1e-10)
    if s.y.shape[1] == len(t_obs): traj_err.append(s.y.T)
R['distilled']['traj_nrmse_vs_truth'] = nrmse(np.stack(traj_err), Zte[:len(traj_err)]) if traj_err else None
if cfg['time_mode'] == 'true': R['prop1'] = prop1_check(lambda Z: field_true(eqs, Z), Fsym, Zte, t_true, Z0te)
else: R['prop1'] = None        # en pseudotemps, le champ appris n'est pas dans les memes unites que le champ vrai
# ---- bases de reference ----
tb = time.time()
Xs, ts = X, t_obs
dX = fd_targets(Xs, ts); Zd_tr, Yd_tr = Xs[tr].reshape(-1, d), dX[tr].reshape(-1, d); Zd_va, Yd_va = Xs[va].reshape(-1, d), dX[va].reshape(-1, d); cand_d = build_candidates(Zd_tr, lib['K_grid'], lib['n_grid']); dm = []
for i in range(d): ch, _ = select_structure(Zd_tr, Yd_tr[:, i], Zd_va, Yd_va[:, i], i, lib, rng, cand=cand_d); dm.append(ch['model'])
Fd_sup = predict_all(dm, Zsup); R['direct_fd'] = dict(structure=structure_metrics(dm, eqs), cos_vs_truth_support=cosine(Fd_sup, Ftrue_sup), cos_vs_truth_offsupport=cosine(predict_all(dm, Zbox), Ftrue_box))
meta, Phi_all = build_candidates(Xs.reshape(-1, d), lib['K_grid'], lib['n_grid']); Phi3 = Phi_all.reshape(len(Xs), Xs.shape[1], -1); wk = cfg['weak']
Atr, Dtr, Ytr = weak_arrays(Xs[tr], ts, wk['window'], wk['stride'], Phi3[tr]) ; Ava, Dva, Yva = weak_arrays(Xs[va], ts, wk['window'], wk['stride'], Phi3[va])
wm = [weak_select(Atr, Dtr[:, i], Ytr[:, i], Ava, Dva[:, i], Yva[:, i], i, meta, lib['k_max'], lib['selection_tolerance']) for i in range(d)]
R['weak_sindy'] = dict(structure=structure_metrics(wm, eqs), cos_vs_truth_support=cosine(predict_all(wm, Zsup), Ftrue_sup), cos_vs_truth_offsupport=cosine(predict_all(wm, Zbox), Ftrue_box))
T_['baselines_direct_weak'] = time.time() - tb
om = []
for e in eqs:
    m0 = dict(i=e['i'], terms=[dict(kind='hill', j=t['j'], typ=t['typ'], a=1.0, K=0.7, n=2.0) for t in e['terms']], g=1.0); mr, _, _ = refine(Zf[st], Ff[st, e['i']], m0, lib, rng); om.append(mr)
R['oracle_structure'] = dict(structure=structure_metrics(om, eqs), cos_vs_truth_support=cosine(predict_all(om, Zsup), Ftrue_sup), cos_vs_node_support=cosine(predict_all(om, Zsup), Fn_sup), cos_vs_truth_offsupport=cosine(predict_all(om, Zbox), Ftrue_box))
tp = time.time(); R['profile'] = profile_coverage(models, eqs, Zf[st], Ff[st], lib); T_['profile_likelihood'] = time.time() - tp
tk = time.time(); n_ko = min(8, len(Z0te)); R['ko'] = ko_eval(lambda Z: field_true(eqs, Z), Fsym, lambda Z: field_np(f, Z), Z0te[:n_ko], cfg['horizon']) if cfg['time_mode'] == 'true' else None; T_['ko_eval'] = time.time() - tk
if args.free_form == 'gplearn': R['free_form_gplearn'] = free_form_gplearn(Zf[st], Ff[st], Zsup, Fn_sup, eqs, cfg['free_form'], seed)
elif args.free_form == 'pysr':
    sys.path.insert(0, ROOT + '/src')
    try:
        from pysr_free_form import free_form_pysr; R['free_form_pysr'] = free_form_pysr(Zf[st], Ff[st], Zsup, Fn_sup, eqs, cfg['free_form'], seed)
    except Exception as e: R['free_form_pysr'] = dict(skipped=str(e)[:200])
R['seconds'] = time.time() - t0; R['timings'] = T_
json.dump(R, open(name, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
s = R['distilled']['structure']
pr = R['profile']; ko = R['ko']
print(f"{args.variant} seed {seed}: profilK cov={(pr and pr['K']['coverage'] is not None and round(pr['K']['coverage'],2))} KO cos sym={(ko and round(ko['cos_sym'],2))} node={(ko and round(ko['cos_node'],2))} | NODE traj={R['node_metrics']['traj_nrmse_vs_truth']:.3f} cos={R['node_metrics']['cos_vs_truth_support']:.3f} | distilled F1={s['F1']:.2f} (P {s['precision']:.2f} R {s['recall']:.2f}) cos_truth={R['distilled']['cos_vs_truth_support']:.3f} | direct F1={R['direct_fd']['structure']['F1']:.2f} weak F1={R['weak_sindy']['structure']['F1']:.2f} oracle relerr={R['oracle_structure']['structure']['param_relerr_all_median']} | prop1={R['prop1'] and round(R['prop1']['holds_fraction'],2)} | {R['seconds']:.0f}s", flush=True)
if args.free_form == 'pysr': sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # evite un blocage a la sortie du processus Julia
