"""Distillation LIBRE par PySR du Neural ODE du toggle (chaine A) : meme systeme, memes donnees et meme Neural ODE que la Phase 1 (graine s). Les cibles sont les valeurs du champ du Neural ODE
aux etats d'entrainement ; PySR cherche des expressions SANS famille imposee (la Phase 1, elle, choisit un terme de Hill sur une grille). Resultat : front de Pareto precision/complexite par equation,
fidelite au champ vrai, trajectoires, points fixes. Necessite Julia et PySR : voir GUIDE_PYSR.md.
   python scripts/run_pysr_toggle.py --seeds 0 1 2 [--iterations 40] [--maxsize 20] [--ops basic|pow] [--quick]"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, json, time, argparse, re, numpy as np
sys.path.insert(0, ROOT + '/src')
from toggle import field, make_dataset, fixed_points
from node import train_node, field_np
from scipy.integrate import solve_ivp
ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[0]); PC = json.load(open(ROOT + '/configs/pysr.json'))['toggle']; ap.add_argument('--iterations', type=int, default=PC['iterations']); ap.add_argument('--maxsize', type=int, default=PC['maxsize'])
ap.add_argument('--ops', choices=['basic', 'pow'], default=PC['ops'], help="basic : + - * / ; pow : ajoute ^ (puissances non entieres, utile pour des exposants de Hill)"); ap.add_argument('--quick', action='store_true'); ap.add_argument('--tag', default='pysr'); args = ap.parse_args()
try: from pysr import PySRRegressor
except Exception as e: sys.exit(f"PySR indisponible ({e}). Voir GUIDE_PYSR.md : python setup_all.py --with-pysr")
P1 = json.load(open(ROOT + '/configs/phase1.json')); D, N = P1['data'], dict(P1['node'])
if args.quick: N['iters_per_stage'] = [10, 10, 10, 10]; args.iterations = min(args.iterations, 2)
OUT = ROOT + ('/results/pysr_quick' if args.quick else '/results/phase1'); _os.makedirs(OUT, exist_ok=True)
rng_ = lambda a: float(a.max() - a.min())
for seed in args.seeds:
    t0 = time.time(); print(f'=== PySR toggle, graine {seed} ===', flush=True)
    t = np.linspace(0, D['T'], D['n_times']); X0, X, V = make_dataset(seed, D['n_traj'], t, D['ic_low'], D['ic_high']); tr, va, te = slice(0, 40), slice(40, 50), slice(50, None)
    f, info = train_node(t, X[tr], X[va], N, seed, log=lambda m: None)
    Xs = X[tr].reshape(-1, 2); Fs = field_np(f, Xs); Xte = X[te].reshape(-1, 2); Ft = field(Xte); Fn = field_np(f, Xte)
    t1 = time.time(); models, rows = [], []
    for i in (0, 1):
        m = PySRRegressor(niterations=args.iterations, binary_operators=['+', '-', '*', '/'] + (['^'] if args.ops == 'pow' else []), unary_operators=[], maxsize=args.maxsize, random_state=seed * 10 + i,
                          deterministic=True, parallelism='serial', progress=False, verbosity=0, model_selection='best', temp_equation_file=False, output_directory=_os.path.join(ROOT, 'results', 'pysr_runs'))
        m.fit(Xs, Fs[:, i]); b = m.get_best(); models.append(m); expr = str(b['equation']); used = sorted({int(v) for v in re.findall(r'x(\d+)', expr)})
        rows.append(dict(eq=i, expression=expr, complexity=int(b['complexity']), loss=float(b['loss']), used_variables=used, uses_other_gene=bool((1 - i) in used), uses_own_gene=bool(i in used),
                         pareto=[dict(complexity=int(r['complexity']), loss=float(r['loss']), equation=str(r['equation'])) for _, r in m.equations_.iterrows()]))
    Fsym = lambda X_: np.stack([np.asarray(m.get_best()['lambda_format'](np.atleast_2d(X_))).ravel() for m in models], 1)
    Fp = Fsym(Xte); r = dict(seed=seed, tag=args.tag, iterations=args.iterations, maxsize=args.maxsize, ops=args.ops, node=dict(info) if isinstance(info, dict) else {}, equations=rows)
    r['field_nrmse_vs_truth'] = float(np.sqrt(np.mean((Fp - Ft) ** 2)) / rng_(Ft)); r['field_nrmse_node_vs_truth'] = float(np.sqrt(np.mean((Fn - Ft) ** 2)) / rng_(Ft))
    r['fidelity_r2_vs_node'] = [float(1 - np.sum((Fp[:, i] - Fn[:, i]) ** 2) / np.sum((Fn[:, i] - Fn[:, i].mean()) ** 2)) for i in (0, 1)]
    errs = []
    for x0 in X0[te]:
        s = solve_ivp(lambda _, x: Fsym(x[None])[0], (t[0], t[-1]), x0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
        if s.y.shape[1] == len(t): errs.append(s.y.T)
    r['traj_nrmse_vs_truth'] = float(np.sqrt(np.mean((np.stack(errs) - X[te][:len(errs)]) ** 2)) / rng_(X[te])) if errs else None
    ref = [d for d in fixed_points(lambda x: field(x)) if d['stable']]; got = [d for d in fixed_points(lambda x: Fsym(np.atleast_2d(x))[0]) if d['stable']]
    r['stable_fixed_points'] = dict(n_ref=len(ref), n_model=len(got), max_rel_err=(float(max(min(np.linalg.norm(g['x'] - q['x']) / np.linalg.norm(q['x']) for g in got) for q in ref)) if got else None))
    r['seconds'] = time.time() - t0; r['seconds_pysr'] = time.time() - t1
    json.dump(r, open(f'{OUT}/{args.tag}_toggle_seed{seed}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(f"graine {seed} : NRMSE champ PySR/verite {r['field_nrmse_vs_truth']:.4f} (NODE/verite {r['field_nrmse_node_vs_truth']:.4f}) ; R2 vs NODE {np.round(r['fidelity_r2_vs_node'], 3).tolist()} ; trajectoire {r['traj_nrmse_vs_truth']} ; points fixes {r['stable_fixed_points']}")
    for q in rows: print(f"   eq{q['eq']} (complexite {q['complexity']}) : {q['expression']}")
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # evite un blocage a la sortie (arret du processus Julia) apres la fin du calcul
