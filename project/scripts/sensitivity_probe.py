"""Sonde de sensibilite de l'entrainement : perturbation relative des poids initiaux (defaut 1e-10) ; on compare la perte de validation, l'erreur de champ et l'erreur de trajectoire
avec et sans perturbation, pour plusieurs unites (systeme x graine). Un ecart relatif eleve = unite sensible : rapporter des fourchettes, pas un chiffre unique.
   python scripts/sensitivity_probe.py [--systems C2 M1 ...] [--seeds 0] [--amp 1e-10]   (un entrainement ~ 1,5 min par essai ; 2 essais par unite)"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time, argparse, numpy as np
sys.path.insert(0, ROOT + '/src')
from misspec2 import field_gen, make_dataset_gen
from node import train_node, field_np, predict
ap = argparse.ArgumentParser(); ap.add_argument('--systems', nargs='+', default=['C2', 'M1']); ap.add_argument('--seeds', type=int, nargs='+', default=[0]); ap.add_argument('--amp', type=float, default=1e-10); args = ap.parse_args()
C = json.load(open(ROOT + '/configs/detect_test.json')); P1 = json.load(open(ROOT + '/configs/phase1.json')); D, N = P1['data'], P1['node']
rng_ = lambda a: float(a.max() - a.min()); res = []
for name in args.systems:
    s = C['systems'][name]; kind = s['kind']; p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ('role', 'kind', 'ic_high')}
    for seed in args.seeds:
        t = np.linspace(0, D['T'], D['n_times']); X0, X, V = make_dataset_gen(seed, 60, t, 0.0, s['ic_high'], p, kind); tr, va, te = slice(0, 40), slice(40, 50), slice(50, None); row = dict(system=name, seed=seed, amp=args.amp)
        for tag, pr in (('nominal', 0.0), ('perturbe', args.amp)):
            f, info = train_node(t, X[tr], X[va], N, seed, log=lambda m: None, perturb_rel=pr); Xte = X[te].reshape(-1, 2); Ft = field_gen(Xte, p, kind)
            row[tag] = dict(best_val=float(info['best_val']) if 'best_val' in info else None, field_nrmse=float(np.sqrt(np.mean((field_np(f, Xte) - Ft) ** 2)) / rng_(Ft)), traj_nrmse=float(np.sqrt(np.mean((predict(f, X0[te], t, N) - X[te]) ** 2)) / rng_(X[te])))
        for k in ('field_nrmse', 'traj_nrmse'): row['rel_diff_' + k] = abs(row['perturbe'][k] - row['nominal'][k]) / max(row['nominal'][k], 1e-12)
        res.append(row); print(f"{name} seed {seed} amp {args.amp:g} : ecart relatif champ {row['rel_diff_field_nrmse']:.2e} ; trajectoire {row['rel_diff_traj_nrmse']:.2e}", flush=True)
_os.makedirs(ROOT + '/results/phase1', exist_ok=True); json.dump(res, open(ROOT + '/results/phase1/sensitivity_probe.json', 'w'), indent=1)
