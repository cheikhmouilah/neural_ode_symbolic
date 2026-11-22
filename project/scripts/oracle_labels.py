"""Etiquettes de nocivite calculees sur la VERITE SEULE (aucun entrainement) : le meilleur modele de Hill est ajuste sur le champ EXACT, puis compare au systeme vrai
sur 500 conditions initiales (erreur de bassin) et sur les points fixes stables (erreur relative maximale, 1,0 si le nombre d'etats stables differe).
Nocif si l'une des deux erreurs >= 0,10 ; benin si les deux <= 0,02 ; intermediaire sinon (exclu). Seuils : voir configs/detect_test.json (rules) ou ci-dessous.
   python scripts/oracle_labels.py [--systems C1 M1 ...]"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, argparse, numpy as np
sys.path.insert(0, ROOT + '/src')
from misspec2 import field_gen
from toggle import fixed_points
from identify import structured_search, refine, make_rhs
from scipy.integrate import solve_ivp
ap = argparse.ArgumentParser(); ap.add_argument('--systems', nargs='+', default=None); ap.add_argument('--n_ic', type=int, default=500); ap.add_argument('--harmful', type=float, default=0.10); ap.add_argument('--benign', type=float, default=0.02); args = ap.parse_args()
C = json.load(open(ROOT + '/configs/detect_test.json')); P1 = json.load(open(ROOT + '/configs/phase1.json')); LIB = P1['library']; STARTS = [(a, b) for a in np.linspace(0, 7, 15) for b in np.linspace(0, 7, 15)]


def attractor(f, x0, T=60.0):
    s = solve_ivp(lambda _, x: f(x), (0, T), x0, method='LSODA', rtol=1e-7, atol=1e-9); return s.y[:, -1] if s.success else np.full(2, np.nan)


out = {}
for name in (args.systems or list(C['systems'])):
    s = C['systems'][name]; kind = s['kind']; p = {k: (tuple(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ('role', 'kind', 'ic_high')}; truth = lambda X: field_gen(X, p, kind)
    rng = np.random.default_rng(0); hi = s['ic_high']; X = rng.uniform(0, hi, (4000, 2)); F = truth(X)         # champ EXACT sur la boite d'etude
    best = []
    for i_ in (0, 1):
        b_, _ = structured_search(X, F, i_, LIB['K_grid'], LIB['n_grid']); best.append(refine(X, F, i_, b_))
    rh = make_rhs(best)
    ic = rng.uniform(0, hi, (args.n_ic, 2)); ref = [d['x'] for d in fixed_points(lambda x: truth(x), starts=STARTS) if d['stable']]
    def basin(f): return np.array([np.argmin([np.linalg.norm(attractor(f, x0) - r) for r in ref]) for x0 in ic])
    b_true, b_model = basin(lambda x: truth(x)), basin(lambda x: rh(np.atleast_2d(x))[0])
    basin_err = float(np.mean(b_true != b_model)); got = [d['x'] for d in fixed_points(lambda x: rh(np.atleast_2d(x))[0], starts=STARTS) if d['stable']]
    fp_err = 1.0 if len(got) != len(ref) else float(max(min(np.linalg.norm(g - r) / max(np.linalg.norm(r), 1e-9) for g in got) for r in ref))
    label = 'nocif' if max(basin_err, fp_err) >= args.harmful else ('benin' if max(basin_err, fp_err) <= args.benign else 'intermediaire (exclu)')
    out[name] = dict(kind=kind, basin_error=basin_err, fixed_point_error=fp_err, label=label); print(f'{name:3s} ({kind:8s}) erreur de bassin {basin_err:.3f} ; erreur de points fixes {fp_err:.3f} -> {label}', flush=True)
_os.makedirs(ROOT + '/results/phase1', exist_ok=True); json.dump(out, open(ROOT + '/results/phase1/oracle_labels.json', 'w'), indent=1)
