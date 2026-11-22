"""Tables d'analyse de la partie SERGIO a partir des JSON produits par validate.py / eiv_test_sergio.py. Usage : python analyze_sergio.py [--dir DOSSIER]"""
import os, sys, json, numpy as np
d = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(f'{d}/{n}'))
print("== Pipeline MLP + symbolique, 14 premieres cibles a regulateur unique (n vrai = 2)")
for k in ('clean', 'noisy_mild', 'noisy_severe'):
    try: s = J(f'result_{k}.json')['summary']
    except FileNotFoundError: continue
    print(f"  {k:13s} type correct {s['type_correct']:.2f} | n_hat median (act+rep melanges) {s['n_hat_median']:.2f} | NRMSE mediane famille {s['median_nrmse_famille']:.3f}, gp {s['median_nrmse_gp']:.3f}")
try:
    r = J('result_clean.json')['rows']
    for t, nm in (('hill_act', 'activations'), ('hill_rep', 'repressions')):
        R = [x for x in r if x['true_type'] == t]; n = [x['family_params'][2] for x in R]; K = [x['family_params'][0] / x['K_true'] for x in R]
        print(f"  clean {nm}: n_hat med {np.median(n):.2f} [{min(n):.2f}-{max(n):.2f}] | K_hat/K med {np.median(K):.2f}")
except FileNotFoundError: pass
try:
    c = J('result_control.json'); print(f"== Controle negatif : R2 median vrai regulateur {np.median(c['real']):.3f} vs non lie {np.median(c['ctrl']):.3f}")
except FileNotFoundError: pass
try: print("== Test plage d'expression (n_hat median) :", {k: round(v['n_hat_median'], 2) for k, v in J('result_identifiability.json').items()})
except FileNotFoundError: pass
try: print("== EIV synthetique (CV -> n_hat q25/med/q75) :", {k: [round(x, 2) for x in v] for k, v in J('result_eiv.json').items()})
except FileNotFoundError: pass
try:
    ss = J('result_steady_state_diag.json'); print("== Quasi-stationnaire :", {k: (round(v['R2_median'], 3), round(v['biais_relatif_max_abs'], 3)) for k, v in ss.items()}, "(R2 median, biais max par gene)")
except FileNotFoundError: pass
try:
    R = J('result_eiv_sergio.json'); conds = ['raw', 'knn5', 'knn15', 'knn50', 'knn150', 'binmean']
    print(f"\n== Test EIV sur SERGIO ({len(R)} cibles, {len({(r['r'], r['type']) for r in R})} couples regulateur-type)")
    for typ in ('act', 'rep'):
        S = [r for r in R if r['type'] == typ]; print(f"  {typ} (n={len(S)}): n_hat median par condition :", {c: round(float(np.median([r[c]['n'] for r in S if r[c]])), 2) for c in conds})
    print("  par regulateur (n_hat median raw / knn50 / knn150) :")
    for reg, typ in sorted({(r['r'], r['type']) for r in R}):
        S = [r for r in R if r['r'] == reg and r['type'] == typ]; m = lambda c: np.median([r[c]['n'] for r in S])
        print(f"    reg {reg:3d} {typ}: {m('raw'):.2f} / {m('knn50'):.2f} / {m('knn150'):.2f}  (CV intra-type {np.median([r['cv_within_bin'] for r in S]):.2f})")
    for k in (15, 50):
        g = np.array([r[f'gain_r2_real_knn{k}'] for r in R]); cc = np.array([r[f'gain_r2_ctrl_knn{k}'] for r in R])
        print(f"  gain de R2 knn{k} : vrai {np.median(g):+.3f} vs controle {np.median(cc):+.3f} ; vrai > controle pour {np.mean(g>cc)*100:.0f}% des cibles")
except FileNotFoundError: pass
