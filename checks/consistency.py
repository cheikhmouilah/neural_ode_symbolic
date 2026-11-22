#!/usr/bin/env python3
"""Controles de coherence des resultats (chapitre, section 10) : un resultat qui s'ameliore quand le bruit ou le dropout augmentent, une metrique constante sur toutes les
configurations, ou un controle qui ne discrimine pas, doivent arreter la chaine AVANT toute conclusion.
   python checks/consistency.py [--synth DIR] [--real DIR] [--toggle DIR] [--strict]   ; code 1 si un controle echoue (avec --strict, un controle non evaluable echoue aussi)"""
import os, sys, json, glob, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--synth', default=os.path.join(ROOT, RESULT_DIRS['synthetic'])); ap.add_argument('--real', default=os.path.join(ROOT, RESULT_DIRS['real']))
ap.add_argument('--toggle', default=os.path.join(ROOT, RESULT_DIRS['toggle'])); ap.add_argument('--strict', action='store_true'); a = ap.parse_args()
res = []; MIN_SEEDS = 3
def add(name, status, detail): res.append(dict(check=name, status=status, detail=detail)); print(f'[{status:^12s}] {name} : {detail}')
S = {}
for f in glob.glob(f'{a.synth}/synth_*_seed*.json'):
    try: r = load_json(f); S.setdefault(r['variant'], []).append(r)
    except Exception as e: add('lecture ' + os.path.basename(f), 'ECHEC', str(e)[:80])
mean = lambda L, g: float(np.nanmean([g(r) for r in L]))
nodecos = lambda r: r['node_metrics']['cos_vs_truth_support']; f1 = lambda r: r['distilled']['structure']['F1']
def monotone(name, variants, levels, metric, tol, label):
    L = [(lv, S[v]) for v, lv in zip(variants, levels) if v in S and len(S[v]) >= MIN_SEEDS]
    if len(L) < 2: add(name, 'NON EVALUE', f'moins de 2 niveaux avec >= {MIN_SEEDS} graines'); return
    vals = [mean(R, metric) for _, R in L]; worst = max([vals[i + 1] - vals[i] for i in range(len(vals) - 1)])
    add(name, 'ECHEC' if worst > tol else 'OK', f"{label} : valeurs {[round(v, 3) for v in vals]} ; amelioration maximale avec la degradation = {worst:+.3f} (tolerance {tol})")
monotone('bruit -> cosinus NODE', ['noise02', 'base', 'noise10', 'noise20'], [0.02, 0.05, 0.10, 0.20], nodecos, 0.03, 'cosinus NODE/verite')
monotone('bruit -> F1 distille', ['noise02', 'base', 'noise10', 'noise20'], [0.02, 0.05, 0.10, 0.20], f1, 0.15, 'F1 de structure')
monotone('dropout -> cosinus NODE', ['base', 'drop10', 'drop20'], [0, 0.1, 0.2], nodecos, 0.03, 'cosinus NODE/verite')
monotone('dropout -> F1 distille', ['base', 'drop10', 'drop20'], [0, 0.1, 0.2], f1, 0.15, 'F1 de structure')
if all(v in S and len(S[v]) >= MIN_SEEDS for v in ('N20', 'N120')):
    d = mean(S['N120'], nodecos) - mean(S['N20'], nodecos); add('quantite de donnees -> cosinus NODE', 'ECHEC' if d < -0.03 else 'OK', f'N120 - N20 = {d:+.3f} (une quantite de donnees 6 fois plus grande ne doit pas degrader le champ au-dela de 0,03)')
else: add('quantite de donnees -> cosinus NODE', 'NON EVALUE', 'N20 et N120 requis avec >= 3 graines')
allruns = [r for L in S.values() for r in L]
if len(allruns) >= 5:
    paths = {'cosinus NODE': nodecos, 'traj NODE': lambda r: r['node_metrics']['traj_nrmse_vs_truth'], 'F1 distille': f1, 'cosinus distille': lambda r: r['distilled']['cos_vs_truth_support'], 'F1 direct': lambda r: r['direct_fd']['structure']['F1'], 'F1 forme faible': lambda r: r['weak_sindy']['structure']['F1']}
    for nm, g in paths.items():
        v = np.array([g(r) for r in allruns], float); v = v[~np.isnan(v)]
        add(f'invariance : {nm}', 'ECHEC' if len(v) >= 5 and np.ptp(v) < 1e-12 else 'OK', f'{len(v)} valeurs, etendue {np.ptp(v):.2e}' + (' : metrique constante sur toutes les configurations (valeur de repli ?)' if len(v) >= 5 and np.ptp(v) < 1e-12 else ''))
else: add('invariance (synthetique)', 'NON EVALUE', f'{len(allruns)} resultat(s) < 5')
# donnees reelles
main = sorted(glob.glob(f'{a.real}/real_none_seed*.json')); ctrl = sorted(glob.glob(f'{a.real}/real_shuffle_seed*.json')); sw = sorted(glob.glob(f'{a.real}/real_sweep_lam*.json'))
if len(main) >= MIN_SEEDS:
    c = [load_json(f)['node_cos_test'] for f in main]; add('invariance reelle : cosinus NODE entre graines', 'ECHEC' if np.ptp(c) < 1e-12 else 'OK', f'etendue {np.ptp(c):.3e} sur {len(c)} graines (une valeur identique signalerait un jeu simule deterministe ou un repli)')
else: add('invariance reelle (graines)', 'NON EVALUE', f'{len(main)} graine(s) < {MIN_SEEDS}')
if len(sw) >= 3:
    c = [load_json(f)['node_cos_test'] for f in sw]; add('invariance reelle : balayage lambda_reg', 'ECHEC' if np.ptp(c) < 1e-12 else 'OK', f'etendue {np.ptp(c):.3e} sur {len(c)} valeurs')
if main and ctrl:
    m_ = np.mean([load_json(f)['node_cos_test'] for f in main]); k_ = np.mean([load_json(f)['node_cos_test'] for f in ctrl])
    add('controle a vitesses melangees', 'OK' if m_ > k_ else 'ECHEC', f'cosinus principal {m_:.3f} contre controle {k_:.3f} (le controle ne doit pas egaler ou depasser le resultat principal)')
# toggle (chaine A) : le bruit ne doit pas ameliorer l'erreur de champ
T = {}
for tag in ('traj', 'n05', 'n10'):
    fs = glob.glob(f'{a.toggle}/phase1_{tag}_seed*.json')
    if len(fs) >= MIN_SEEDS: T[tag] = float(np.mean([load_json(f)['P1_node_field_nrmse_test'] for f in fs]))
if len(T) == 3:
    worst = max(T['n05'] - T['traj'], T['n10'] - T['n05']) * -1; ok = (T['n05'] >= T['traj'] - 0.005) and (T['n10'] >= T['n05'] - 0.005)
    add('toggle : bruit -> erreur de champ', 'OK' if ok else 'ECHEC', f"NRMSE champ moyen : sigma=0 {T['traj']:.4f} ; 0,05 {T['n05']:.4f} ; 0,10 {T['n10']:.4f} (ne doit pas diminuer de plus de 0,005 quand le bruit augmente)")
else: add('toggle : bruit -> erreur de champ', 'NON EVALUE', 'resultats traj, n05, n10 (>= 3 graines) requis')
fails = [r for r in res if r['status'] == 'ECHEC']; ne = [r for r in res if r['status'] == 'NON EVALUE']
json.dump(res, open(os.path.join(ROOT, 'consistency_report.json'), 'w'), indent=1)
print(f'\n{len(res)} controle(s) : {len(fails)} echec(s), {len(ne)} non evalue(s)')
sys.exit(1 if fails or (a.strict and ne) else 0)
