#!/usr/bin/env python3
"""Cout de calcul par etape (partie theorique VI-D) a partir des durees enregistrees dans les resultats. Ecrit figures_article/tab_cost.csv/.tex.
   python checks/analyze_cost.py [--synth DIR] [--real DIR] [--out DIR]"""
import os, sys, json, glob, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--synth', default=os.path.join(ROOT, RESULT_DIRS['synthetic'])); ap.add_argument('--real', default=os.path.join(ROOT, RESULT_DIRS['real'])); ap.add_argument('--out', default=os.path.join(ROOT, 'figures_article')); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); rows = []
S = {}
for f in glob.glob(f'{a.synth}/synth_*_seed*.json'):
    r = load_json(f)
    if 'timings' in r: S.setdefault(r['variant'], []).append(r)
stages = ['node_train', 'distill_sample_select', 'baselines_direct_weak', 'profile_likelihood', 'ko_eval']
for v, L in sorted(S.items()):
    rows.append([f'synthetique:{v}', len(L), L[0]['cfg']['d'], L[0]['cfg']['n_traj']] + [f"{np.mean([r['timings'].get(s, np.nan) for r in L]):.1f}" for s in stages] + [f"{np.mean([r['seconds'] for r in L]):.1f}"])
for tag in ('real', 'real_module', 'real_latent', 'real_dg'):
    L = [load_json(f) for f in glob.glob(f'{a.real}/{tag}_none_seed*.json')]
    if L: rows.append([f'reel:{tag}', len(L), len(L[0]['genes']), '-', f"{np.mean([r['node']['seconds'] for r in L]):.1f}", '-', '-', '-', '-', f"{np.mean([r['seconds_total'] for r in L]):.1f}"])
hdr = ['configuration', 'runs', 'd', 'n_traj', 's_NODE', 's_distillation', 's_bases', 's_profil', 's_KO', 's_total']
if not rows: sys.exit('aucun resultat avec durees')
with open(f'{a.out}/tab_cost.csv', 'w', encoding='utf-8') as f: f.write(','.join(hdr) + '\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')
with open(f'{a.out}/tab_cost.tex', 'w', encoding='utf-8') as f:
    f.write('\\begin{tabular}{l' + 'r' * (len(hdr) - 1) + '}\n\\hline\n' + ' & '.join(h.replace('_', '\\_') for h in hdr) + ' \\\\\n\\hline\n' + '\n'.join(' & '.join(str(x).replace('_', '\\_') for x in r) + ' \\\\' for r in rows) + '\n\\hline\n\\end{tabular}\n')
print('  '.join(f'{h:>14s}' for h in hdr)); [print('  '.join(f'{str(x):>14s}' for x in r)) for r in rows]
d = [(r['cfg']['d'] * r['cfg']['n_traj'], r['timings']['node_train']) for L in S.values() for r in L if 'node_train' in r['timings']]
if len(d) >= 4 and len({x for x, _ in d}) >= 3:
    x, y = np.log([a_ for a_, _ in d]), np.log([b_ for _, b_ in d]); sl = np.polyfit(x, y, 1)[0]; print(f'\npente log-log du temps d entrainement du NODE en fonction de d x n_traj : {sl:.2f} ({len(d)} runs ; indicatif)')
