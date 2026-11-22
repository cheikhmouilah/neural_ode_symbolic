#!/usr/bin/env python3
"""Resume des resultats PySR (toggle, synthetique, donnees reelles) : complexite, fidelite, variables utilisees. Ecrit figures_article/tab_pysr.csv.
   python checks/analyze_pysr.py"""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
out = os.path.join(ROOT, 'figures_article'); os.makedirs(out, exist_ok=True); rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'project/results/phase1/pysr_toggle_seed*.json'))):
    r = load_json(f); rows.append(['toggle', r['seed'], f"NRMSE champ/verite {r['field_nrmse_vs_truth']:.4f} (NODE {r['field_nrmse_node_vs_truth']:.4f})", f"R2 vs NODE {np.round(r['fidelity_r2_vs_node'], 3).tolist()}", 'complexites ' + str([q['complexity'] for q in r['equations']]), 'autre gene utilise : ' + str([q['uses_other_gene'] for q in r['equations']]), f"points fixes stables {r['stable_fixed_points']['n_model']}/{r['stable_fixed_points']['n_ref']}"])
for f in sorted(glob.glob(os.path.join(ROOT, 'synthetic/results/synth/synthpysr_*_seed*.json'))):
    r = load_json(f); p = r.get('free_form_pysr')
    if p and 'equations' in p: rows.append(['synthetique', r['seed'], f"rappel des variables {p['mean_var_recall']:.2f}", f"precision {p['mean_var_precision']:.2f}", f"longueur moyenne {p['mean_length']:.1f}", f"R2 vs NODE {p['mean_r2_vs_node']:.2f}", ''])
    elif p: rows.append(['synthetique', r['seed'], 'PySR ignore : ' + str(p.get('skipped')), '', '', '', ''])
for f in sorted(glob.glob(os.path.join(ROOT, 'real_data/results/real/real_pysr_seed*.json'))):
    r = load_json(f); rows.append(['reel', r['seed'], f"cos PySR~NODE {r['pysr_vs_node_cos_test']:.3f}", f"cos PySR~vitesse {r['pysr_vs_measured_cos_test']:.3f}", f"cos NODE~vitesse (memes genes) {r['node_vs_measured_cos_same_genes']:.3f}", f"{len(r['equations_fitted'])} equations", 'complexites ' + str([q['complexity'] for q in r['equations']])])
if not rows: sys.exit('aucun resultat PySR')
open(os.path.join(out, 'tab_pysr.csv'), 'w', encoding='utf-8').write('chaine,graine,mesure1,mesure2,mesure3,mesure4,mesure5\n' + '\n'.join(','.join('"' + str(x) + '"' for x in r) for r in rows) + '\n')
for r in rows: print(' | '.join(str(x) for x in r))

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    srcs = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'project/results/phase1/pysr_toggle_seed0.json'))): r = load_json(f); srcs.append(('toggle, graine 0', [(f"eq. {q['eq']}", q['pareto']) for q in r['equations']]))
    for f in sorted(glob.glob(os.path.join(ROOT, 'real_data/results/real/real_pysr_seed0.json'))): r = load_json(f); srcs.append(('pancreas, graine 0', [(q['gene'], q['pareto']) for q in r['equations']]))
    if srcs:
        fig, axs = plt.subplots(1, len(srcs), figsize=(4.2 * len(srcs), 3.2), squeeze=False)
        for ax, (title, curves) in zip(axs[0], srcs):
            for lab, par in curves:
                par = sorted(par, key=lambda d: d['complexity']); ax.step([d['complexity'] for d in par], [max(d['loss'], 1e-12) for d in par], where='post', marker='o', ms=3, label=lab)
            ax.set_yscale('log'); ax.set_xlabel('complexite'); ax.set_ylabel('perte (MSE)'); ax.set_title(title, fontsize=9); ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(); [fig.savefig(os.path.join(out, 'fig_pysr_pareto.' + e), dpi=200) for e in ('png', 'pdf')]; print('figure fig_pysr_pareto')
except Exception as e: print('figure Pareto non produite :', e)
