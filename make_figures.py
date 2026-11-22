#!/usr/bin/env python3
"""Figures (PNG + PDF) et tableaux (CSV + LaTeX) de l'article a partir des JSON de resultats. Ne recalcule rien ; saute ce qui manque.
   python make_figures.py [--real-dir real_data/results/real] [--synth-dir synthetic/results/synth] [--out figures_article]"""
import os, sys, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--real-dir', default=HERE + '/real_data/results/real'); ap.add_argument('--synth-dir', default=HERE + '/synthetic/results/synth'); ap.add_argument('--out', default=HERE + '/figures_article')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
load = lambda d, pat: [json.load(open(f)) for f in sorted(glob.glob(f'{d}/{pat}'))]
def save(fig, name): fig.tight_layout(); [fig.savefig(f'{a.out}/{name}.{e}', dpi=200) for e in ('png', 'pdf')]; plt.close(fig); print('figure', name)
def table(name, header, rows):
    with open(f'{a.out}/{name}.csv', 'w', encoding='utf-8') as f: f.write(','.join(header) + '\n' + '\n'.join(','.join(str(x) for x in r) for r in rows) + '\n')
    with open(f'{a.out}/{name}.tex', 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{' + 'l' + 'r' * (len(header) - 1) + '}\n\\hline\n' + ' & '.join(h.replace('_', '\\_') for h in header) + ' \\\\\n\\hline\n')
        for r in rows: f.write(' & '.join(str(x).replace('_', '\\_') for x in r) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print('table', name)
# ------------------------------------------------ real data ------------------------------------------------
main, ctrl, sweep = load(a.real_dir, 'real_none_seed*.json'), load(a.real_dir, 'real_shuffle_seed*.json'), load(a.real_dir, 'real_sweep_lam*.json')
if main:
    seeds = [r['seed'] for r in main]; keys = [('node_cos_test', 'NODE'), ('linear_baseline_cos_test', 'Linear'), ('sym_cos_vs_node_test', 'sym vs NODE'), ('sym_cos_vs_measured_test', 'sym vs velocity'), ('direct_cos_vs_measured_test', 'direct vs velocity')]
    fig, ax = plt.subplots(figsize=(6.2, 3.2)); w = 0.16
    for k, (key, lab) in enumerate(keys): ax.bar(np.arange(len(seeds)) + (k - 2) * w, [r[key] for r in main], w, label=lab)
    if ctrl: ax.axhline(np.mean([r['node_cos_test'] for r in ctrl]), color='k', ls='--', lw=0.8, label='Control (shuffled velocities)')
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f'seed {s}' for s in seeds]); ax.set_ylabel('Cosine similarity (test cells)'); ax.legend(fontsize=6, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False); save(fig, 'fig_real_cosines')
    table('tab_real_summary', ['seed', 'NODE_cos', 'linear_cos', 'sym_vs_NODE', 'sym_vs_V', 'direct_vs_V', 'Omega_total'], [[r['seed']] + [f"{r[k]:.3f}" for k, _ in keys] + [sum(r['omega'])] for r in main])
    if 'endpoints_node' in main[0]:
        cl = sorted({c for r in main for e in ('endpoints_node', 'endpoints_symbolic') for c in r[e]['distribution']}); fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
        for ax, e, t in zip(axs, ('endpoints_node', 'endpoints_symbolic'), ('NODE', 'Symbolic')):
            M = np.array([[r[e]['distribution'].get(c, 0) for c in cl] for r in main]); ax.bar(cl, M.mean(0), yerr=M.std(0), capsize=2); ax.set_title(t); ax.tick_params(axis='x', rotation=70)
        axs[0].set_ylabel("Arrival cells (mean +/- s.d. over seeds)"); save(fig, 'fig_real_endpoints')
if os.path.exists('docs/lambda_sweep_5seeds_summary.json'):
    s5 = json.load(open('docs/lambda_sweep_5seeds_summary.json'))
    lams = s5['lambdas']; sby = s5['summary_by_lambda']
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); x = np.arange(len(lams))
    c_m = [sby[str(l)]['cos_test_mean'] for l in lams]; c_s = [sby[str(l)]['cos_test_std'] for l in lams]
    v_m = [sby[str(l)]['cos_val_mean'] for l in lams]; v_s = [sby[str(l)]['cos_val_std'] for l in lams]
    cc_m = [sby[str(l)]['centered_cos_test_mean'] for l in lams]; cc_s = [sby[str(l)]['centered_cos_test_std'] for l in lams]
    ax.errorbar(x, c_m, yerr=c_s, fmt='o-', capsize=3, label='NODE test cosine')
    ax.errorbar(x, cc_m, yerr=cc_s, fmt='s--', capsize=3, label='NODE centered test cos')
    ax.errorbar(x, v_m, yerr=v_s, fmt='^:', capsize=3, label='NODE val cosine', alpha=0.7)
    lin_val = s5['linear_baseline_summary']['cos_test_mean']
    ax.axhline(lin_val, color='gray', linestyle='--', label=f'Linear baseline ({lin_val:.3f})')
    shuff_val = s5['shuffled_control_summary']['centered_cos_test_mean']
    ax.axhline(shuff_val, color='red', linestyle=':', label=f'Shuffled control ({shuff_val:.3f})')
    ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_xlabel(r'Regularization strength $\lambda$ (mean $\pm$ s.d., 5 seeds)')
    ax.set_ylabel('Cosine similarity')
    ax.legend(fontsize=6.5, loc='lower left')
    save(fig, 'fig_real_lambda_sweep')
elif sweep:
    sw = sorted(sweep, key=lambda r: r['lam_reg']); fig, ax = plt.subplots(figsize=(4.4, 3)); x = range(len(sw))
    ax.plot(x, [r['node_cos_test'] for r in sw], 'o-', label='NODE vs V'); ax.plot(x, [r['sym_cos_vs_node_test'] for r in sw], 's-', label='sym vs NODE'); ax.plot(x, [r['sym_cos_vs_measured_test'] for r in sw], '^-', label='sym vs V')
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{r['lam_reg']:g}" for r in sw]); ax.set_xlabel(r'$\lambda_{reg}$ (seed 0)'); ax.set_ylabel('Test cosine similarity'); ax.legend(fontsize=7); save(fig, 'fig_real_lambda_sweep')
sp = {}
for tag in ('real', 'real_module', 'real_latent'):
    L = load(a.real_dir, f'{tag}_none_seed*.json')
    if L: sp[{'real': 'genes', 'real_module': 'modules', 'real_latent': 'latent'}[tag]] = L
if len(sp) > 1:
    fig, ax = plt.subplots(figsize=(5, 3)); w = 0.25
    for k, key in enumerate(('node_cos_test', 'sym_cos_vs_node_test', 'sym_cos_vs_measured_test')): ax.bar(np.arange(len(sp)) + (k - 1) * w, [np.mean([r[key] for r in L]) for L in sp.values()], w, yerr=[np.std([r[key] for r in L]) for L in sp.values()], capsize=2, label=['NODE vs V', 'sym vs NODE', 'sym vs V'][k])
    ax.set_xticks(range(len(sp))); ax.set_xticklabels([f'{k} (n={len(v)})' for k, v in sp.items()]); ax.set_ylabel('Test cosine similarity'); ax.legend(fontsize=7); save(fig, 'fig_real_spaces')
    table('tab_real_spaces', ['space', 'seeds', 'NODE_cos', 'sym_vs_NODE', 'sym_vs_V', 'direct_vs_V', 'linear'], [[k, len(L)] + [f"{np.mean([r[x] for r in L]):.3f}" for x in ('node_cos_test', 'sym_cos_vs_node_test', 'sym_cos_vs_measured_test', 'direct_cos_vs_measured_test', 'linear_baseline_cos_test')] for k, L in sp.items()])
# ------------------------------------------------ synthetic benchmark ------------------------------------------------
S = {}
for f in sorted(glob.glob(f'{a.synth_dir}/synth_*_seed*.json')): r = json.load(open(f)); S.setdefault(r['variant'], []).append(r)
if S:
    order = [v for v in ('base', 'noise02', 'noise10', 'noise20', 'drop10', 'drop20', 'N20', 'N120', 'd3', 'd8', 'pseudo2', 'manifold') if v in S]
    g = lambda L, f: [f(r) for r in L]
    meth = [('distilled', 'Distilled'), ('direct_fd', 'Direct (finite diff.)'), ('weak_sindy', 'Weak form')]
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); w = 0.26
    for k, (key, lab) in enumerate(meth): ax.bar(np.arange(len(order)) + (k - 1) * w, [np.mean(g(S[v], lambda r: r[key]['structure']['F1'])) for v in order], w, yerr=[np.std(g(S[v], lambda r: r[key]['structure']['F1'])) for v in order], capsize=1.5, label=lab)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f'{v}\n(n={len(S[v])})' for v in order], fontsize=7); ax.set_ylabel('Structure recovery F1'); ax.legend(fontsize=7); save(fig, 'fig_synth_structure_F1')
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, vs, xs, xl in ((axs[0], ['noise02', 'base', 'noise10', 'noise20'], [0.02, 0.05, 0.10, 0.20], 'Relative noise'), (axs[1], ['base', 'drop10', 'drop20'], [0, 0.10, 0.20], 'Dropout rate')):
        ok = [(v, x) for v, x in zip(vs, xs) if v in S]
        if len(ok) > 1:
            ax.plot([x for _, x in ok], [np.mean(g(S[v], lambda r: r['distilled']['structure']['F1'])) for v, _ in ok], 'o-', label='Distilled F1'); ax.plot([x for _, x in ok], [np.mean(g(S[v], lambda r: r['node_metrics']['cos_vs_truth_support'])) for v, _ in ok], 's-', label='NODE vs true field'); ax.set_xlabel(xl); ax.legend(fontsize=7)
    save(fig, 'fig_synth_robustness')
    rows = []
    fmt_mean = lambda vals: (lambda v: f"{np.mean(v):.3f}" if v else "nan")([float(x) for x in vals if x is not None and not np.isnan(x)])
    for v in order:
        L = S[v]; f = lambda k: fmt_mean(g(L, k))
        rows.append([v, len(L), f(lambda r: r['node_metrics']['traj_nrmse_vs_truth']), f(lambda r: r['node_metrics']['cos_vs_truth_support']), f(lambda r: r['distilled']['structure']['F1']), f(lambda r: r['direct_fd']['structure']['F1']), f(lambda r: r['weak_sindy']['structure']['F1']),
                     f(lambda r: r['distilled']['structure']['param_relerr_all_median'] if r['distilled']['structure']['param_relerr_all_median'] is not None else np.nan), f(lambda r: r['prop1']['holds_fraction'] if r.get('prop1') else np.nan)])
    table('tab_synth_summary', ['variante', 'graines', 'NODE_traj_NRMSE', 'NODE_cos_verite', 'F1_distille', 'F1_direct', 'F1_faible', 'erreur_param_med', 'prop1_tient'], rows)
print('termine :', a.out)
