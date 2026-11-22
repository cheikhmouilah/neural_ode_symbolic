"""Verdicts S1-S6 (seuils de configs/synth.json fixes avant execution) et tableau par variante."""
import os, sys, json, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/results/synth'
C = json.load(open(ROOT + '/configs/synth.json')); T = C['preregistered_thresholds']
res = {}
for f in sorted(glob.glob(f'{D}/synth_*_seed*.json')):
    r = json.load(open(f)); res.setdefault(r['variant'], []).append(r)
m = lambda L, g: float(np.nanmean([g(r) for r in L])) if L else float('nan')
def row(v, L):
    return dict(variant=v, n=len(L), node_traj=m(L, lambda r: r['node_metrics']['traj_nrmse_vs_truth']), node_cos=m(L, lambda r: r['node_metrics']['cos_vs_truth_support']),
                dist_F1=m(L, lambda r: r['distilled']['structure']['F1']), dist_P=m(L, lambda r: r['distilled']['structure']['precision']), dist_R=m(L, lambda r: r['distilled']['structure']['recall']),
                dist_cos=m(L, lambda r: r['distilled']['cos_vs_truth_support']), dist_cos_off=m(L, lambda r: r['distilled']['cos_vs_truth_offsupport']),
                direct_F1=m(L, lambda r: r['direct_fd']['structure']['F1']), weak_F1=m(L, lambda r: r['weak_sindy']['structure']['F1']),
                oracle_relerr=m(L, lambda r: r['oracle_structure']['structure']['param_relerr_all_median'] if r['oracle_structure']['structure']['param_relerr_all_median'] is not None else np.nan),
                dist_relerr=m(L, lambda r: r['distilled']['structure']['param_relerr_all_median'] if r['distilled']['structure']['param_relerr_all_median'] is not None else np.nan),
                prop1=m(L, lambda r: r['prop1']['holds_fraction'] if r.get('prop1') else np.nan), speed_ratio=m(L, lambda r: r['distilled']['speed_ratio_median']))
rows = [row(v, res[v]) for v in C['variants'] if v in res]
cols = ['n', 'node_traj', 'node_cos', 'dist_F1', 'dist_P', 'dist_R', 'direct_F1', 'weak_F1', 'dist_relerr', 'oracle_relerr', 'dist_cos', 'dist_cos_off', 'prop1', 'speed_ratio']
print('variante      ' + ' '.join(f'{c:>11s}' for c in cols))
for r in rows: print(f"{r['variant']:13s} " + ' '.join(f"{r[c]:11.3f}" if c != 'n' else f"{r[c]:11d}" for c in cols))
out = {}; b = res.get('base', [])
if b:
    out['S1'] = m(b, lambda r: r['distilled']['structure']['F1']) >= T['S1_base_structure_F1_min']; print(f"\nS1 F1 de structure (distille, base) = {m(b, lambda r: r['distilled']['structure']['F1']):.3f} (>= {T['S1_base_structure_F1_min']}) -> {'OK' if out['S1'] else 'ECHEC'}  [graines : {len(b)}]")
    med = float(np.nanmedian([r['distilled']['structure']['param_relerr_all_median'] for r in b if r['distilled']['structure']['param_relerr_all_median'] is not None] or [np.nan]))
    out['S2'] = med <= T['S2_base_matched_param_median_relerr_max']; print(f"S2 erreur relative mediane des parametres (termes retrouves) = {med:.3f} (<= {T['S2_base_matched_param_median_relerr_max']}) -> {'OK' if out['S2'] else 'ECHEC'}")
    out['S3'] = m(b, lambda r: r['node_metrics']['cos_vs_truth_support']) >= T['S3_base_node_vs_truth_cosine_min']; print(f"S3 cosinus NODE/verite = {m(b, lambda r: r['node_metrics']['cos_vs_truth_support']):.3f} (>= {T['S3_base_node_vs_truth_cosine_min']}) -> {'OK' if out['S3'] else 'ECHEC'}")
    ph = [r['prop1']['holds_fraction'] for v, L in res.items() for r in L if r.get('prop1') and r['cfg']['time_mode'] == 'true']
    out['S4'] = bool(ph) and float(np.mean(ph)) >= T['S4_prop1_bound_holds_fraction_min']; print(f"S4 proposition 1 : borne verifiee sur {np.mean(ph):.3f} des trajectoires de test, {len(ph)} runs (>= {T['S4_prop1_bound_holds_fraction_min']}) -> {'OK' if out['S4'] else 'ECHEC'}")
ps = res.get('pseudo2', [])
if ps: out['S5'] = m(ps, lambda r: r['distilled']['cos_vs_truth_support']) >= T['S5_pseudo_direction_cosine_min']; print(f"S5 pseudotemps (gamma=2) : cosinus direction distille/verite = {m(ps, lambda r: r['distilled']['cos_vs_truth_support']):.3f} (>= {T['S5_pseudo_direction_cosine_min']}) ; rapport de vitesse median = {m(ps, lambda r: r['distilled']['speed_ratio_median']):.2f} (1 = temps vrai) -> {'OK' if out['S5'] else 'ECHEC'}")
mf = res.get('manifold', [])
if mf:
    drop = m(mf, lambda r: r['distilled']['cos_vs_truth_support'] - r['distilled']['cos_vs_truth_offsupport']); out['S6'] = drop >= T['S6_manifold_offsupport_cosine_drop_min']
    print(f"S6 variete : chute de cosinus hors support = {drop:.3f} (>= {T['S6_manifold_offsupport_cosine_drop_min']} attendu si l'ambiguite existe) -> {'OK' if out['S6'] else 'ECHEC'}")
ff = [r['free_form_gplearn'] for L in res.values() for r in L if 'free_form_gplearn' in r]
if ff: print(f"\nforme libre (gplearn) : rappel des variables {np.mean([x['mean_var_recall'] for x in ff]):.2f}, precision {np.mean([x['mean_var_precision'] for x in ff]):.2f}, longueur moyenne {np.mean([x['mean_length'] for x in ff]):.1f}, R2 vs NODE {np.mean([x['mean_r2_vs_node'] for x in ff]):.2f} ({len(ff)} runs)")
json.dump(dict(verdicts=out, table=rows), open(f'{D}/verdicts_synth.json', 'w'), indent=1, default=float); print('\nVERDICTS', out)
