#!/usr/bin/env python3
"""Genere RAPPORT_RESULTATS.md A PARTIR DES CHIFFRES (aucune phrase de conclusion ecrite a la main). Les verdicts sont RECALCULES depuis les resultats bruts et compares aux
verdicts enregistres par les scripts d'analyse ; toute contradiction arrete la generation (code 1). La provenance des donnees est ecrite en tete.
   python checks/make_report.py [--synth DIR] [--real DIR] [--out FILE]"""
import os, sys, json, glob, argparse, platform, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--synth', default=os.path.join(ROOT, RESULT_DIRS['synthetic'])); ap.add_argument('--real', default=os.path.join(ROOT, RESULT_DIRS['real'])); ap.add_argument('--out', default=os.path.join(ROOT, 'RAPPORT_RESULTATS.md')); a = ap.parse_args()
DS = json.load(open(os.path.join(ROOT, 'datasets.json'))); out = []; contradictions = []
W = out.append
def ok(b): return 'satisfait' if b else '**NON satisfait**'
def cmp_stored(name, recomputed, stored):
    if stored is not None and bool(stored) != bool(recomputed): contradictions.append(f'{name} : verdict enregistre {stored} != recalcul {recomputed}')
def stored(path, key):
    try: return json.load(open(path)).get(key)
    except Exception: return None
W('# Rapport de résultats (généré automatiquement)\n')
W('Ce document est produit par `checks/make_report.py` à partir des fichiers de résultats ; chaque verdict est recalculé depuis les valeurs brutes et comparé à celui des scripts d\'analyse. Aucune phrase n\'est écrite à la main.\n')
# ---------------- provenance ----------------
W('## Provenance\n'); rows = []
for k, v in DS.items():
    p = os.path.join(ROOT, v['file']); present = os.path.exists(p); good = present and sha256(p) == v['sha256']
    rows.append(f"| {k} | {'présent' if present else 'absent'} | {'oui' if good else ('NON' if present else '—')} | `{v['sha256'][:16]}…` |")
W('| Jeu de données | Fichier | SHA-256 conforme | Empreinte attendue |\n|:--|:--|:--|:--|\n' + '\n'.join(rows) + '\n')
real_main = sorted(glob.glob(f'{a.real}/real_none_seed*.json')); R = [load_json(f) for f in real_main]
ds_ok = [k for k, v in DS.items() if os.path.exists(os.path.join(ROOT, v['file'])) and sha256(os.path.join(ROOT, v['file'])) == v['sha256']]
W(f"- **using_real_data : {('true' if ds_ok else 'résultats présents mais jeux de données absents ou non conformes sur cette machine : provenance non re-vérifiable ici') if R else 'false (aucun résultat sur données réelles)'}** ; la chaîne C vérifie le SHA-256 du fichier h5ad avant de calculer et n'a aucun repli sur des données simulées (une erreur interrompt l'exécution).")
shas = sorted({r['config_sha'] for r in R}); W(f"- Empreintes de configuration des résultats réels : {', '.join('`'+s+'`' for s in shas) or '—'}")
pj = os.path.join(ROOT, 'PREREGISTRATION.json'); W(f"- Gel du préenregistrement : {'présent (' + json.load(open(pj))['frozen_utc'] + ')' if os.path.exists(pj) else 'ABSENT : les seuils ne sont pas ancrés'}")
W(f"- Python {platform.python_version()}, {platform.system()} {platform.release()}\n")
# ---------------- chaine C ----------------
W('## Données réelles (chaîne C)\n')
if R:
    T = json.load(open(os.path.join(ROOT, 'real_data/configs/real_pancreas.json')))['preregistered_thresholds']
    ctrl = [load_json(f) for f in sorted(glob.glob(f'{a.real}/real_shuffle_seed*.json'))]; sw = [load_json(f) for f in sorted(glob.glob(f'{a.real}/real_sweep_lam*.json'))]
    m = lambda L, k: float(np.mean([r[k] for r in L])) if L else float('nan')
    node, ctr, s2, r7 = m(R, 'node_cos_test'), m(ctrl, 'node_cos_test'), m(R, 'sym_cos_vs_node_test'), m(R, 'R7_ratio')
    r1 = node >= T['R1_node_test_cosine_min'] and bool(ctrl) and node >= ctr + T['R1_margin_over_shuffled_control']; r2 = s2 >= T['R2_symbolic_vs_node_test_cosine_min']; r7ok = r7 >= T['R7_symbolic_vs_measured_cosine_ratio_min']
    sv = os.path.join(a.real, 'verdicts.json'); cmp_stored('R1', r1, stored(sv, 'R1')); cmp_stored('R2', r2, stored(sv, 'R2')); cmp_stored('R7', r7ok, stored(sv, 'R7'))
    lin = m(R, 'linear_baseline_cos_test')
    W(f"{len(R)} graines ; gènes de l'espace d'état : {', '.join(R[0]['genes'])}.\n")
    W('| Critère | Valeur | Seuil | Verdict |\n|:--|:--|:--|:--|')
    W(f"| R1 cosinus NODE (test) | {node:.3f} | ≥ {T['R1_node_test_cosine_min']} | {ok(node >= T['R1_node_test_cosine_min'])} |")
    W(f"| R1 marge sur le contrôle à vitesses mélangées ({ctr:.3f}) | {node - ctr:.3f} | ≥ {T['R1_margin_over_shuffled_control']} | {ok(bool(ctrl) and node - ctr >= T['R1_margin_over_shuffled_control'])} |")
    W(f"| R2 cosinus symbolique/NODE | {s2:.3f} | ≥ {T['R2_symbolic_vs_node_test_cosine_min']} | {ok(r2)} |")
    W(f"| R7 rapport cos(sym,V)/cos(NODE,V) | {r7:.3f} | ≥ {T['R7_symbolic_vs_measured_cosine_ratio_min']} | {ok(r7ok)} |\n")
    W(f"Référence linéaire : cosinus {lin:.3f} — le Neural ODE {'dépasse' if node > lin else 'ne dépasse pas'} cette référence (écart {node - lin:+.3f}). "
      f"Identification directe : {m(R, 'direct_cos_vs_measured_test'):.3f} contre {m(R, 'sym_cos_vs_measured_test'):.3f} pour la distillation (cosinus avec la vitesse mesurée).")
    if sw: W(f"Balayage de λ_reg (graine 0) : cosinus NODE de {min(r['node_cos_test'] for r in sw):.3f} à {max(r['node_cos_test'] for r in sw):.3f} selon λ_reg.")
    W('\nAucune vérité terrain : ces critères ne mesurent ni la récupération d\'équations ni la causalité.\n')
else: W('Aucun résultat.\n')
# ---------------- chaine D ----------------
W('## Benchmark synthétique (chaîne D)\n'); S = {}
for f in sorted(glob.glob(f'{a.synth}/synth_*_seed*.json')): r = load_json(f); S.setdefault(r['variant'], []).append(r)
if S:
    T = json.load(open(os.path.join(ROOT, 'synthetic/configs/synth.json')))['preregistered_thresholds']; b = S.get('base', []); sv = os.path.join(a.synth, 'verdicts_synth.json')
    try: SV = json.load(open(sv))['verdicts']
    except Exception: SV = {}
    mm = lambda L, g: float(np.nanmean([g(r) for r in L]))
    W(f"Variantes disponibles : {', '.join(f'{v} ({len(L)} graine(s))' for v, L in S.items())}.\n")
    if b:
        f1 = mm(b, lambda r: r['distilled']['structure']['F1']); s1 = f1 >= T['S1_base_structure_F1_min']; cmp_stored('S1', s1, SV.get('S1'))
        pe = [r['distilled']['structure']['param_relerr_all_median'] for r in b if r['distilled']['structure']['param_relerr_all_median'] is not None]; med = float(np.median(pe)) if pe else float('nan'); s2 = med <= T['S2_base_matched_param_median_relerr_max']; cmp_stored('S2', s2, SV.get('S2'))
        c = mm(b, lambda r: r['node_metrics']['cos_vs_truth_support']); s3 = c >= T['S3_base_node_vs_truth_cosine_min']; cmp_stored('S3', s3, SV.get('S3'))
        ph = [r['prop1']['holds_fraction'] for L in S.values() for r in L if r.get('prop1') and r['cfg']['time_mode'] == 'true']; s4 = bool(ph) and float(np.mean(ph)) >= T['S4_prop1_bound_holds_fraction_min']; cmp_stored('S4', s4, SV.get('S4'))
        W('| Critère | Valeur | Seuil | Verdict |\n|:--|:--|:--|:--|')
        W(f"| S1 F1 de structure (distillé, base, {len(b)} graines) | {f1:.3f} | ≥ {T['S1_base_structure_F1_min']} | {ok(s1)} |")
        W(f"| S2 erreur relative médiane des paramètres | {med:.3f} | ≤ {T['S2_base_matched_param_median_relerr_max']} | {ok(s2)} |")
        W(f"| S3 cosinus NODE/champ vrai | {c:.3f} | ≥ {T['S3_base_node_vs_truth_cosine_min']} | {ok(s3)} |")
        W(f"| S4 proposition 1 : borne tenue ({len(ph)} runs) | {float(np.mean(ph)) if ph else float('nan'):.3f} | ≥ {T['S4_prop1_bound_holds_fraction_min']} | {ok(s4)} |")
        if 'pseudo2' in S:
            ps = S['pseudo2']; v = mm(ps, lambda r: r['distilled']['cos_vs_truth_support']); s5 = v >= T['S5_pseudo_direction_cosine_min']; cmp_stored('S5', s5, SV.get('S5')); W(f"| S5 pseudotemps : cosinus de direction ({len(ps)} graine(s)) | {v:.3f} | ≥ {T['S5_pseudo_direction_cosine_min']} | {ok(s5)} |")
        if 'manifold' in S:
            mf = S['manifold']; v = mm(mf, lambda r: r['distilled']['cos_vs_truth_support'] - r['distilled']['cos_vs_truth_offsupport']); s6 = v >= T['S6_manifold_offsupport_cosine_drop_min']; cmp_stored('S6', s6, SV.get('S6')); W(f"| S6 variété : chute de cosinus hors support ({len(mf)} graine(s)) | {v:.3f} | ≥ {T['S6_manifold_offsupport_cosine_drop_min']} | {ok(s6)} |")
        dd, ww = mm(b, lambda r: r['direct_fd']['structure']['F1']), mm(b, lambda r: r['weak_sindy']['structure']['F1'])
        W(f"\nF1 de structure sur la variante de base : distillation {f1:.3f}, identification directe {dd:.3f}, forme faible {ww:.3f} — la distillation {'dépasse' if f1 > max(dd, ww) else 'ne dépasse pas'} les deux bases de référence.")
        ko = [r['ko'] for r in b if r.get('ko')]
        if ko: W(f"Interventions in silico (délétion d'un gène) : cosinus des décalages prédits/vrais {np.mean([k['cos_sym'] for k in ko]):.3f} (symbolique) contre {np.mean([k['cos_node'] for k in ko]):.3f} (Neural ODE).")
    n1 = [v for v, L in S.items() if len(L) < 3]
    if n1: W(f"\nVariantes avec moins de 3 graines (à ne pas interpréter) : {', '.join(n1)}.")
else: W('Aucun résultat.\n')
# ---------------- PySR ----------------
W('\n## Recherche symbolique libre (PySR)\n')
tg = sorted(glob.glob(os.path.join(ROOT, 'project/results/phase1/pysr_toggle_seed*.json'))); T_ = [load_json(f) for f in tg]
rp = sorted(glob.glob(f'{a.real}/real_pysr_seed*.json')); RP = [load_json(f) for f in rp]
sp_ = [r for L in S.values() for r in L if r.get('free_form_pysr') and 'equations' in r['free_form_pysr']] if S else []
sp_ += [load_json(f) for f in sorted(glob.glob(f'{a.synth}/synthpysr_*_seed*.json')) if 'equations' in (load_json(f).get('free_form_pysr') or {})]
if not (T_ or RP or sp_): W('**Aucun résultat PySR** : le projet complet attend les tâches `toggle_pysr_*`, `real_pysr_*` et `synthpysr_*` (voir `GUIDE_PYSR.md`).')
else:
    W('Recherche d\'expressions sans famille imposée (opérateurs + − × ÷) sur le champ du Neural ODE ; seuils lus dans `project/configs/pysr.json`, `real_data/configs/real_pysr.json`, `synthetic/configs/synth.json`. Chaque seuil a été fixé après un essai pilote d\'une graine (divulgué dans les fichiers de configuration).\n')
    W('| Critère | Valeur | Seuil | Verdict |\n|:--|:--|:--|:--|')
    if T_:
        TP = json.load(open(os.path.join(ROOT, 'project/configs/pysr.json')))['preregistered_thresholds']
        r2 = float(np.mean([np.mean(r['fidelity_r2_vs_node']) for r in T_])); nr = float(np.mean([r['field_nrmse_vs_truth'] for r in T_])); fp = all(r['stable_fixed_points']['n_model'] == TP['PT3_stable_fixed_points_required'] for r in T_)
        og = float(np.mean([q['uses_other_gene'] for r in T_ for q in r['equations']]))
        W(f"| PT1 toggle : fidélité R² au Neural ODE ({len(T_)} graines) | {r2:.3f} | ≥ {TP['PT1_fidelity_r2_vs_node_min']} | {ok(r2 >= TP['PT1_fidelity_r2_vs_node_min'])} |")
        W(f"| PT2 toggle : NRMSE de champ contre la vérité | {nr:.4f} | ≤ {TP['PT2_field_nrmse_vs_truth_max']} | {ok(nr <= TP['PT2_field_nrmse_vs_truth_max'])} |")
        W(f"| PT3 toggle : {TP['PT3_stable_fixed_points_required']} points fixes stables conservés dans chaque graine | {sum(r['stable_fixed_points']['n_model'] == 2 for r in T_)}/{len(T_)} graines | toutes | {ok(fp)} |")
        W(f"| PT4 toggle : le gène régulateur apparaît dans l'équation | {og:.2f} | ≥ {TP['PT4_uses_other_gene_fraction_min']} | {ok(og >= TP['PT4_uses_other_gene_fraction_min'])} |")
    if RP:
        RT = json.load(open(os.path.join(ROOT, 'real_data/configs/real_pysr.json')))['preregistered_thresholds']; c1 = float(np.mean([r['pysr_vs_node_cos_test'] for r in RP])); c2 = float(np.mean([r['pysr_vs_measured_cos_test'] / max(r['node_vs_measured_cos_same_genes'], 1e-9) for r in RP]))
        W(f"| RP1 données réelles : cosinus PySR/NODE ({len(RP)} graines) | {c1:.3f} | ≥ {RT['RP1_pysr_vs_node_cos_min']} | {ok(c1 >= RT['RP1_pysr_vs_node_cos_min'])} |")
        W(f"| RP2 données réelles : cosinus (PySR, vitesse) / cosinus (NODE, vitesse) | {c2:.3f} | ≥ {RT['RP2_pysr_vs_measured_over_node_vs_measured_min']} | {ok(c2 >= RT['RP2_pysr_vs_measured_over_node_vs_measured_min'])} |")
    if sp_:
        ST = json.load(open(os.path.join(ROOT, 'synthetic/configs/synth.json')))['preregistered_thresholds']; vr = float(np.mean([r['free_form_pysr']['mean_var_recall'] for r in sp_]))
        W(f"| S7 synthétique : rappel moyen des variables vraies ({len(sp_)} runs) | {vr:.3f} | ≥ {ST['S7_pysr_mean_variable_recall_min']} | {ok(vr >= ST['S7_pysr_mean_variable_recall_min'])} |")
    if T_:
        W('\nExpressions trouvées sur le toggle (une ligne par graine et équation ; vérité : dx/dt = a/(1+(y/K)^n) − g·x) :\n')
        for r in T_: W(f"- graine {r['seed']} : " + ' ; '.join(f"éq. {q['eq']} (complexité {q['complexity']}) : `{q['expression']}`" for q in r['equations']))
    if RP: W('\nGènes utilisés par chaque équation (données réelles, graine ' + str(RP[0]['seed']) + ') : ' + ' ; '.join(f"{q['gene']} ← {', '.join(q['used_genes']) or 'aucun'}" for q in RP[0]['equations']) + '.')
    W('\nLes expressions sont des hypothèses de modélisation (non uniques, VI-A) ; PySR ne fournit ni mécanisme ni causalité.')
W('\n## Limites\n\n- Chaînes A et B (toggle 2D, SERGIO) : leurs verdicts sont produits par leurs propres scripts d\'analyse (`analyze_results.py`, `analyze_sergio.py`) ; ce rapport ne les recalcule pas.\n- dynamo, CellRank, BoolODE non exécutés ; aucune perturbation expérimentale.\n- Les verdicts ci-dessus portent sur les seuils préenregistrés ; toute variante ajoutée après coup est exploratoire.')
if contradictions:
    print('CONTRADICTION entre verdicts enregistrés et recalcul :'); [print('  -', c) for c in contradictions]; sys.exit(1)
open(a.out, 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('rapport écrit :', a.out, f'({len(out)} blocs, aucune contradiction)')
