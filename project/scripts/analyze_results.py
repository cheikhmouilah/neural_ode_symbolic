"""Tables d'analyse (criteres preenregistres) a partir des JSON de results/phase1. Remplace les analyses executees en ligne.
Usage : python scripts/analyze_results.py {phase1|phase6|misspec|detect} [--dir CHEMIN]"""
import os, sys, json, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['phase1', 'phase6', 'misspec', 'detect']); ap.add_argument('--dir', default=ROOT + '/results/phase1'); a = ap.parse_args()
L = lambda pat, seeds=(0, 1, 2): [json.load(open(f"{a.dir}/{pat.format(s=s)}")) for s in seeds]
f3 = lambda v: '/'.join(f'{x:.3f}' for x in v)


def phase1():
    R = L('phase1_traj_seed{s}.json')
    print("== Criteres preenregistres Phase 1 (protocole : trajectoire seule, D_f = etats d'entrainement)")
    chk = [('P1 champ NODE NRMSE<0.05', [r['P1_node_field_nrmse_test'] for r in R], lambda v: v < 0.05), ('P3a err med<0.05', [r['distilled']['median_rel_err'] for r in R], lambda v: v < 0.05),
           ('P3b err max<0.10', [r['distilled']['max_rel_err'] for r in R], lambda v: v < 0.10), ('P4 traj symb NRMSE<0.05', [r['P4_symbolic_traj_nrmse_test'] for r in R], lambda v: v < 0.05)]
    print(f"P2 structure exacte : {[r['distilled']['structure_exact'] for r in R]}")
    for n, v, c in chk: print(f"{n:28s}: {f3(v)} -> {all(c(x) for x in v)}")
    print(f"P5 points fixes <5% : {[r['P5_fixed_points_symbolic']['max_rel_err'] for r in R]} -> {all(r['P5_fixed_points_symbolic']['ok'] for r in R)}")
    print("\nErreur par parametre (eq1, eq2) par graine :")
    for r in R:
        for i in (0, 1): print(f"  seed {r['seed']} eq{i+1}:", {k: f"{v*100:.1f}%" for k, v in r['distilled']['rel_err'][i].items()})
    print("\nReference derivees exactes (erreur max) :", [round(r['direct_exact_derivatives']['max_rel_err'], 6) for r in R])
    print("Fidelite NRMSE champ : NODE->verite", f3([r['P1_node_field_nrmse_test'] for r in R]), "| symbolique->verite", f3([r['field_nrmse_symbolic_vs_truth_test'] for r in R]), "| symbolique->NODE", f3([r['field_nrmse_symbolic_vs_node_test'] for r in R]))
    if os.path.exists(f"{a.dir}/phase1_vel_seed0.json"):
        V = L('phase1_vel_seed{s}.json'); print("\nAblation vitesse : err med", f3([r['distilled']['median_rel_err'] for r in V]), "| err max", f3([r['distilled']['max_rel_err'] for r in V]))
    print("\nSTLSQ generique (seed 0) : seuil -> nb termes [eq1,eq2] ; support exact")
    for row in R[0]['stlsq_baseline_on_distilled']: print("  ", row['threshold'], row['n_terms'], row['exact'])


def phase6():
    conds = [('traj', 'ref  s=0.00 N=40'), ('n05', 's=0.05 N=40'), ('n10', 's=0.10 N=40'), ('N15', 's=0.00 N=15')]
    print("== Phase 6 : R1 structure>=2/3 ; R2 mediane des err med<0.10 ; R3 traj<0.05 dans >=2/3")
    for tag, name in conds:
        try: D = L(f'phase1_{tag}_seed{{s}}.json')
        except FileNotFoundError: print(name, ": fichiers absents"); continue
        st = sum(r['distilled']['structure_exact'] for r in D); em = [r['distilled']['median_rel_err'] for r in D]; ex = [r['distilled']['max_rel_err'] for r in D]; tj = [r['P4_symbolic_traj_nrmse_test'] for r in D]
        ok = st >= 2 and np.median(em) < 0.10 and sum(v < 0.05 for v in tj) >= 2
        print(f"{name:16s} struct {st}/3 | err med {f3(em)} | err max {f3(ex)} (<0.10 : {sum(v<0.10 for v in ex)}/3, post hoc) | traj {f3(tj)} | reussie: {ok}")
        if tag != 'traj' and 'direct_finite_differences' in D[0]:
            print(f"{'':16s} distillation max {f3(ex)} vs differences finies max {f3([r['direct_finite_differences']['max_rel_err'] for r in D])}")


def misspec():
    D = {k: L(f'misspec_{k}_seed{{s}}.json') for k in ('hill', 'leaky', 'gauss')}
    print("== D1 (sym Hill->NODE > 0.03 ; controle 0/3, leaky>=2/3, gauss>=2/3)")
    for k, rs in D.items(): print(f"  {k:6s}", [round(r['hill_only']['nrmse_sym_to_node'], 4) for r in rs], sum(r['hill_only']['flagged_D1'] for r in rs), "/3")
    print("== S1 (bonne famille, 2 equations, >=2/3 graines)")
    for k, rs in D.items(): print(f"  {k:6s}", [r['selection']['chosen'] for r in rs], sum(r['selection']['correct_both'] for r in rs), "/3")
    print("== S2 (erreur mediane famille correcte <0.10)")
    for k, rs in D.items(): print(f"  {k:6s}", [round(r['true_family_median_rel_err'], 3) for r in rs], "mediane", round(float(np.median([r['true_family_median_rel_err'] for r in rs])), 3))
    print("== Hill seul : traj NRMSE / pts fixes err")
    for k, rs in D.items(): print(f"  {k:6s}", [round(r['hill_only']['traj_nrmse'], 3) for r in rs], [r['hill_only']['fixed_points']['max_rel_err'] for r in rs])


def detect():
    S = ['C1', 'C2', 'C3', 'M1', 'M2', 'M3', 'M4']; U = [json.load(open(f"{a.dir}/detect_{s}_seed{k}.json")) for s in S for k in (0, 1, 2)]
    ctrl = [u for u in U if u['system'][0] == 'C' and u['node_ok']]; mis = [u for u in U if u['system'][0] == 'M' and u['node_ok']]
    print("unites exclues (NODE seul > 0.10) :", [(u['system'], u['seed']) for u in U if not u['node_ok']] or 'aucune')
    for rule, key in (('T', 'rule_T_flag'), ('F', 'rule_F_flag')):
        fp, tp = sum(u[key] for u in ctrl), sum(u[key] for u in mis); print(f"regle {rule}: FP {fp}/{len(ctrl)} | detection {tp}/{len(mis)}")
    need = {'M1': 'leaky', 'M2': 'gauss'}
    keep = sum(g == 'hill' for u in U if u['system'][0] == 'C' for g in u['rule_G_choice']); good = sum(g == need[u['system']] for u in U if u['system'] in need for g in u['rule_G_choice'])
    bic = sum(g == 'hill' for u in U if u['system'][0] == 'C' for g in u['rule_B_choice'])
    print(f"regle G: Hill conserve {keep}/18 (SG1 >=16) | bonne famille M1/M2 {good}/12 (SG2 >=9) | BIC: Hill {bic}/18")
    DT1 = sum(u['rule_T_flag'] for u in ctrl) / len(ctrl) <= 0.12; DT2 = sum(u['rule_T_flag'] for u in mis) / len(mis) >= 0.75
    print(f"DT1 {DT1} | DT2 {DT2} | SG1 {keep>=16} | SG2 {good>=9} -> succes global {DT1 and DT2 and keep>=16 and good>=9}")


{'phase1': phase1, 'phase6': phase6, 'misspec': misspec, 'detect': detect}[a.what]()
