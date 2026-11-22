"""Verdicts sur les criteres preenregistres (configs/real_pancreas.json) a partir des resultats de run_real.py.
Inclut les comparaisons avec les baselines dynamo, CellRank et BoolODE (section VII-D)."""
import os, sys, json, glob, itertools, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/results/real'; TAG = sys.argv[2] if len(sys.argv) > 2 else 'real'
cfg = json.load(open(ROOT + '/configs/' + (sys.argv[3] if len(sys.argv) > 3 else 'real_pancreas.json'))); T = cfg['preregistered_thresholds']
load = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(f'{D}/{pat}'))]
main, ctrl, sweep = load(f'{TAG}_none_seed*.json'), load(f'{TAG}_shuffle_seed*.json'), load(f'{TAG}_sweep_lam*.json')
sets = lambda r, key='structure': [set(x) for x in r[key]]
def jac(A, B):
    js = [(1.0 if not (a | b) else len(a & b) / len(a | b)) for a, b in zip(A, B)]; return float(np.mean(js))
m = lambda L, k: float(np.mean([r[k] for r in L])) if L else float('nan')
if not main: sys.exit(f'aucun resultat {TAG}_none_seed*.json dans {D}')
print(f"config {main[0]['config_sha'] if main else '?'} | genes {main[0]['genes'] if main else '?'} | graines {[r['seed'] for r in main]}")
print('\n=== par graine ===\nseed | NODE cos test | lineaire | sym~NODE | sym~V | direct~V | R7 | Omega')
for r in main: print(f"{r['seed']:4d} | {r['node_cos_test']:.3f} | {r['linear_baseline_cos_test']:.3f} | {r['sym_cos_vs_node_test']:.3f} | {r['sym_cos_vs_measured_test']:.3f} | {r['direct_cos_vs_measured_test']:.3f} | {r['R7_ratio']:.2f} | {r['omega']}")
out = {}
c_ctrl = m(ctrl, 'node_cos_test'); c_node = m(main, 'node_cos_test')
out['R1'] = (c_node >= T['R1_node_test_cosine_min'] and c_node >= c_ctrl + T['R1_margin_over_shuffled_control']); print(f"\nR1 NODE cos test moyen={c_node:.3f} (>= {T['R1_node_test_cosine_min']}) ; controle melange={c_ctrl:.3f} (marge exigee {T['R1_margin_over_shuffled_control']}) -> {'OK' if out['R1'] else 'ECHEC'}")
out['R2'] = m(main, 'sym_cos_vs_node_test') >= T['R2_symbolic_vs_node_test_cosine_min']; print(f"R2 sym~NODE test moyen={m(main,'sym_cos_vs_node_test'):.3f} (>= {T['R2_symbolic_vs_node_test_cosine_min']}) -> {'OK' if out['R2'] else 'ECHEC'}")
J = [jac(sets(a), sets(b)) for a, b in itertools.combinations(main, 2)]
out['R3'] = bool(J) and float(np.mean(J)) >= T['R3_support_jaccard_across_seeds_min']; print(f"R3 Jaccard de support entre graines moyen={np.mean(J):.3f} (>= {T['R3_support_jaccard_across_seeds_min']}) -> {'OK' if out['R3'] else 'ECHEC'}")
ref = [r for r in sweep if abs(r['lam_reg'] - cfg['node']['lam_reg']) < 1e-12]
if ref and len(sweep) > 1:
    Jl = {r['lam_reg']: jac(sets(ref[0]), sets(r)) for r in sweep if r is not ref[0]}
    out['R4'] = all(v >= T['R4_support_jaccard_across_lam_reg_min'] for v in Jl.values()); print(f"R4 Jaccard vs lam_ref par lam={ {k: round(v,3) for k,v in Jl.items()} } -> {'OK' if out['R4'] else 'ECHEC'}")
else: out['R4'] = None; print('R4 : balayage lam_reg absent')
out['R7'] = m(main, 'R7_ratio') >= T['R7_symbolic_vs_measured_cosine_ratio_min']; print(f"R7 rapport cos(sym,V)/cos(NODE,V) moyen={m(main,'R7_ratio'):.3f} (>= {T['R7_symbolic_vs_measured_cosine_ratio_min']}) -> {'OK' if out['R7'] else 'ECHEC'}")
print('\n=== descriptif R5 / R6 (sans seuil) ===')
for r in main:
    if 'eq_node' in r:
        cn = [c['dist_over_radius'] for c in r['eq_node']['clusters'] if c['dist_over_radius'] is not None]; cs = [c['dist_over_radius'] for c in r['eq_symbolic']['clusters'] if c['dist_over_radius'] is not None]
        print(f"seed {r['seed']}: equilibres NODE {r['eq_node']['n']} (stables {r['eq_node']['n_stable']}) | symbolique {r['eq_symbolic']['n']} (stables {r['eq_symbolic']['n_stable']}) | clusters a <1 rayon d'un stable : NODE {sum(x<1 for x in cn)}/{len(cn)}, sym {sum(x<1 for x in cs)}/{len(cs)} | arrivees NODE {r['endpoints_node']['distribution']} | sym {r['endpoints_symbolic']['distribution']}")
print(f"distille vs direct : cos vs V = {m(main,'sym_cos_vs_measured_test'):.3f} vs {m(main,'direct_cos_vs_measured_test'):.3f} ; Jaccard de structure moyen = {m(main,'distilled_vs_direct_jaccard'):.3f}")
print(f"lineaire (reference) cos test = {m(main,'linear_baseline_cos_test'):.3f}")
if main:
    print('\nEquations (graine', main[0]['seed'], ', distillation) :'); [print('  ', e) for e in main[0]['equations']]
    print('Equations directes (identification sur vitesses mesurees) :'); [print('  ', e) for e in main[0]['direct_equations']]
    print('\nSTLSQ (PySINDy) sur D_f, graine', main[0]['seed'], ':', [(s['threshold'], round(s['mean_terms'], 1), round(s['cos_vs_node_test'], 3)) for s in main[0].get('stlsq_baseline', [])])

# --- Baselines VII-D : dynamo, CellRank, BoolODE ---
baselines_path = f'{D}/baselines_{TAG}.json'
if os.path.exists(baselines_path):
    bl = json.load(open(baselines_path))
    print('\n=== Baselines VII-D (dynamo / CellRank / BoolODE) ===')
    print(f"{'méthode':<25} {'cos_test':>10} {'nrmse_test':>12}")
    print('-' * 50)
    # Reference : NODE et symbolique
    if main:
        c_node = float(np.mean([r['node_cos_test'] for r in main]))
        c_sym  = float(np.mean([r['sym_cos_vs_measured_test'] for r in main]))
        c_lin  = float(np.mean([r['linear_baseline_cos_test'] for r in main]))
        print(f"  {'node_scvelo':<23} {c_node:>10.4f} {'':>12}")
        print(f"  {'symbolique_distille':<23} {c_sym:>10.4f} {'':>12}")
        print(f"  {'lineaire':<23} {c_lin:>10.4f} {'':>12}")
    for name, r in bl.get('baselines', {}).items():
        c = r.get('cos_test', float('nan'))
        n = r.get('nrmse_test', float('nan'))
        print(f"  {name:<23} {c:>10.4f} {n:>12.4f}")
        if name == 'cellrank' and 'absorption' in r:
            ab = r['absorption']
            print(f"    états terminaux : {ab.get('terminal_states', [])} "
                  f"| pureté test = {ab.get('cluster_purity_test', float('nan')):.3f}")
        if name == 'boolode' and 'network' in r:
            print(f"    réseau : " + ', '.join(f"{g}←{len(v)}" for g, v in r['network'].items()))
    out['baselines'] = {name: r.get('cos_test') for name, r in bl.get('baselines', {}).items()}
else:
    print(f'\n[info] Baselines non disponibles ({baselines_path}). Lancer :\n'
          f'  python scripts/run_real_baselines.py --config {sys.argv[3] if len(sys.argv) > 3 else "real_pancreas.json"} '
          + ('--quick ' if D.endswith("real_quick") else '') + f'--tag {TAG}')

json.dump(out, open(f'{D}/verdicts' + ('' if TAG == 'real' else '_' + TAG) + '.json', 'w'), indent=1); print('\nVERDICTS', out)
