"""Experiment: Discrete n Misspecification Stress Test
Investigate whether the stoichiometric discrete prior n in {1, 2, 4} maintains structural F1
and reasonable parameter recovery when the ground truth exponent lies outside the grid
(n_true = 3.0 and n_true = 1.5), comparing against continuous refinement and matched discrete grid.
"""
import os, sys, json, time, copy
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'real_data', 'src'))
sys.path.insert(0, os.path.join(ROOT, 'synthetic', 'src'))

from synth_system import field_true, simulate, make_data, zhi, sample_ics
from synth_methods import (
    train_traj_node, integrate_node, sample_Df, structure_metrics,
    cosine, nrmse
)
from real_node import field_np
from real_identify import build_candidates, select_structure, predict_all, describe
from scipy.integrate import solve_ivp

def gen_system_fixed_n(d, seed, n_val):
    rng = np.random.default_rng(10_000 + seed)
    eqs = []
    for i in range(d):
        nreg = 1 if rng.random() < 0.5 else 2
        regs = rng.choice(d, size=min(nreg, d), replace=False)
        terms = [
            dict(
                kind='hill',
                j=int(j),
                typ=str(rng.choice(['rep', 'act'])),
                K=float(rng.uniform(0.4, 1.5)),
                n=float(n_val),
                a=float(rng.uniform(1.0, 3.0))
            )
            for j in regs
        ]
        eqs.append(dict(i=i, terms=terms, g=float(rng.uniform(0.8, 1.4))))
    return eqs

def run_misspecification_experiment():
    print("=" * 75, flush=True)
    print("STARTING DISCRETE n MISSPECIFICATION EXPERIMENT", flush=True)
    print("Testing true exponents n_true in {3.0, 1.5, 2.0} under discrete prior {1, 2, 4} vs continuous", flush=True)
    print("=" * 75, flush=True)

    t_global_start = time.time()
    synth_cfg_path = os.path.join(ROOT, 'synthetic', 'configs', 'synth.json')
    with open(synth_cfg_path) as f_in:
        full_cfg = json.load(f_in)
    base_cfg = full_cfg['base']
    
    seeds = [0, 1, 2, 3, 4]
    n_true_conditions = [3.0, 1.5, 2.0]
    
    all_results = []
    
    for n_true in n_true_conditions:
        print(f"\n>>> Running Ground Truth condition: n_true = {n_true} <<<", flush=True)
        
        for seed in seeds:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            d = base_cfg['d']
            
            # 1. Generate system with fixed n_true
            eqs = gen_system_fixed_n(d, seed, n_true)
            
            # 2. Simulate data
            data = make_data(eqs, base_cfg, 1000 + seed)
            X, t_obs, t_true = data['X'], data['t_obs'], data['t_true']
            
            perm = rng.permutation(len(X))
            nv = max(2, int(base_cfg['node']['val_fraction'] * len(X)))
            va, tr = perm[:nv], perm[nv:]
            
            # 3. Train Trajectory Neural ODE
            f_node, info_node = train_traj_node(X[tr], X[va], t_obs, base_cfg['node'], seed)
            
            # Test set on support and off support
            Z0te = sample_ics(eqs, base_cfg['n_test'], np.random.default_rng(5000 + seed), base_cfg['ic_mode'])
            Zte = simulate(eqs, Z0te, t_true)
            Zsup = Zte.reshape(-1, d)
            Ftrue_sup = field_true(eqs, Zsup)
            Fnode_sup = field_np(f_node, Zsup)
            node_cos = cosine(Fnode_sup, Ftrue_sup)
            
            # 4. Sample D_f for distillation
            Zf = sample_Df(f_node, X[tr][:, 0], t_obs[-1], rng, base_cfg['distillation'])
            Ff = field_np(f_node, Zf)
            p = rng.permutation(len(Zf))
            nvf = int(base_cfg['distillation']['val_fraction'] * len(Zf))
            sv, st = p[:nvf], p[nvf:]
            
            # Evaluate Regime 1: Discrete stoichiometric prior {1.0, 2.0, 4.0}
            lib_discrete = copy.deepcopy(base_cfg['library'])
            lib_discrete['discrete_n'] = True
            lib_discrete['discrete_n_set'] = [1.0, 2.0, 4.0]
            lib_discrete['n_grid'] = [1.0, 2.0, 4.0]
            cand_discrete = build_candidates(Zf[st], lib_discrete['K_grid'], lib_discrete['n_grid'])
            
            models_disc = []
            for i in range(d):
                ch, _ = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, lib_discrete, rng, cand=cand_discrete)
                models_disc.append(ch['model'])
            
            struct_disc = structure_metrics(models_disc, eqs)
            Fs_disc = predict_all(models_disc, Zsup)
            cos_disc = cosine(Fs_disc, Ftrue_sup)
            nrmse_disc = nrmse(Fs_disc, Ftrue_sup)
            
            # Track discrete n choices
            chosen_n_disc = [t['n'] for m in models_disc for t in m['terms'] if t['kind'] == 'hill']
            
            # Evaluate Regime 2: Continuous refinement over [1.0, 6.0]
            lib_cont = copy.deepcopy(base_cfg['library'])
            lib_cont['discrete_n'] = False
            lib_cont['n_grid'] = [1.0, 1.5, 2.0, 3.0, 4.0]
            cand_cont = build_candidates(Zf[st], lib_cont['K_grid'], lib_cont['n_grid'])
            
            models_cont = []
            for i in range(d):
                ch, _ = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, lib_cont, rng, cand=cand_cont)
                models_cont.append(ch['model'])
                
            struct_cont = structure_metrics(models_cont, eqs)
            Fs_cont = predict_all(models_cont, Zsup)
            cos_cont = cosine(Fs_cont, Ftrue_sup)
            nrmse_cont = nrmse(Fs_cont, Ftrue_sup)
            
            chosen_n_cont = [t['n'] for m in models_cont for t in m['terms'] if t['kind'] == 'hill']

            # Evaluate Regime 3: Matched discrete grid (contains n_true)
            lib_matched = copy.deepcopy(base_cfg['library'])
            lib_matched['discrete_n'] = True
            lib_matched['discrete_n_set'] = sorted(list(set([1.0, 2.0, 4.0, float(n_true)])))
            lib_matched['n_grid'] = sorted(list(set([1.0, 2.0, 4.0, float(n_true)])))
            cand_matched = build_candidates(Zf[st], lib_matched['K_grid'], lib_matched['n_grid'])
            
            models_matched = []
            for i in range(d):
                ch, _ = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, lib_matched, rng, cand=cand_matched)
                models_matched.append(ch['model'])
                
            struct_matched = structure_metrics(models_matched, eqs)
            Fs_matched = predict_all(models_matched, Zsup)
            cos_matched = cosine(Fs_matched, Ftrue_sup)
            nrmse_matched = nrmse(Fs_matched, Ftrue_sup)

            dt = time.time() - t0
            print(f"  [n_true={n_true} seed={seed}] ({dt:.1f}s) | NODE cos={node_cos:.3f} | "
                  f"DISCRETE {1,2,4}: F1={struct_disc['F1']:.2f} (P={struct_disc['precision']:.2f}, R={struct_disc['recall']:.2f}), "
                  f"K_err={struct_disc['param_relerr_median'].get('K')}, n_chosen={chosen_n_disc[:3]}... | "
                  f"CONT: F1={struct_cont['F1']:.2f}, K_err={struct_cont['param_relerr_median'].get('K')}, n_err={struct_cont['param_relerr_median'].get('n')}",
                  flush=True)
            
            all_results.append({
                'n_true': float(n_true),
                'seed': seed,
                'runtime_s': dt,
                'node_cos_support': node_cos,
                'discrete_1_2_4': {
                    'structure': struct_disc,
                    'cos_support': cos_disc,
                    'nrmse_support': nrmse_disc,
                    'chosen_n': chosen_n_disc,
                    'equations': describe(models_disc, [f'z{i}' for i in range(d)])
                },
                'continuous': {
                    'structure': struct_cont,
                    'cos_support': cos_cont,
                    'nrmse_support': nrmse_cont,
                    'chosen_n': chosen_n_cont,
                    'equations': describe(models_cont, [f'z{i}' for i in range(d)])
                },
                'matched_discrete': {
                    'structure': struct_matched,
                    'cos_support': cos_matched,
                    'nrmse_support': nrmse_matched,
                    'equations': describe(models_matched, [f'z{i}' for i in range(d)])
                }
            })

    # Summary analysis
    summary = {}
    for n_true in n_true_conditions:
        subset = [r for r in all_results if r['n_true'] == n_true]
        
        disc_f1 = [r['discrete_1_2_4']['structure']['F1'] for r in subset]
        disc_p = [r['discrete_1_2_4']['structure']['precision'] for r in subset]
        disc_r = [r['discrete_1_2_4']['structure']['recall'] for r in subset]
        disc_cos = [r['discrete_1_2_4']['cos_support'] for r in subset]
        disc_k_err = [r['discrete_1_2_4']['structure']['param_relerr_median']['K'] for r in subset if r['discrete_1_2_4']['structure']['param_relerr_median']['K'] is not None]
        
        cont_f1 = [r['continuous']['structure']['F1'] for r in subset]
        cont_cos = [r['continuous']['cos_support'] for r in subset]
        cont_k_err = [r['continuous']['structure']['param_relerr_median']['K'] for r in subset if r['continuous']['structure']['param_relerr_median']['K'] is not None]
        cont_n_err = [r['continuous']['structure']['param_relerr_median']['n'] for r in subset if r['continuous']['structure']['param_relerr_median']['n'] is not None]
        
        match_f1 = [r['matched_discrete']['structure']['F1'] for r in subset]
        match_cos = [r['matched_discrete']['cos_support'] for r in subset]
        
        summary[str(n_true)] = {
            'discrete_1_2_4': {
                'F1_mean': float(np.mean(disc_f1)),
                'F1_std': float(np.std(disc_f1)),
                'precision_mean': float(np.mean(disc_p)),
                'recall_mean': float(np.mean(disc_r)),
                'cos_support_mean': float(np.mean(disc_cos)),
                'K_relerr_median_mean': float(np.mean(disc_k_err)) if disc_k_err else None
            },
            'continuous': {
                'F1_mean': float(np.mean(cont_f1)),
                'F1_std': float(np.std(cont_f1)),
                'cos_support_mean': float(np.mean(cont_cos)),
                'K_relerr_median_mean': float(np.mean(cont_k_err)) if cont_k_err else None,
                'n_relerr_median_mean': float(np.mean(cont_n_err)) if cont_n_err else None
            },
            'matched_discrete': {
                'F1_mean': float(np.mean(match_f1)),
                'F1_std': float(np.std(match_f1)),
                'cos_support_mean': float(np.mean(match_cos))
            }
        }

    output_data = {
        'title': 'Discrete Exponent Misspecification Benchmark',
        'n_true_conditions': n_true_conditions,
        'seeds': seeds,
        'summary': summary,
        'runs': all_results,
        'total_time_seconds': time.time() - t_global_start
    }
    
    out_path = os.path.join(ROOT, 'docs', 'misspecification_n_results.json')
    with open(out_path, 'w') as f_out:
        json.dump(output_data, f_out, indent=2)
    print(f"\nSaved misspecification results to: {out_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("MISSPECIFICATION BENCHMARK SUMMARY TABLE", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Condition':<12} | {'Method':<20} | {'F1 Score':<16} | {'Field Cosine':<14} | {'K RelErr':<12} | {'n RelErr':<10}", flush=True)
    print("-" * 90, flush=True)
    for n_cond, s_cond in summary.items():
        d_ = s_cond['discrete_1_2_4']
        c_ = s_cond['continuous']
        m_ = s_cond['matched_discrete']
        k_err_d_str = f"{d_['K_relerr_median_mean']:.3f}" if d_['K_relerr_median_mean'] is not None else "--"
        k_err_c_str = f"{c_['K_relerr_median_mean']:.3f}" if c_['K_relerr_median_mean'] is not None else "--"
        n_err_c_str = f"{c_['n_relerr_median_mean']:.3f}" if c_['n_relerr_median_mean'] is not None else "--"
        
        print(f"n_true={n_cond:<6} | {'Discrete {1,2,4}':<20} | {d_['F1_mean']:.3f} +/- {d_['F1_std']:.3f}   | {d_['cos_support_mean']:.3f}        | {k_err_d_str:<12} | {'N/A':<10}", flush=True)
        print(f"{'':<12} | {'Continuous':<20} | {c_['F1_mean']:.3f} +/- {c_['F1_std']:.3f}   | {c_['cos_support_mean']:.3f}        | {k_err_c_str:<12} | {n_err_c_str:<10}", flush=True)
        print(f"{'':<12} | {'Matched Discrete':<20} | {m_['F1_mean']:.3f} +/- {m_['F1_std']:.3f}   | {m_['cos_support_mean']:.3f}        | {'--':<12} | {'0.000':<10}", flush=True)
        print("-" * 90, flush=True)
    print("=" * 75, flush=True)

if __name__ == '__main__':
    run_misspecification_experiment()
