"""Experiment: Regularization Lambda Sweep on Pancreas across 5 Seeds
Validation-based hyperparameter selection with amplitude and centering metrics.
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'real_data', 'src'))
from real_data import load_prepared
from real_node import train_vel_node, field_np
from real_identify import build_candidates, select_structure, predict_all, describe, structure_set
from sklearn.neighbors import NearestNeighbors

def cosine(Fp, Ft, eps=1e-9):
    a, b = np.linalg.norm(Fp, axis=1), np.linalg.norm(Ft, axis=1)
    m = (a > eps) & (b > eps)
    if not np.any(m):
        return 0.0
    val = np.sum(Fp[m] * Ft[m], axis=1) / (a[m] * b[m])
    val = val[np.isfinite(val)]
    return float(np.mean(val)) if len(val) > 0 else 0.0

def centered_cosine(Fp, Ft, eps=1e-9):
    Fp_c = Fp - np.mean(Fp, axis=0, keepdims=True)
    Ft_c = Ft - np.mean(Ft, axis=0, keepdims=True)
    return cosine(Fp_c, Ft_c, eps=eps)

def nrmse(Fp, Ft):
    denom = Ft.max() - Ft.min()
    if denom < 1e-12: denom = 1.0
    return float(np.sqrt(np.mean((Fp - Ft) ** 2)) / denom)

def sample_Df(f, Ztr, rng, dist):
    M, k = dist['M'], dist['knn']
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Ztr)
    dk = nn.kneighbors(Ztr)[0][:, -1]
    thr = np.quantile(dk, dist['reject_quantile'])
    half = M // 2
    obs = Ztr[rng.choice(len(Ztr), half)]
    i = rng.choice(len(Ztr), M)
    cand = Ztr[i] + rng.normal(size=(M, Ztr.shape[1])) * (dist['jitter_frac'] * dk[i] / np.sqrt(Ztr.shape[1]))[:, None]
    cand = np.maximum(cand, 0)
    keep = nn.kneighbors(cand, n_neighbors=1)[0][:, 0] <= thr
    Zf = np.vstack([obs, cand[keep][:M - half]])
    return Zf

def run_experiment():
    print("=" * 70)
    print("STARTING REGULARIZATION LAMBDA SWEEP ACROSS 5 SEEDS (Pancreas E15.5)")
    print("=" * 70)
    t_start = time.time()
    
    cfg_path = os.path.join(ROOT, 'real_data', 'configs', 'real_pancreas.json')
    with open(cfg_path) as f_in:
        cfg = json.load(f_in)
        
    prep_path = os.path.join(ROOT, 'real_data', 'results', 'real', 'prepared.npz')
    if not os.path.exists(prep_path):
        prep_path = os.path.join(ROOT, 'real_data', 'results', 'real_quick', 'prepared.npz')
    print(f"Loading prepared data from: {prep_path}")
    P = load_prepared(prep_path)
    Z, V0, W, genes, clusters, split = P['Z'], P['V'], P['w'], P['genes'], P['clusters'], P['split']
    tr, va, te = [np.where(split == k)[0] for k in (0, 1, 2)]
    print(f"Dataset: {len(Z)} cells, {len(genes)} genes, train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    seeds = [0, 1, 2, 3, 4]
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]

    # Precompute linear baseline and shuffled control per seed
    linear_results = []
    shuffled_results = []
    for s in seeds:
        rng = np.random.default_rng(s)
        # Linear baseline
        A = np.column_stack([Z[tr], np.ones(len(tr))])
        Bl = np.linalg.lstsq(A, V0[tr], rcond=None)[0]
        F_lin = np.column_stack([Z[te], np.ones(len(te))]) @ Bl
        lin_cos = cosine(F_lin, V0[te])
        lin_c_cos = centered_cosine(F_lin, V0[te])
        lin_nrmse = nrmse(F_lin, V0[te])
        linear_results.append({
            'seed': s,
            'cos_test': lin_cos,
            'centered_cos_test': lin_c_cos,
            'nrmse_test': lin_nrmse
        })

        # Shuffled velocity control
        V_shuff = V0[np.random.default_rng(1000 + s).permutation(len(V0))]
        f_shuff, _ = train_vel_node(Z, V_shuff, W, tr, va, cfg['node'], s, lam_reg=0.001, log=lambda _: None)
        F_shuff = field_np(f_shuff, Z)
        shuff_cos = cosine(F_shuff[te], V0[te])
        shuff_c_cos = centered_cosine(F_shuff[te], V0[te])
        shuff_nrmse = nrmse(F_shuff[te], V0[te])
        shuffled_results.append({
            'seed': s,
            'cos_test': shuff_cos,
            'centered_cos_test': shuff_c_cos,
            'nrmse_test': shuff_nrmse
        })

    print(f"Linear baseline test cos: {np.mean([r['cos_test'] for r in linear_results]):.3f} +/- {np.std([r['cos_test'] for r in linear_results]):.3f}")
    print(f"Linear baseline test centered cos: {np.mean([r['centered_cos_test'] for r in linear_results]):.3f} +/- {np.std([r['centered_cos_test'] for r in linear_results]):.3f}")
    print(f"Shuffled control test cos: {np.mean([r['cos_test'] for r in shuffled_results]):.3f} +/- {np.std([r['cos_test'] for r in shuffled_results]):.3f}")
    print(f"Shuffled control test centered cos: {np.mean([r['centered_cos_test'] for r in shuffled_results]):.3f} +/- {np.std([r['centered_cos_test'] for r in shuffled_results]):.3f}")

    # Sweep grid: seed x lambda
    grid_runs = []
    selected_by_seed = {}

    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        best_val_cos = -1e9
        best_lam = None
        best_model_f = None
        
        for lam in lambdas:
            t0 = time.time()
            f, info = train_vel_node(Z, V0, W, tr, va, cfg['node'], s, lam_reg=lam, log=lambda _: None)
            F = field_np(f, Z)
            
            cos_tr = cosine(F[tr], V0[tr])
            cos_va = cosine(F[va], V0[va])
            c_cos_va = centered_cosine(F[va], V0[va])
            nrmse_va = nrmse(F[va], V0[va])
            
            cos_te = cosine(F[te], V0[te])
            c_cos_te = centered_cosine(F[te], V0[te])
            nrmse_te = nrmse(F[te], V0[te])
            dt = time.time() - t0
            
            run_rec = {
                'seed': s,
                'lambda': lam,
                'cos_val': cos_va,
                'centered_cos_val': c_cos_va,
                'nrmse_val': nrmse_va,
                'cos_test': cos_te,
                'centered_cos_test': c_cos_te,
                'nrmse_test': nrmse_te,
                'epochs': info['best_epoch'],
                'seconds': dt
            }
            grid_runs.append(run_rec)
            print(f"  seed={s} lam={lam:<5}: val_cos={cos_va:.3f} | test_cos={cos_te:.3f} (centered={c_cos_te:.3f}, nrmse={nrmse_te:.3f}) [{dt:.1f}s]")

            # Selection strictly on validation cosine
            if cos_va > best_val_cos:
                best_val_cos = cos_va
                best_lam = lam
                best_model_f = f

        print(f"  --> Seed {s} selected lambda* = {best_lam} (val_cos = {best_val_cos:.3f})")
        
        # Now run symbolic distillation on the selected model
        rng = np.random.default_rng(s)
        td0 = time.time()
        Zf = sample_Df(best_model_f, Z[tr], rng, cfg['distillation'])
        Ff = field_np(best_model_f, Zf)
        p = rng.permutation(len(Zf))
        nv = int(cfg['distillation']['val_fraction'] * len(Zf))
        sv, st = p[:nv], p[nv:]
        cand = build_candidates(Zf[st], cfg['library']['K_grid'], cfg['library']['n_grid'])
        models = []
        for i in range(Z.shape[1]):
            ch, _ = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, cfg['library'], rng, cand=cand)
            models.append(ch['model'])
        Fs = predict_all(models, Z[te])
        F_best_node = field_np(best_model_f, Z[te])
        sym_cos_vs_node = cosine(Fs, F_best_node)
        sym_cos_vs_meas = cosine(Fs, V0[te])
        sym_c_cos_vs_meas = centered_cosine(Fs, V0[te])
        sym_nrmse_vs_meas = nrmse(Fs, V0[te])
        dist_time = time.time() - td0
        print(f"  Distillation on selected lambda*: sym~node={sym_cos_vs_node:.3f}, sym~V={sym_cos_vs_meas:.3f} (centered={sym_c_cos_vs_meas:.3f}, nrmse={sym_nrmse_vs_meas:.3f}) [{dist_time:.1f}s]")

        # Find the test record for the chosen lambda
        chosen_rec = next(r for r in grid_runs if r['seed'] == s and r['lambda'] == best_lam)
        selected_by_seed[s] = {
            'selected_lambda': best_lam,
            'val_cos': best_val_cos,
            'test_cos': chosen_rec['cos_test'],
            'test_centered_cos': chosen_rec['centered_cos_test'],
            'test_nrmse': chosen_rec['nrmse_test'],
            'sym_cos_vs_node_test': sym_cos_vs_node,
            'sym_cos_vs_measured_test': sym_cos_vs_meas,
            'sym_centered_cos_vs_measured_test': sym_c_cos_vs_meas,
            'sym_nrmse_vs_measured_test': sym_nrmse_vs_meas,
            'equations': describe(models, genes)
        }

    # Summary statistics across seeds
    summary_by_lambda = {}
    for lam in lambdas:
        runs_lam = [r for r in grid_runs if r['lambda'] == lam]
        summary_by_lambda[str(lam)] = {
            'cos_val_mean': float(np.mean([r['cos_val'] for r in runs_lam])),
            'cos_val_std': float(np.std([r['cos_val'] for r in runs_lam])),
            'cos_test_mean': float(np.mean([r['cos_test'] for r in runs_lam])),
            'cos_test_std': float(np.std([r['cos_test'] for r in runs_lam])),
            'centered_cos_test_mean': float(np.mean([r['centered_cos_test'] for r in runs_lam])),
            'centered_cos_test_std': float(np.std([r['centered_cos_test'] for r in runs_lam])),
            'nrmse_test_mean': float(np.mean([r['nrmse_test'] for r in runs_lam])),
            'nrmse_test_std': float(np.std([r['nrmse_test'] for r in runs_lam]))
        }

    # Summary for frozen lambda* selected on validation
    frozen_test_cos = [v['test_cos'] for v in selected_by_seed.values()]
    frozen_test_c_cos = [v['test_centered_cos'] for v in selected_by_seed.values()]
    frozen_test_nrmse = [v['test_nrmse'] for v in selected_by_seed.values()]
    frozen_sym_cos = [v['sym_cos_vs_measured_test'] for v in selected_by_seed.values()]
    frozen_sym_c_cos = [v['sym_centered_cos_vs_measured_test'] for v in selected_by_seed.values()]
    frozen_sym_node = [v['sym_cos_vs_node_test'] for v in selected_by_seed.values()]

    frozen_summary = {
        'selected_lambdas': {str(s): v['selected_lambda'] for s, v in selected_by_seed.items()},
        'node_test_cos_mean': float(np.mean(frozen_test_cos)),
        'node_test_cos_std': float(np.std(frozen_test_cos)),
        'node_test_centered_cos_mean': float(np.mean(frozen_test_c_cos)),
        'node_test_centered_cos_std': float(np.std(frozen_test_c_cos)),
        'node_test_nrmse_mean': float(np.mean(frozen_test_nrmse)),
        'node_test_nrmse_std': float(np.std(frozen_test_nrmse)),
        'sym_cos_vs_node_mean': float(np.mean(frozen_sym_node)),
        'sym_cos_vs_node_std': float(np.std(frozen_sym_node)),
        'sym_test_cos_mean': float(np.mean(frozen_sym_cos)),
        'sym_test_cos_std': float(np.std(frozen_sym_cos)),
        'sym_test_centered_cos_mean': float(np.mean(frozen_sym_c_cos)),
        'sym_test_centered_cos_std': float(np.std(frozen_sym_c_cos)),
    }

    linear_summary = {
        'cos_test_mean': float(np.mean([r['cos_test'] for r in linear_results])),
        'cos_test_std': float(np.std([r['cos_test'] for r in linear_results])),
        'centered_cos_test_mean': float(np.mean([r['centered_cos_test'] for r in linear_results])),
        'centered_cos_test_std': float(np.std([r['centered_cos_test'] for r in linear_results])),
        'nrmse_test_mean': float(np.mean([r['nrmse_test'] for r in linear_results])),
        'nrmse_test_std': float(np.std([r['nrmse_test'] for r in linear_results]))
    }

    shuffled_summary = {
        'cos_test_mean': float(np.mean([r['cos_test'] for r in shuffled_results])),
        'cos_test_std': float(np.std([r['cos_test'] for r in shuffled_results])),
        'centered_cos_test_mean': float(np.mean([r['centered_cos_test'] for r in shuffled_results])),
        'centered_cos_test_std': float(np.std([r['centered_cos_test'] for r in shuffled_results])),
        'nrmse_test_mean': float(np.mean([r['nrmse_test'] for r in shuffled_results])),
        'nrmse_test_std': float(np.std([r['nrmse_test'] for r in shuffled_results]))
    }

    full_output = {
        'protocol': '5-seed regularization sweep with validation-based freeze, uncentered and centered cosine, NRMSE, and distillation',
        'seeds': seeds,
        'lambdas': lambdas,
        'summary_by_lambda': summary_by_lambda,
        'frozen_selection_summary': frozen_summary,
        'linear_baseline_summary': linear_summary,
        'shuffled_control_summary': shuffled_summary,
        'selected_by_seed': selected_by_seed,
        'grid_runs': grid_runs,
        'linear_runs': linear_results,
        'shuffled_runs': shuffled_results,
        'total_time_seconds': time.time() - t_start
    }

    out_file = os.path.join(ROOT, 'docs', 'lambda_sweep_5seeds_summary.json')
    with open(out_file, 'w') as f_out:
        json.dump(full_output, f_out, indent=2)
    print(f"\nSaved comprehensive summary to: {out_file}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY: 5-SEED REGULARIZATION SWEEP")
    print("=" * 70)
    print(f"{'Lambda':<10} | {'Val Cosine':<16} | {'Test Cosine':<16} | {'Test Centered Cos':<18} | {'Test NRMSE':<14}")
    print("-" * 75)
    for lam in lambdas:
        s_lam = summary_by_lambda[str(lam)]
        print(f"{str(lam):<10} | {s_lam['cos_val_mean']:.3f} +/- {s_lam['cos_val_std']:.3f}  | {s_lam['cos_test_mean']:.3f} +/- {s_lam['cos_test_std']:.3f}  | {s_lam['centered_cos_test_mean']:.3f} +/- {s_lam['centered_cos_test_std']:.3f}    | {s_lam['nrmse_test_mean']:.3f} +/- {s_lam['nrmse_test_std']:.3f}")
    print("-" * 75)
    print(f"{'FROZEN lam*':<10} | {'(val selected)':<16} | {frozen_summary['node_test_cos_mean']:.3f} +/- {frozen_summary['node_test_cos_std']:.3f}  | {frozen_summary['node_test_centered_cos_mean']:.3f} +/- {frozen_summary['node_test_centered_cos_std']:.3f}    | {frozen_summary['node_test_nrmse_mean']:.3f} +/- {frozen_summary['node_test_nrmse_std']:.3f}")
    print(f"{'Linear':<10} | {'--':<16} | {linear_summary['cos_test_mean']:.3f} +/- {linear_summary['cos_test_std']:.3f}  | {linear_summary['centered_cos_test_mean']:.3f} +/- {linear_summary['centered_cos_test_std']:.3f}    | {linear_summary['nrmse_test_mean']:.3f} +/- {linear_summary['nrmse_test_std']:.3f}")
    print(f"{'Shuffled':<10} | {'--':<16} | {shuffled_summary['cos_test_mean']:.3f} +/- {shuffled_summary['cos_test_std']:.3f}  | {shuffled_summary['centered_cos_test_mean']:.3f} +/- {shuffled_summary['centered_cos_test_std']:.3f}    | {shuffled_summary['nrmse_test_mean']:.3f} +/- {shuffled_summary['nrmse_test_std']:.3f}")
    print("=" * 70)
    print(f"Distilled Symbolic on frozen lam*: cos vs meas = {frozen_summary['sym_test_cos_mean']:.3f} +/- {frozen_summary['sym_test_cos_std']:.3f} | centered = {frozen_summary['sym_test_centered_cos_mean']:.3f} +/- {frozen_summary['sym_test_centered_cos_std']:.3f} | sym~node = {frozen_summary['sym_cos_vs_node_mean']:.3f} +/- {frozen_summary['sym_cos_vs_node_std']:.3f}")
    print("=" * 70)

if __name__ == '__main__':
    run_experiment()
