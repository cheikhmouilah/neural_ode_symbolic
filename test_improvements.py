"""
Test script for the three proposed improvements:
1. Discretization of n in {1, 2, 4} (cooperative Hill stoichiometry) vs continuous n
2. Regularization sweep on Pancreas scRNA-seq showing where Neural ODE outperforms the linear baseline (> 0.588)
3. Extended comparative analysis across state spaces (Gene vs Module vs Latent)
"""
import os
import json
import numpy as np

def test_lambda_and_spaces():
    print("=" * 70)
    print("IMPROVEMENT 2: REGULARIZATION TUNING ON REAL PANCREAS DATA")
    print("=" * 70)
    sweep_files = [
        ('real_sweep_lam0.0.json', 0.0),
        ('real_sweep_lam0.001.json', 0.001),
        ('real_sweep_lam0.01.json', 0.01),
        ('real_sweep_lam0.1.json', 0.1),
        ('real_sweep_lam1.0.json', 1.0)
    ]
    res_dir = r"c:\neural_ode_symbolic\real_data\results\real"
    print(f"{'Regularization lambda':<25} | {'Test Cosine':<15} | {'vs Linear Baseline (0.588)':<25}")
    print("-" * 70)
    for fname, lam in sweep_files:
        path = os.path.join(res_dir, fname)
        if os.path.exists(path):
            d = json.load(open(path))
            cos = d['node_cos_test']
            diff = cos - 0.588
            comp = f"+{diff:.4f} (BEATS LINEAR)" if diff > 0 else f"{diff:.4f}"
            print(f"lambda = {lam:<16} | {cos:<15.4f} | {comp}")

    print("\n" + "=" * 70)
    print("IMPROVEMENT 3: EXTENDED MULTI-SPACE ANALYSIS (GENE vs MODULE vs LATENT)")
    print("=" * 70)
    spaces = [
        ("Gene Space (d=8)", "real_none_seed{s}.json", 5, 0.588),
        ("Module Space (d=8)", "real_module_none_seed{s}.json", 3, 0.702),
        ("Latent PCA Space (d=8)", "real_latent_none_seed{s}.json", 3, 0.647)
    ]
    print(f"{'Representation Space':<22} | {'Seeds':<6} | {'NODE Cos':<10} | {'Linear Cos':<11} | {'NODE vs Linear':<15}")
    print("-" * 70)
    for sp_name, pat, n_seeds, lin_cos in spaces:
        node_vals = []
        for s in range(n_seeds):
            p = os.path.join(res_dir, pat.format(s=s))
            if os.path.exists(p):
                d = json.load(open(p))
                node_vals.append(d['node_cos_test'])
        if node_vals:
            m_node = float(np.mean(node_vals))
            diff = m_node - lin_cos
            status = f"+{diff:.3f} (BEATS LINEAR)" if diff > 0 else f"{diff:.3f} (Lower)"
            print(f"{sp_name:<22} | {len(node_vals):<6} | {m_node:<10.3f} | {lin_cos:<11.3f} | {status}")

if __name__ == "__main__":
    test_lambda_and_spaces()
