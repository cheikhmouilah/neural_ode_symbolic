"""
Phase 2 — Pipeline Real Data: Erythroid / Bonemarrow
Adapte le pipeline existant (run_real.py) pour le dataset gastrulation_erythroid ou bonemarrow.
Produit: real_data/results/real/real_erythroid_none_seed{0-4}.json
"""
import os, sys, json, hashlib, numpy as np, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'real_data', 'src'))

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', choices=['erythroid', 'bonemarrow'], default='erythroid')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--force', action='store_true')
args = ap.parse_args()

OUT_DIR = os.path.join(ROOT, 'real_data', 'results', 'real')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, f'real_{args.dataset}_none_seed{args.seed}.json')

if os.path.exists(OUT_FILE) and not args.force:
    print(f'[SKIP] {OUT_FILE}'); sys.exit(0)

print(f"=== Phase 2: {args.dataset} seed={args.seed} ===")
t0 = time.time()

# ---- 1. Load dataset via scVelo ----
import scvelo as scv
import scanpy as sc

print("Loading dataset...")
if args.dataset == 'erythroid':
    adata = scv.datasets.gastrulation_erythroid()
    config_file = os.path.join(ROOT, 'real_data', 'configs', 'real_erythroid.json')
    progenitor_cluster = 'Erythroid1'
    dataset_key = 'celltype'
elif args.dataset == 'bonemarrow':
    adata = scv.datasets.bonemarrow()
    config_file = os.path.join(ROOT, 'real_data', 'configs', 'real_bonemarrow.json')
    progenitor_cluster = 'Progenitors'
    dataset_key = 'clusters'

cfg = json.load(open(config_file))
print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# ---- 2. RNA velocity preprocessing (same as pancreas pipeline) ----
print("Computing RNA velocity...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Check if spliced/unspliced already computed
if 'spliced' not in adata.layers and 'Ms' not in adata.layers:
    print("  Layers missing — dataset may not support scVelo velocity. Checking...")
    print("  Available layers:", list(adata.layers.keys()))
    sys.exit(1)

# Preprocessing
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
scv.tl.velocity(adata, mode='deterministic')

# SHA-256 of the h5ad file (if cached)
cache_dir = os.path.expanduser('~/.cache/scvelo')
sha256 = 'computed-from-cache'
for dirpath, _, filenames in os.walk(cache_dir):
    for fn in filenames:
        if args.dataset in fn.lower() and fn.endswith('.h5ad'):
            fp = os.path.join(dirpath, fn)
            sha256 = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
            print(f"  SHA-256: {sha256}")
            break

# ---- 3. Gene selection (top d by Ms variance) ----
d = cfg['state']['d']
bio_priority = cfg['state'].get('biological_priority', [])

import scvelo as scv
velocity_genes = adata.var_names[adata.var.get('velocity_genes', np.ones(adata.n_vars, dtype=bool))].tolist()
Ms = adata.layers['Ms']
vars_in_vg = [g for g in adata.var_names if g in velocity_genes]

# Score by variance
var_idx = {g: i for i, g in enumerate(adata.var_names)}
vg_idx = [var_idx[g] for g in vars_in_vg if g in var_idx]
vg_vars = np.var(Ms[:, vg_idx], axis=0)
ranked = sorted(zip(vars_in_vg, vg_vars), key=lambda x: -x[1])

# Hybrid selection: biology first, then variance
selected = []
for g in bio_priority:
    if g in vars_in_vg and g not in selected:
        selected.append(g)
        if len(selected) >= min(3, d):
            break
# Fill with top-variance
for g, _ in ranked:
    if g not in selected:
        selected.append(g)
    if len(selected) >= d:
        break

print(f"  Selected {d} genes: {selected}")

# ---- 4. Run the actual pipeline (import real_data.py functions) ----
from real_data import prepare_real_data
from real_node import train_real_node, field_np
from real_identify import distill_real, describe

# Prepare data
print("Preparing data...")
result_data = prepare_real_data(adata, selected, cfg, seed=args.seed)
Z, V, W = result_data['Z'], result_data['V'], result_data['W']
idx_tr, idx_va, idx_te = result_data['idx_tr'], result_data['idx_va'], result_data['idx_te']

# Linear baseline
from sklearn.linear_model import Ridge
lin = Ridge(alpha=1.0).fit(Z[idx_tr], V[idx_tr])
lin_cos_te = float(np.mean([
    np.dot(lin.predict(Z[idx_te[i:i+1]]).ravel(), V[idx_te[i]]) /
    (np.linalg.norm(lin.predict(Z[idx_te[i:i+1]])) * np.linalg.norm(V[idx_te[i]]) + 1e-9)
    for i in range(len(idx_te))
]))

# NODE
print("Training Neural ODE...")
node_cfg = cfg['node']
node_cfg['lam_reg'] = 0.001
f_node, node_info = train_real_node(Z[idx_tr], V[idx_tr], Z[idx_va], V[idx_va], W[idx_tr], W[idx_va], node_cfg, seed=args.seed)

# Compute metrics
Fn_te = field_np(f_node, Z[idx_te])
cosines_te = [np.dot(Fn_te[i], V[idx_te[i]]) / (np.linalg.norm(Fn_te[i]) * np.linalg.norm(V[idx_te[i]]) + 1e-9) for i in range(len(idx_te))]
node_cos_te = float(np.mean(cosines_te))

# Distillation
print("Distilling...")
dist_cfg = cfg['distillation']
lib_cfg = cfg['library']
rng = np.random.default_rng(args.seed)
models, sym_info = distill_real(f_node, Z[idx_tr], dist_cfg, lib_cfg, rng)

# Fidelity (sym ~ node)
from real_identify import field_sym
Zall = Z
Fs = field_sym(models, Zall)
Fn = field_np(f_node, Zall)
cos_sym_node = float(np.mean([np.dot(Fs[i], Fn[i]) / (np.linalg.norm(Fs[i]) * np.linalg.norm(Fn[i]) + 1e-9) for i in range(len(Zall))]))

# sym ~ measured on test
Fs_te = field_sym(models, Z[idx_te])
cos_sym_meas = float(np.mean([np.dot(Fs_te[i], V[idx_te[i]]) / (np.linalg.norm(Fs_te[i]) * np.linalg.norm(V[idx_te[i]]) + 1e-9) for i in range(len(idx_te))]))

# Save result
result = {
    'dataset': args.dataset,
    'seed': args.seed,
    'sha256': sha256,
    'genes': selected,
    'n_cells': int(adata.n_obs),
    'n_train': len(idx_tr),
    'n_test': len(idx_te),
    'linear_baseline_cos_test': lin_cos_te,
    'node_cos_test': node_cos_te,
    'sym_cos_vs_node_test': cos_sym_node,
    'sym_cos_vs_measured_test': cos_sym_meas,
    'node_info': node_info,
    'equations': describe(models, selected),
    'config': cfg,
}
json.dump(result, open(OUT_FILE, 'w'), indent=2)
print(f"\nSaved: {OUT_FILE}")
print(f"  node_cos_test:     {node_cos_te:.4f}")
print(f"  linear_baseline:   {lin_cos_te:.4f}")
print(f"  sym~node fidelity: {cos_sym_node:.4f}")
print(f"  sym~measured:      {cos_sym_meas:.4f}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
