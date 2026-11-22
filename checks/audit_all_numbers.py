"""
Audit script: extracts all key numbers from JSON outputs for manuscript correction.
"""
import json, glob, numpy as np

print("=" * 60)
print("LAMBDA SWEEP (Pancreas E15.5, 5 seeds)")
print("=" * 60)
d = json.load(open('docs/lambda_sweep_5seeds_summary.json'))
print("\nPer-lambda test cosine (linear baseline = 0.588):")
for k, v in sorted(d['summary_by_lambda'].items()):
    m = v['cos_test_mean']
    s = v['cos_test_std']
    cc = v['centered_cos_test_mean']
    ccs = v['centered_cos_test_std']
    nm = v['nrmse_test_mean']
    print(f"  lam={k}: cos={m:.3f}+/-{s:.3f}  centered={cc:.3f}+/-{ccs:.3f}  nrmse={nm:.4f}")

fs = d['frozen_selection_summary']
print(f"\nFrozen selection (val-based):")
print(f"  Selected lambdas: {fs['selected_lambdas']}")
print(f"  NODE cos test: {fs['node_test_cos_mean']:.4f} +/- {fs['node_test_cos_std']:.4f}")
print(f"  NODE centered cos: {fs['node_test_centered_cos_mean']:.4f} +/- {fs['node_test_centered_cos_std']:.4f}")
print(f"  NODE nrmse: {fs['node_test_nrmse_mean']:.5f} +/- {fs['node_test_nrmse_std']:.5f}")
print(f"  Sym vs node (R2 proxy): {fs['sym_cos_vs_node_mean']:.4f} +/- {fs['sym_cos_vs_node_std']:.4f}")
print(f"  Sym centered cos: {fs['sym_test_centered_cos_mean']:.4f} +/- {fs['sym_test_centered_cos_std']:.4f}")

print("\n" + "=" * 60)
print("DENTATE GYRUS (5 seeds)")
print("=" * 60)
files = sorted(glob.glob('real_data/results/real/real_dg_none_seed*.json'))
node_cos = []
lin_cos = []
sym_node = []
sym_meas = []
direct_meas = []
for f in files:
    r = json.load(open(f))
    nc = r.get('node_cos_test', None)
    lc = r.get('linear_baseline_cos_test', None)
    sn = r.get('sym_cos_vs_node_test', None)
    sm = r.get('sym_cos_vs_measured_test', None)
    dm = r.get('direct_cos_vs_measured_test', None)
    node_cos.append(nc)
    if lc is not None: lin_cos.append(lc)
    if sn is not None: sym_node.append(sn)
    if sm is not None: sym_meas.append(sm)
    if dm is not None: direct_meas.append(dm)
    nc_s = f"{nc:.3f}" if nc is not None else "N/A"
    lc_s = f"{lc:.3f}" if lc is not None else "N/A"
    sn_s = f"{sn:.3f}" if sn is not None else "N/A"
    sm_s = f"{sm:.3f}" if sm is not None else "N/A"
    dm_s = f"{dm:.3f}" if dm is not None else "N/A"
    print(f"  seed {r.get('seed','?')}: node={nc_s}  linear={lc_s}  sym~node={sn_s}  sym~meas={sm_s}  direct={dm_s}")

print(f"\nDG means:")
print(f"  NODE cos: {np.mean(node_cos):.4f} +/- {np.std(node_cos):.4f}")
if lin_cos: print(f"  Linear cos: {np.mean(lin_cos):.4f} +/- {np.std(lin_cos):.4f}")
if sym_node: print(f"  Sym~node (distil fidelity): {np.mean(sym_node):.4f} +/- {np.std(sym_node):.4f}")
if sym_meas: print(f"  Sym~meas: {np.mean(sym_meas):.4f} +/- {np.std(sym_meas):.4f}")
if direct_meas: print(f"  Direct~meas: {np.mean(direct_meas):.4f} +/- {np.std(direct_meas):.4f}")

print("\n" + "=" * 60)
print("NOISE ROBUSTNESS (FD F1 drop at sigma=20%)")
print("=" * 60)
f20 = sorted(glob.glob('synthetic/results/synth/synth_noise20_seed*.json'))
fb_files = sorted(glob.glob('synthetic/results/synth/synth_base_seed*.json'))
r20 = [json.load(open(f)) for f in f20]
rb = [json.load(open(f)) for f in fb_files]
fd20 = [r['direct_fd']['structure']['F1'] for r in r20 if 'direct_fd' in r]
fdb = [r['direct_fd']['structure']['F1'] for r in rb if 'direct_fd' in r]
print(f"  Base FD F1s: {[round(x,3) for x in fdb]}")
print(f"  Base mean: {np.mean(fdb):.4f} +/- {np.std(fdb):.4f}")
print(f"  Noise20 FD F1s: {[round(x,3) for x in fd20]}")
print(f"  Noise20 mean: {np.mean(fd20):.4f} +/- {np.std(fd20):.4f}")
print(f"  ACTUAL DROP: {np.mean(fdb)-np.mean(fd20):.4f}")

# Also check NODE in noise conditions
node20 = [r.get('node', {}).get('structure', {}).get('F1', None) for r in r20]
nodeb = [r.get('node', {}).get('structure', {}).get('F1', None) for r in rb]
node20 = [x for x in node20 if x is not None]
nodeb = [x for x in nodeb if x is not None]
if node20 and nodeb:
    print(f"  NODE base F1: {np.mean(nodeb):.4f}")
    print(f"  NODE noise20 F1: {np.mean(node20):.4f}")
    print(f"  NODE drop: {np.mean(nodeb)-np.mean(node20):.4f}")

print("\n" + "=" * 60)
print("MISSPECIFICATION (discrete n vs continuous)")
print("=" * 60)
ms = json.load(open('docs/misspecification_n_results.json'))
for cond in ms.get('conditions', ms.get('results', [])):
    nc = cond.get('n_true', '?')
    df = cond.get('discrete_f1_mean', cond.get('disc_f1_mean', '?'))
    ds = cond.get('discrete_f1_std', cond.get('disc_f1_std', '?'))
    cf = cond.get('continuous_f1_mean', cond.get('cont_f1_mean', '?'))
    cs = cond.get('continuous_f1_std', cond.get('cont_f1_std', '?'))
    print(f"  n_true={nc}: discrete={df:.3f}+/-{ds:.3f}  continuous={cf:.3f}+/-{cs:.3f}")

print("\nDONE")
