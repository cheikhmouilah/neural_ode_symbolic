"""
Script de critique complète du projet: lit tous les JSONs de résultats
et imprime un tableau synthétique par méthode et condition.
"""
import json, glob, numpy as np

def safe_f1(r, key):
    v = r.get(key, {}).get('structure', {}).get('F1', None)
    return v

def safe_cos(r, key):
    v = r.get(key, {}).get('cos', None)
    if v is None:
        v = r.get(key, {}).get('cosine', None)
    return v

print("=" * 70)
print("TABLEAU COMPLET — TOUTES MÉTHODES, TOUTES CONDITIONS")
print("=" * 70)

conditions = [
    ('base',      'synth_base'),
    ('noise5',    'synth_noise5'),
    ('noise10',   'synth_noise10'),
    ('noise15',   'synth_noise15'),
    ('noise20',   'synth_noise20'),
    ('dropout10', 'synth_dropout10'),
    ('dropout20', 'synth_dropout20'),
    ('dropout40', 'synth_dropout40'),
]

methods = ['sym', 'node', 'direct_fd', 'sindy', 'pysr']

for label, pattern in conditions:
    files = sorted(glob.glob(f'synthetic/results/synth/{pattern}_seed*.json'))
    if not files:
        continue
    rs = [json.load(open(f)) for f in files]
    row = [label]
    for m in methods:
        vals = [safe_f1(r, m) for r in rs]
        vals = [v for v in vals if v is not None]
        if vals:
            row.append(f"{np.mean(vals):.3f}")
        else:
            row.append("N/A")
    print(f"  {row[0]:15s} sym={row[1]}  node={row[2]}  FD={row[3]}  SINDy={row[4]}  PySR={row[5]}")

print()
print("=" * 70)
print("CRISPR PERTURB-SEQ (par cible)")
print("=" * 70)
try:
    d = json.load(open('docs/perturb_seq_results.json'))
    targets = d.get('targets', [])
    node_cos_all = []
    hill_cos_all = []
    lin_cos_all = []
    for t in targets:
        name = t.get('target', '?')
        hc = t.get('hill', {}).get('cosine', None)
        nc = t.get('node', {}).get('cosine', None)
        lc = t.get('linear', {}).get('cosine', None)
        if hc: hill_cos_all.append(hc)
        if nc: node_cos_all.append(nc)
        if lc: lin_cos_all.append(lc)
        print(f"  {name:12s}: hill={hc:.3f}  node={nc:.3f}  linear={lc:.3f}")
    print(f"  {'MEAN':12s}: hill={np.mean(hill_cos_all):.3f}  node={np.mean(node_cos_all):.3f}  linear={np.mean(lin_cos_all):.3f}")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 70)
print("MISSPECIFICATION COMPLÈTE")
print("=" * 70)
try:
    ms = json.load(open('docs/misspecification_n_results.json'))
    summary = ms.get('summary', {})
    for n_true, conds in summary.items():
        disc = conds.get('discrete_1_2_4', {})
        cont = conds.get('continuous', {})
        matched = conds.get('matched_discrete', {})
        df = disc.get('F1_mean', None)
        ds = disc.get('F1_std', None)
        cf = cont.get('F1_mean', None)
        cs = cont.get('F1_std', None)
        mf = matched.get('F1_mean', None)
        print(f"  n_true={n_true}: discrete={df:.3f}+/-{ds:.3f}  continuous={cf:.3f}+/-{cs:.3f}  matched={mf:.3f}")
except Exception as e:
    print(f"Error: {e}")
