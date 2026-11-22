"""
Phase 3 — Repositionnement manuscrit
Script de validation : vérifie que les affirmations du manuscrit sont
cohérentes avec les résultats réels (après Phase 1 et Phase 2).
Génère un rapport de conformité.
"""
import json, glob, numpy as np, os, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("PHASE 3 AUDIT — Manuscript vs Actual Results")
print(f"Run: {datetime.now().isoformat()}")
print("=" * 70)

issues = []
confirmations = []

# ---- 1. Dropout advantage check (Phase 1 results) ----
print("\n[1] DROPOUT ADVANTAGE (Phase 1 hypothesis)")
for variant in ['drop10', 'drop20', 'drop40', 'drop60']:
    files = sorted(glob.glob(os.path.join(ROOT, f'synthetic/results/synth/synth_{variant}_seed*.json')))
    if not files:
        print(f"  {variant}: NO DATA YET")
        continue
    sym_f1s, fd_f1s = [], []
    for f in files:
        r = json.load(open(f))
        sf = r.get('distilled', {}).get('structure', {}).get('F1')
        ff = r.get('direct_fd', {}).get('structure', {}).get('F1')
        if sf: sym_f1s.append(sf)
        if ff: fd_f1s.append(ff)
    if sym_f1s and fd_f1s:
        diff = np.mean(sym_f1s) - np.mean(fd_f1s)
        status = "✅ SYM WINS" if diff > 0.05 else ("⚠️  TIED" if abs(diff) < 0.05 else "❌ FD WINS")
        print(f"  {variant}: sym={np.mean(sym_f1s):.3f}  FD={np.mean(fd_f1s):.3f}  diff={diff:+.3f}  {status}")
        if diff > 0.05:
            confirmations.append(f"Distillation > FD at {variant} (+{diff:.3f})")
        elif diff < -0.05:
            issues.append(f"FD > Distillation at {variant} ({diff:.3f}) — update manuscript")

# ---- 2. Erythroid dataset check (Phase 2 results) ----
print("\n[2] ERYTHROID DATASET (Phase 2)")
erythroid_files = sorted(glob.glob(os.path.join(ROOT, 'real_data/results/real/real_erythroid_none_seed*.json')))
if erythroid_files:
    node_cos = [json.load(open(f))['node_cos_test'] for f in erythroid_files]
    lin_cos = [json.load(open(f))['linear_baseline_cos_test'] for f in erythroid_files]
    fid = [json.load(open(f))['sym_cos_vs_node_test'] for f in erythroid_files]
    print(f"  Seeds: {len(erythroid_files)}")
    print(f"  NODE cos: {np.mean(node_cos):.3f} ± {np.std(node_cos):.3f}")
    print(f"  Linear:   {np.mean(lin_cos):.3f}")
    print(f"  Fidelity: {np.mean(fid):.3f} ± {np.std(fid):.3f}")
    if np.mean(node_cos) >= 0.50:
        confirmations.append(f"Erythroid NODE cos {np.mean(node_cos):.3f} >= 0.50 threshold")
    else:
        issues.append(f"Erythroid NODE cos {np.mean(node_cos):.3f} < 0.50 — below threshold")
    if np.mean(fid) >= 0.90:
        confirmations.append(f"Erythroid distillation fidelity {np.mean(fid):.3f} >= 0.90")
    else:
        issues.append(f"Erythroid fidelity {np.mean(fid):.3f} < 0.90 — below threshold")
else:
    print("  NO DATA YET — run run_phase2_erythroid.py")

# ---- 3. Pre-registration anchor check ----
print("\n[3] PRE-REGISTRATION ANCHOR")
prereg = json.load(open(os.path.join(ROOT, 'PREREGISTRATION.json')))
git_head = prereg.get('git_head')
zenodo = prereg.get('zenodo_doi')
amendments = prereg.get('amendments', [])
print(f"  git_head: {git_head or 'MISSING ❌'}")
print(f"  zenodo_doi: {zenodo or 'PENDING ⚠️'}")
print(f"  amendments: {len(amendments)}")
if git_head:
    confirmations.append(f"Pre-registration anchored to git {git_head[:8]}")
else:
    issues.append("Pre-registration git_head is null — PHASE 0 incomplete")
if zenodo == 'PENDING':
    issues.append("Zenodo DOI still pending — upload manually at zenodo.org")

# ---- 4. Manuscript framing check ----
print("\n[4] MANUSCRIPT FRAMING")
ms_file = os.path.join(ROOT, 'docs', 'IEEE_SECTION_IMPLEMENTATION_AND_CONCLUSION.md')
with open(ms_file, 'r', encoding='utf-8') as f:
    ms = f.read()

checks = [
    ("decisively outperforms", "❌ OVERCLAIMING — 'decisively' should be removed"),
    ("actually outperforming", "❌ OVERCLAIMING — 'actually outperforming' should be softened"),
    ("H. Hochgerner", "✅ Author name correct"),
    ("exploratory", "✅ Exploratory label present"),
    ("NODE below linear", "✅ DG limitation documented"),
    ("within 1 s.d.", "✅ Statistical caveat present"),
]
for phrase, verdict in checks:
    found = phrase.lower() in ms.lower()
    sym = "✅" if ("✅" in verdict and found) or ("❌" in verdict and not found) else "⚠️"
    print(f"  {sym} '{phrase}': {'FOUND' if found else 'NOT FOUND'}  → {verdict}")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY")
print(f"  Confirmations: {len(confirmations)}")
for c in confirmations:
    print(f"    ✅ {c}")
print(f"  Issues: {len(issues)}")
for i in issues:
    print(f"    ⚠️  {i}")
print("=" * 70)
