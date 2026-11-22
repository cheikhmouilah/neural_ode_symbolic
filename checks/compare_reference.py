#!/usr/bin/env python3
"""Compare des resultats a une reference, fichier par fichier : identique / numerique (ecart relatif <= tol) / ecart (> tol) / structure / illisible. Les temps de calcul sont ignores.
   python checks/compare_reference.py REF_DIR NEW_DIR [--tol 1e-3]   ; code 1 s'il y a un ecart, une difference de structure ou un fichier illisible"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('ref'); ap.add_argument('new'); ap.add_argument('--tol', type=float, default=1e-3); a = ap.parse_args()
cnt = dict(identique=0, numerique=0, ecart=0, structure=0, illisible=0, absent=0); worst = []
for fr in result_files(a.ref):
    rel = os.path.relpath(fr, a.ref); fn = os.path.join(a.new, rel)
    if not os.path.exists(fn): cnt['absent'] += 1; print('[absent ]', rel); continue
    try: R, N = load_json(fr), load_json(fn)
    except Exception as e: cnt['illisible'] += 1; print('[illisible]', rel, str(e)[:80]); continue
    LR, LN = dict(leaves(R)), dict(leaves(N))
    if set(LR) != set(LN): cnt['structure'] += 1; print('[structure]', rel, f'cles differentes ({len(set(LR) ^ set(LN))})'); continue
    mx = 0.0; diff = False
    for p, v in LR.items():
        w = LN[p]
        if isnum(v) and isnum(w):
            if v != w: diff = True; mx = max(mx, abs(v - w) / max(abs(v), abs(w), 1e-12))
        elif v != w: cnt['structure'] += 1; print('[structure]', rel, p, v, '->', w); mx = None; break
    if mx is None: continue
    if not diff: cnt['identique'] += 1
    elif mx <= a.tol: cnt['numerique'] += 1; worst.append((mx, rel))
    else: cnt['ecart'] += 1; worst.append((mx, rel)); print(f'[ECART  ] {rel} : ecart relatif max {mx:.2e}')
print('bilan :', cnt)
if worst: print('plus grand ecart numerique :', max(worst))
sys.exit(1 if (cnt['ecart'] or cnt['structure'] or cnt['illisible']) else 0)
