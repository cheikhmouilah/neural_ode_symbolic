#!/usr/bin/env python3
"""Integrite et provenance des resultats : JSON lisibles et non vides, aucune cle '_from_log', pas de NaN/Inf, empreinte de configuration coherente (chaines C et D).
   python checks/verify_results.py [--dirs d1 d2 ...] [--strict]   ; code 1 en cas d'echec (NaN : avertissement, echec avec --strict)"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--dirs', nargs='*'); ap.add_argument('--strict', action='store_true'); a = ap.parse_args()
dirs = a.dirs or [os.path.join(ROOT, v) for v in RESULT_DIRS.values()]
cfg_sha = {}
for d in CONFIG_DIRS:
    for f in glob.glob(os.path.join(ROOT, d, '*.json')): cfg_sha[sha256(f)[:16]] = os.path.basename(f)
prereg_file = os.path.join(ROOT, 'PREREGISTRATION.json')
if os.path.exists(prereg_file):
    try:
        pr = json.load(open(prereg_file))
        for p, sh in pr.get('configs', {}).items(): cfg_sha[sh[:16]] = os.path.basename(p) + " (prereg)"
    except Exception: pass
rows = []; fails = 0; warns = 0
for d in dirs:
    for f in result_files(d):
        name = os.path.relpath(f, ROOT); st = 'OK'; msgs = []
        try:
            J = load_json(f)
            if any(k.endswith('_from_log') or k == '_from_log' for p, _ in leaves(J) for k in p.split('/')): st = 'ECHEC'; msgs.append("cle '_from_log' : resultat reconstruit depuis un journal")
            nb = [p for p, v in leaves(J) if bad_number(v)]
            if nb: msgs.append(f'{len(nb)} valeur(s) NaN/Inf (ex. {nb[0]})'); st = 'ECHEC' if a.strict else ('AVERT' if st == 'OK' else st)
            if isinstance(J, dict) and 'config_sha' in J and J['config_sha'] not in cfg_sha: msgs.append(f"config_sha {J['config_sha']} ne correspond a aucune configuration actuelle (configuration modifiee depuis l'execution)"); st = 'AVERT' if st == 'OK' else st
            if isinstance(J, dict) and 'config_sha' in J and 'seed' not in J: msgs.append("pas de cle 'seed'"); st = 'ECHEC'
        except Exception as e: st = 'ECHEC'; msgs.append(str(e)[:120])
        fails += st == 'ECHEC'; warns += st == 'AVERT'; rows.append(dict(file=name, status=st, messages=msgs))
        if st != 'OK': print(f'[{st}] {name} : ' + ' ; '.join(msgs))
print(f'{len(rows)} fichier(s) examines : {len(rows) - fails - warns} OK, {warns} avertissement(s), {fails} echec(s)')
json.dump(dict(n=len(rows), failures=fails, warnings=warns, files=rows), open(os.path.join(ROOT, 'verification_report.json'), 'w'), indent=1)
sys.exit(1 if fails else 0)
