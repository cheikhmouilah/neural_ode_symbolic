#!/usr/bin/env python3
"""Preenregistrement : gel et verification des fichiers de configuration (seuils) par SHA-256.
   python checks/prereg.py freeze                      # a lancer AVANT les executions (refuse d'ecraser)
   python checks/prereg.py amend "motif de la modification"   # enregistre un amendement date et motive (conserve l'ancienne empreinte)
   python checks/prereg.py verify                      # compare l'etat actuel au gel ; code 1 si une configuration a change sans amendement
Limite : le fichier PREREGISTRATION.json est auto-declare. Pour une preuve externe, versionnez-le dans git (git add / commit / tag) ou horodatez-le avec un service tiers AVANT les executions."""
import os, sys, json, glob, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, CONFIG_DIRS, sha256
PATH = os.path.join(ROOT, 'PREREGISTRATION.json')


def current():
    out = {}
    for d in CONFIG_DIRS:
        for f in sorted(glob.glob(os.path.join(ROOT, d, '*.json'))): out[os.path.relpath(f, ROOT).replace('\\', '/')] = sha256(f)
    return out


def git_head():
    try: return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: return None


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'verify'; cur = current(); now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if cmd == 'freeze':
        if os.path.exists(PATH): sys.exit('PREREGISTRATION.json existe deja : utiliser  amend "motif"  pour une modification, ou verify.')
        json.dump(dict(frozen_utc=now, git_head=git_head(), configs=cur, amendments=[], note='auto-declare ; ancrer dans git ou un horodatage externe'), open(PATH, 'w'), indent=1)
        print(f'gel enregistre : {len(cur)} configurations ({now}). Ancrer maintenant : git add PREREGISTRATION.json && git commit && git tag prereg'); return 0
    if not os.path.exists(PATH): print('aucun gel : lancer  python checks/prereg.py freeze  avant les executions'); return 2
    P = json.load(open(PATH))
    expected = dict(P['configs'])
    for a in P['amendments']: expected.update(a['new'])
    if cmd == 'amend':
        if len(sys.argv) < 3: sys.exit('motif requis : amend "motif"')
        diff = {k: dict(old=expected.get(k), new=v) for k, v in cur.items() if expected.get(k) != v}
        if not diff: print('aucune difference a amender'); return 0
        P['amendments'].append(dict(utc=now, reason=sys.argv[2], changes={k: v['old'] for k, v in diff.items()}, new={k: v['new'] for k, v in diff.items()}))
        json.dump(P, open(PATH, 'w'), indent=1); print(f'amendement enregistre ({len(diff)} fichier(s)) : {sys.argv[2]}'); return 0
    bad = [k for k, v in cur.items() if k in expected and expected[k] != v]; new = [k for k in cur if k not in expected]; gone = [k for k in expected if k not in cur]
    print(f"gel du {P['frozen_utc']} ; amendements : {len(P['amendments'])}")
    for k in bad: print('  MODIFIE sans amendement :', k)
    for k in new: print('  NOUVEAU (absent du gel) :', k)
    for k in gone: print('  ABSENT :', k)
    if not (bad or gone): print('  configurations conformes au gel' + (' (fichiers nouveaux signales)' if new else ''))
    return 1 if (bad or gone) else 0


if __name__ == '__main__': sys.exit(main())
