import os, json, math, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIRS = ['project/configs', 'real_data/configs', 'synthetic/configs']
RESULT_DIRS = {'toggle': 'project/results/phase1', 'sergio': 'sergio_validation', 'real': 'real_data/results/real', 'synthetic': 'synthetic/results/synth'}
TIME_KEYS = {'seconds', 'seconds_total', 'node_seconds', 'elapsed_seconds', 'timings', 'minutes'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def leaves(o, path='', skip=TIME_KEYS):
    """Parcourt un JSON : (chemin, valeur) pour chaque feuille ; ignore les cles de temps de calcul."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in skip: continue
            yield from leaves(v, f'{path}/{k}', skip)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, f'{path}[{i}]', skip)
    else: yield path, o


def isnum(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


def bad_number(v): return isnum(v) and (math.isnan(v) or math.isinf(v))


def load_json(path):
    with open(path, 'rb') as f: raw = f.read()
    if not raw.strip(): raise ValueError('fichier vide (0 octet ou espaces)')
    return json.loads(raw.decode('utf-8'))


def result_files(d):
    out = []
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith('.json') and not f.startswith(('verification', 'verdicts', 'PREREG', 'run_everything')): out.append(os.path.join(dp, f))
    return sorted(out)
