#!/usr/bin/env python3
"""Installation et telechargement de TOUT ce qu'il faut pour lancer les trois chaines (toggle, SERGIO, donnees reelles).

  python setup_all.py                 # venv .venv + dependances + jeu pancreas + SERGIO (commit fige) + verification
  python setup_all.py --check         # ne telecharge rien : verifie seulement ce qui est deja en place
  python setup_all.py --no-pysr       # n'installe PAS PySR/Julia (par defaut ils sont installes : ~2-3 Go, plusieurs minutes)
  python setup_all.py --no-venv       # installe dans l'interpreteur courant (conda, Colab...)
  python setup_all.py --skip-pip      # saute l'installation des paquets (telecharge seulement les donnees)

Ensuite : <python du venv> run_everything.py   (le chemin exact est affiche a la fin). Rapport : setup_report.json.
Sources : jeux pancreas et gyrus dentele = depot theislab/scvelo_notebooks (GitHub, SHA-256 dans datasets.json) ; SERGIO = PayamDiba/SERGIO au commit a6190b74 (teste).
Teste ici : telechargement + verification SHA-256, clone partiel de SERGIO, application du correctif numpy. NON teste ici : creation du venv,
index PyTorch CPU (download.pytorch.org), PySR/Julia (domaines bloques dans l'environnement de redaction)."""
import os, sys, json, hashlib, shutil, argparse, platform, subprocess, urllib.request, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = json.load(open(os.path.join(HERE, 'datasets.json')))   # source unique : URL, fichier, SHA-256 (utilise aussi par make_report.py)
SERGIO_URL = 'https://github.com/PayamDiba/SERGIO.git'; SERGIO_COMMIT = 'a6190b74425112834c8fa9b4b6157d9cb3d1ab88'
SERGIO_DIR = os.path.join(HERE, 'sergio_validation', 'SERGIO'); PATCH = os.path.join(HERE, 'sergio_validation', 'sergio_numpy_compat.patch')
SERGIO_DATASET = 'data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6'
REQUIRED = ['numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'sympy', 'torch', 'torchdiffeq', 'pysindy', 'gplearn', 'scanpy', 'anndata', 'scvelo']

ap = argparse.ArgumentParser()
ap.add_argument('--check', action='store_true'); ap.add_argument('--no-pysr', action='store_true', help='ne pas installer PySR/Julia (le projet complet les inclut)'); ap.add_argument('--with-pysr', action='store_true', help=argparse.SUPPRESS); ap.add_argument('--no-venv', action='store_true')
ap.add_argument('--skip-pip', action='store_true'); ap.add_argument('--cuda-torch', action='store_true', help='ne pas forcer le torch CPU (Linux/Windows : les roues par defaut embarquent CUDA, ~3 Go de plus)')
args = ap.parse_args(); report = dict(platform=platform.platform(), python=sys.version.split()[0], steps={})


def step(name, ok, detail=''):
    report['steps'][name] = dict(ok=bool(ok), detail=detail); print(('[OK]    ' if ok else '[ECHEC] ') + name + (f' : {detail}' if detail else ''), flush=True)
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


ENV = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


def run(cmd, **kw): return subprocess.run(cmd, env=ENV, **kw)


def rm_readonly(func, path, _):
    import stat; os.chmod(path, stat.S_IWRITE); func(path)          # Windows : les fichiers .git sont en lecture seule


def venv_python():
    return os.path.join(HERE, '.venv', 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')


def in_venv(): return sys.prefix != getattr(sys, 'base_prefix', sys.prefix) or bool(os.environ.get('CONDA_PREFIX'))


def ensure_python():
    py = sys.executable
    if args.no_venv or in_venv(): return py
    vp = venv_python()
    if not os.path.exists(vp):
        print('creation du venv .venv ...', flush=True); run([sys.executable, '-m', 'venv', os.path.join(HERE, '.venv')], check=True)
    return vp


def install_packages(py):
    pipv = [py, '-m', 'pip']
    run(pipv + ['install', '-q', '--upgrade', 'pip'])
    if not args.cuda_torch and platform.system() != 'Darwin':
        r = run(pipv + ['install', 'torch>=2.14', '--index-url', 'https://download.pytorch.org/whl/cpu'])
        step('torch CPU (index officiel PyTorch)', r.returncode == 0, 'echec : relancer avec --cuda-torch ou installer torch a la main' if r.returncode else '')
    r = run(pipv + ['install', '-r', os.path.join(HERE, 'requirements.txt')]); step('pip install -r requirements.txt', r.returncode == 0)
    if not args.no_pysr:
        r = run(pipv + ['install', 'pysr']); ok = r.returncode == 0
        if ok:
            print("premier import de PySR : installe Julia et ses paquets (necessite internet ; ~1 a 2 Go ; plusieurs minutes, parfois 10+) ...", flush=True)
            test = ("import time, numpy as np; t=time.time(); from pysr import PySRRegressor; X=np.random.RandomState(0).randn(200,2); y=X[:,0]**2-1.0\n"
                    "m=PySRRegressor(niterations=3, binary_operators=['+','-','*'], unary_operators=[], parallelism='serial', deterministic=True, random_state=0, progress=False, verbosity=0, temp_equation_file=False, output_directory=__import__('tempfile').gettempdir())\n"
                    "m.fit(X,y); print('PySR fonctionne :', m.get_best()['equation'], '(%.0f s)' % (time.time()-t))")
            r = run([py, '-c', test], capture_output=True, text=True); ok = r.returncode == 0
            print((r.stdout or '').strip()[-300:] if ok else (r.stderr or r.stdout)[-1200:])
            if not ok: print("Si l'installation automatique de Julia echoue (reseau d'entreprise, pare-feu) : installer Julia a la main (https://julialang.org/downloads ou  conda install -c conda-forge julia), puis definir PYTHON_JULIAPKG_EXE=<chemin de julia> avant de relancer. Voir GUIDE_PYSR.md.")
        step('PySR + Julia (test fonctionnel)', ok, 'recherche symbolique libre (scripts *_pysr, run_everything)' if ok else 'voir GUIDE_PYSR.md, section Depannage ; sinon relancer avec --no-pysr')


def download_one(key, d):
    path = os.path.join(HERE, d['file']); os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and sha256(path) == d['sha256']: return step(f'jeu {key} (deja present, SHA-256 verifie)', True, f"{os.path.getsize(path)/1e6:.1f} Mo")
    if args.check: return step(f'jeu {key}', False, 'absent ou corrompu (lancer sans --check)')
    for attempt in range(1, 4):
        try:
            print(f"telechargement du jeu {key} (tentative {attempt}/3, ~{d['bytes']/1e6:.0f} Mo) ...", flush=True)
            urllib.request.urlretrieve(d['url'], path + '.part'); got = sha256(path + '.part')
            if got == d['sha256']: os.replace(path + '.part', path); return step(f'jeu {key} (SHA-256 verifie)', True, f"{os.path.getsize(path)/1e6:.1f} Mo")
            print(f'  SHA-256 different ({got[:16]}...). Le fichier amont a pu changer.', flush=True)
        except Exception as e: print('  echec :', e, flush=True)
        time.sleep(3)
    return step(f'jeu {key}', False, 'telechargement ou empreinte en echec ; telecharger a la main : ' + d['url'] + f' -> {path}')


def download_datasets():
    return all([download_one(k, v) for k, v in DATASETS.items()])


def setup_sergio():
    if not shutil.which('git'): return step('SERGIO', False, 'git requis : Windows -> winget install Git.Git (ou https://git-scm.com), puis relancer')
    ok_ds = os.path.isdir(os.path.join(SERGIO_DIR, SERGIO_DATASET))
    if os.path.isdir(os.path.join(SERGIO_DIR, '.git')) and ok_ds:
        head = run(['git', 'rev-parse', 'HEAD'], cwd=SERGIO_DIR, capture_output=True, text=True).stdout.strip()
        patched = run(['git', 'apply', '--reverse', '--check', PATCH], cwd=SERGIO_DIR, capture_output=True).returncode == 0
        return step('SERGIO (deja present)', head == SERGIO_COMMIT and patched, f'commit {head[:8]}, correctif {"applique" if patched else "ABSENT"}')
    if args.check: return step('SERGIO', False, 'absent (lancer sans --check)')
    os.makedirs(os.path.dirname(SERGIO_DIR), exist_ok=True)
    if os.path.isdir(SERGIO_DIR) and not os.path.isdir(os.path.join(SERGIO_DIR, '.git')): shutil.rmtree(SERGIO_DIR, onerror=rm_readonly)
    try:
        if not os.path.isdir(SERGIO_DIR): subprocess.check_call(['git', '-c', 'core.autocrlf=false', 'clone', '-q', '--filter=blob:none', '--sparse', SERGIO_URL, SERGIO_DIR]); subprocess.check_call(['git', 'config', 'core.autocrlf', 'false'], cwd=SERGIO_DIR)   # depot ~2,5 Go : clone partiel
        subprocess.check_call(['git', 'sparse-checkout', 'set', 'SERGIO', SERGIO_DATASET], cwd=SERGIO_DIR)
        subprocess.check_call(['git', 'checkout', '-q', SERGIO_COMMIT], cwd=SERGIO_DIR)
        if run(['git', 'apply', '--reverse', '--check', PATCH], cwd=SERGIO_DIR, capture_output=True).returncode != 0: subprocess.check_call(['git', 'apply', PATCH], cwd=SERGIO_DIR)
        return step('SERGIO (commit fige + correctif numpy)', True, SERGIO_COMMIT[:8])
    except Exception as e: return step('SERGIO', False, str(e))


def verify(py):
    code = ("import importlib, importlib.util, json\nout = {}\nfor m in %r:\n    try:\n        mod = importlib.import_module(m); out[m] = getattr(mod, '__version__', '?')\n"
            "    except Exception as e: out[m] = 'MANQUANT : ' + str(e)[:80]\n"
            "if %r: out['pysr'] = 'present (non importe ici : Julia demarre a l import)' if importlib.util.find_spec('pysr') else 'MANQUANT : python setup_all.py'\n"
            "print(json.dumps(out))") % (REQUIRED, not args.no_pysr)
    r = run([py, '-c', code], capture_output=True, text=True)
    try: vers = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: return step('verification des paquets', False, (r.stderr or r.stdout)[-300:])
    missing = [m for m, v in vers.items() if v.startswith('MANQUANT')]
    report['packages'] = vers; step('verification des paquets', not missing, ', '.join(f'{m} {v}' for m, v in vers.items() if m in missing) or 'tous importables')
    if not missing:
        r = run([py, '-c', 'import torch; x=torch.randn(4,4); print(float((x@x).sum()) == float((x@x).sum()))'], capture_output=True, text=True)
        step('torch fonctionne (calcul CPU)', r.returncode == 0)
    return not missing


if sys.version_info < (3, 10): sys.exit('Python >= 3.10 requis (teste : 3.12 ; sous Windows : python.org, cocher \"Add python.exe to PATH\").')
if os.name == 'nt' and len(HERE) > 70: print(f'[ATTENTION] chemin long ({len(HERE)} car.) : sous Windows (limite 260), installer le projet dans un dossier court, ex. C:\\neural_ode_symbolic')
print(f'systeme : {platform.system()} {platform.release()} | coeurs : {os.cpu_count()} | python : {sys.version.split()[0]}', flush=True); report['cpu_count'] = os.cpu_count()
py = ensure_python() if not args.check else (sys.executable if (args.no_venv or in_venv() or not os.path.exists(venv_python())) else venv_python())
report['interpreter'] = py
if not args.check and not args.skip_pip: install_packages(py)
download_datasets(); setup_sergio(); verify(py)
json.dump(report, open(os.path.join(HERE, 'setup_report.json'), 'w'), indent=1)
bad = [k for k, v in report['steps'].items() if not v['ok']]
print('\n' + ('TOUT EST PRET.' if not bad else 'A CORRIGER : ' + '; '.join(bad)))
print('Lancer ensuite :', f'"{py}" run_everything.py' + ('   (ajouter --with-pysr n\'a pas d\'effet : PySR n\'est pas utilise)' if args.with_pysr else ''))
sys.exit(1 if bad else 0)
