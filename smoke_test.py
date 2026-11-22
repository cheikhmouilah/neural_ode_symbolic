#!/usr/bin/env python3
"""Test de fonctionnement (quelques minutes) AVANT les executions completes : chaque chaine tourne en mode --quick (resultats NON interpretables, ecrits dans des dossiers *_quick)
et les garde-fous sont verifies par des TESTS NEGATIFS (ils doivent echouer sur des entrees volontairement fautives).
   <python du venv> smoke_test.py [--skip-toggle] [--skip-real] [--skip-sergio-check] [--skip-pysr]"""
import os, sys, json, time, shutil, tempfile, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
ap = argparse.ArgumentParser(); ap.add_argument('--skip-toggle', action='store_true'); ap.add_argument('--skip-real', action='store_true'); ap.add_argument('--skip-sergio-check', action='store_true'); ap.add_argument('--skip-pysr', action='store_true', help='ne pas tester PySR (le projet complet l inclut)'); ap.add_argument('--with-pysr', action='store_true', help=argparse.SUPPRESS); a = ap.parse_args()
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NODE_THREADS='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
rows = []


def run(cmd, cwd=HERE, timeout=900):
    p = subprocess.run([PY] + cmd, cwd=cwd, env=ENV, capture_output=True,
                       text=True, encoding='utf-8', errors='replace', timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def test(name, fn):
    t0 = time.time()
    try: ok, detail = fn()
    except Exception as e: ok, detail = False, f'exception : {str(e)[:200]}'
    rows.append(dict(test=name, ok=bool(ok), seconds=round(time.time() - t0, 1), detail=detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.time() - t0:.0f} s) {'' if ok else '-> ' + detail}", flush=True)


tmp = tempfile.mkdtemp(prefix='smoke_')
# ---- 1. garde-fous : tests negatifs ----
def t_guard():
    rc, out = run(['-c', "import sys; sys.path.insert(0,'project/src')\nfrom toggle import field\nfrom guards import assert_bistable, GuardError\nassert_bistable(lambda x: field(x), 'ok')\n"
                   "p=dict(a=(3.0,2.6),K=(0.8,1.0),n=2.0,g=(1.0,0.9))\ntry:\n    assert_bistable(lambda x: field(x,p),'monostable')\n    print('PAS DE REFUS'); sys.exit(2)\nexcept GuardError: print('refus attendu')"])
    return rc == 0 and 'refus attendu' in out, out.strip()[-120:]
test('garde-fou de bistabilite : accepte le toggle, refuse un systeme monostable', t_guard)
def t_verify_neg():
    d = os.path.join(tmp, 'bad'); os.makedirs(d)
    open(os.path.join(d, 'empty.json'), 'w').close(); json.dump({'a': 1, '_from_log': True}, open(os.path.join(d, 'fromlog.json'), 'w')); open(os.path.join(d, 'nan.json'), 'w').write('{"x": NaN}')
    rc, out = run(['checks/verify_results.py', '--dirs', d, '--strict']); return rc == 1 and 'ECHEC' in out, out.strip()[-160:]
test('verify_results : detecte fichier vide, cle _from_log, NaN', t_verify_neg)
def t_consistency_neg():
    d = os.path.join(tmp, 'synth'); os.makedirs(d); rng = __import__('random').Random(0)
    for v, c in (('noise02', 0.50), ('base', 0.70), ('noise10', 0.80), ('noise20', 0.90)):      # le cosinus S'AMELIORE avec le bruit : incoherent
        for s in range(3): json.dump(dict(variant=v, seed=s, node_metrics=dict(cos_vs_truth_support=c + 0.01 * rng.random(), traj_nrmse_vs_truth=0.02 + 0.01 * rng.random()), distilled=dict(structure=dict(F1=0.8 + 0.1 * rng.random()), cos_vs_truth_support=0.9 + 0.01 * rng.random()),
                                          direct_fd=dict(structure=dict(F1=0.8 + 0.1 * rng.random())), weak_sindy=dict(structure=dict(F1=0.8 + 0.1 * rng.random()))), open(os.path.join(d, f'synth_{v}_seed{s}.json'), 'w'))
    rc, out = run(['checks/consistency.py', '--synth', d, '--real', tmp, '--toggle', tmp]); return rc == 1 and 'bruit -> cosinus NODE' in out and 'ECHEC' in out, out.strip()[-200:]
test('consistency : detecte un resultat qui s ameliore avec le bruit', t_consistency_neg)
test('preenregistrement : verify renvoie un code valide (0 conforme, 2 pas de gel)', lambda: ((lambda r: (r[0] in (0, 2), r[1].strip()[-120:]))(run(['checks/prereg.py', 'verify']))))
test('compare_reference : la reference du toggle est identique a elle-meme', lambda: ((lambda r: (r[0] == 0 and 'identique' in r[1], r[1].strip()[-100:]))(run(['checks/compare_reference.py', 'reference_results/toggle', 'reference_results/toggle']))))
# ---- 2. chaines en mode rapide ----
if not a.skip_toggle:
    test('chaine A (toggle) : run_all.py toggle --quick + verification automatique de la reference', lambda: ((lambda r: (r[0] == 0 and 'OK' in r[1].split('VERIFICATION AUTOMATIQUE')[-1], r[1].strip()[-250:]))(run(['run_all.py', 'toggle', '--quick'], timeout=1500))))
test('chaine D (synthetique) : base, graine 0, --quick', lambda: ((lambda r: (r[0] == 0 and os.path.exists(os.path.join(HERE, 'synthetic/results/synth_quick/synth_base_seed0.json')), r[1].strip()[-200:]))(run(['synthetic/scripts/run_synth.py', '--variant', 'base', '--seed', '0', '--quick', '--force', '--free-form', 'gplearn']))))
if not a.skip_real:
    for key, cfgf, tag in (('pancreas', 'real_pancreas.json', 'real'), ('gyrus denele', 'real_dentategyrus.json', 'real_dg')):
        def t_real(cfgf=cfgf, tag=tag):
            rc, out = run(['real_data/scripts/run_real.py', '--config', cfgf, '--tag', tag, '--seeds', '0', '--quick', '--force'], timeout=900)
            f = os.path.join(HERE, 'real_data/results/real_quick', f'{tag}_none_seed0.json'); return rc == 0 and os.path.exists(f), out.strip()[-250:]
        test(f'chaine C (donnees reelles, {key}) : graine 0, --quick (verifie le SHA-256 du jeu)', t_real)
    # Baselines VII-D en mode quick (dynamo, CellRank, BoolODE)
    def t_baselines():
        rc, out = run(['real_data/scripts/run_real_baselines.py', '--config', 'real_pancreas.json', '--tag', 'real', '--quick', '--force'], timeout=300)
        f = os.path.join(HERE, 'real_data/results/real_quick', 'baselines_real.json')
        return rc == 0 and os.path.exists(f), out.strip()[-250:]
    test('chaine C (baselines VII-D) : dynamo, CellRank, BoolODE --quick', t_baselines)
if not a.skip_sergio_check:
    def t_sergio():
        rc, out = run(['-c', "import sys,os; p=os.path.join('sergio_validation','SERGIO'); assert os.path.isdir(os.path.join(p,'SERGIO')), 'SERGIO absent : python setup_all.py'; sys.path.insert(0,p); import SERGIO.sergio as m; print('SERGIO importable')"])
        return rc == 0, out.strip()[-160:]
    test('chaine B (SERGIO) : depot present et importable (la simulation complete dure ~5 min : non lancee ici)', t_sergio)
if not a.skip_pysr:
    test('PySR : import et mini-recherche (Julia installe)', lambda: ((lambda r: (r[0] == 0 and 'ok' in r[1], r[1].strip()[-200:]))(run(['-c', "import numpy as np; from pysr import PySRRegressor; X=np.random.RandomState(0).randn(100,2); y=X[:,0]**2\nm=PySRRegressor(niterations=2,binary_operators=['+','*'],unary_operators=[],parallelism='serial',deterministic=True,random_state=0,progress=False,verbosity=0,temp_equation_file=False,output_directory=__import__('tempfile').gettempdir()); m.fit(X,y); print('ok', m.get_best()['equation'])"], timeout=1500))))
    test('PySR chaine A (toggle) : --quick', lambda: ((lambda r: (r[0] == 0 and os.path.exists(os.path.join(HERE, 'project/results/pysr_quick/pysr_toggle_seed0.json')), r[1].strip()[-250:]))(run(['scripts/run_pysr_toggle.py', '--seeds', '0', '--quick'], cwd=os.path.join(HERE, 'project'), timeout=1500))))
    test('PySR chaine D (synthetique) : --quick', lambda: ((lambda r: (r[0] == 0 and 'free_form_pysr' in open(os.path.join(HERE, 'synthetic/results/synth_quick/synthpysr_base_seed0.json')).read() and 'skipped' not in open(os.path.join(HERE, 'synthetic/results/synth_quick/synthpysr_base_seed0.json')).read(), r[1].strip()[-250:]))(run(['synthetic/scripts/run_synth.py', '--variant', 'base', '--seed', '0', '--quick', '--free-form', 'pysr', '--tag', 'synthpysr', '--force'], timeout=1500))))
    test('PySR chaine C (donnees reelles) : --quick, 2 equations', lambda: ((lambda r: (r[0] == 0 and os.path.exists(os.path.join(HERE, 'real_data/results/real_quick/real_pysr_seed0.json')), r[1].strip()[-250:]))(run(['real_data/scripts/run_real_pysr.py', '--quick', '--seeds', '0', '--equations', '0', '1'], timeout=1500))))
test('figures et tableaux : make_figures.py sur les resultats quick', lambda: ((lambda r: (r[0] == 0, r[1].strip()[-160:]))(run(['make_figures.py', '--real-dir', 'real_data/results/real_quick', '--synth-dir', 'synthetic/results/synth_quick', '--out', os.path.join(tmp, 'fig')]))))
n_ok = sum(r['ok'] for r in rows); json.dump(rows, open(os.path.join(HERE, 'smoke_report.json'), 'w'), indent=1, ensure_ascii=False); shutil.rmtree(tmp, ignore_errors=True)
print(f'\n{n_ok}/{len(rows)} tests reussis (rapport : smoke_report.json)'); print('Les valeurs des executions --quick ne sont PAS interpretables ; seul le fonctionnement est teste.')
sys.exit(0 if n_ok == len(rows) else 1)
