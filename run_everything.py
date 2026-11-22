#!/usr/bin/env python3
"""Lance TOUTES les chaines en parallele sur N coeurs (defaut : min(4, nb de coeurs)), puis les analyses.
Chaque tache = un processus a 1 coeur (1 graine / 1 configuration) : pas de multiprocessing interne, donc sans risque sous Windows.
Reprise automatique (taches deja produites sautees ; --force pour tout recalculer). Journaux : logs/<tache>.log.

  <python du venv> run_everything.py                  # tout, 4 coeurs
  <python du venv> run_everything.py --workers 3      # moins de coeurs / moins de RAM
  <python du venv> run_everything.py --only real      # une seule chaine : toggle | sergio | real
  <python du venv> run_everything.py --dry-run        # liste les taches sans rien lancer
  <python du venv> run_everything.py --filter "real_ctrl"   # sous-ensemble (expression reguliere sur le nom des taches)

Duree indicative sur 4 coeurs : de l'ordre de 1 h 30 a 2 h au total, PySR inclus (estimation ; non mesure d'un bout a l'autre ; mesure partielle : PySR toggle ~2 min par graine, PySR reel ~80 s pour 8 equations, hors entrainement). Figures : figures_article/.
Les resultats peuvent differer a la 3e-4e decimale de ceux de reference (nombre de threads, versions de bibliotheques)."""
import os, sys, re, time, json, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
P, R_, S, SY = (os.path.join(HERE, d) for d in ('project', 'real_data', 'sergio_validation', 'synthetic'))
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1)); ap.add_argument('--only', nargs='+', choices=['toggle', 'sergio', 'real', 'synthetic'], default=['toggle', 'sergio', 'real', 'synthetic'])
ap.add_argument('--strict', action='store_true', help='les controles de coherence non evaluables comptent comme echec'); ap.add_argument('--allow-config-change', action='store_true', help='ne pas arreter si une configuration differe du gel (a documenter par  prereg.py amend)'); ap.add_argument('--no-pysr', action='store_true', help='NE PAS lancer les taches PySR (le projet complet les inclut ; voir GUIDE_PYSR.md)'); ap.add_argument('--pysr', action='store_true', help=argparse.SUPPRESS); ap.add_argument('--force', action='store_true'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--filter', default=None)
ap.add_argument('--sergio-parallel', type=int, default=2, help='taches SERGIO simultanees max (memoire : la simulation demande ~2 Go)')
a = ap.parse_args()

ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
           NODE_THREADS='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')


class Job:
    def __init__(s, name, args, cwd, out, deps=(), group='', prio=5):
        s.name, s.args, s.cwd, s.out, s.deps, s.group, s.prio = name, args, cwd, out if isinstance(out, (list, tuple)) else [out], set(deps), group, prio

    def done(s): return all(os.path.exists(f) for f in s.out)


def build():
    J = []; R1 = P + '/results/phase1'; add = J.append
    if 'toggle' in a.only:
        for s in (0, 1, 2):
            add(Job(f'toggle_traj_s{s}', ['scripts/run_phase1.py', '--seeds', str(s), '--tag', 'traj'], P, f'{R1}/phase1_traj_seed{s}.json', prio=3))
            add(Job(f'toggle_vel_s{s}', ['scripts/run_phase1.py', '--seeds', str(s), '--velocity', '1.0', '--tag', 'vel'], P, f'{R1}/phase1_vel_seed{s}.json', prio=3))
            for tag, noise, nt in (('n05', '0.05', '40'), ('n10', '0.10', '40'), ('N15', '0.0', '15')):
                add(Job(f'toggle_{tag}_s{s}', ['scripts/run_phase6.py', '--seeds', str(s), '--tag', tag, '--noise', noise, '--ntrain', nt], P, f'{R1}/phase1_{tag}_seed{s}.json', prio=4))
            for k in ('hill', 'leaky', 'gauss'):
                add(Job(f'toggle_misspec_{k}_s{s}', ['scripts/run_misspec.py', '--kinds', k, '--seeds', str(s)], P, f'{R1}/misspec_{k}_seed{s}.json', prio=5))
            for X in ('C1', 'C2', 'C3', 'M1', 'M2', 'M3', 'M4'):
                add(Job(f'toggle_detect_{X}_s{s}', ['scripts/run_detect_test.py', '--systems', X, '--seeds', str(s)], P, f'{R1}/detect_{X}_seed{s}.json', prio=6))
    if 'sergio' in a.only:
        add(Job('sergio_sim', ['run_sim.py'], S, [S + '/sergio_clean.npz', S + '/sergio_noisy.npz'], group='sergio', prio=1))
        for w in ('clean', 'noisy_mild', 'noisy_severe', 'control'): add(Job(f'sergio_validate_{w}', ['validate.py', w], S, f'{S}/result_{w}.json', ['sergio_sim'], 'sergio', 2))
        add(Job('sergio_identifiability', ['identifiability.py'], S, S + '/result_identifiability.json', ['sergio_sim'], 'sergio', 2))
        add(Job('sergio_steady_state', ['steady_state_diagnostic.py'], S, S + '/result_steady_state_diag.json', ['sergio_sim'], 'sergio', 2))
        add(Job('sergio_eiv_synthetic', ['eiv_synthetic.py'], S, S + '/result_eiv.json', [], 'sergio', 2))
        add(Job('sergio_eiv_test', ['eiv_test_sergio.py'], S, S + '/result_eiv_sergio.json', ['sergio_sim'], 'sergio', 2))
    if 'real' in a.only:
        O = R_ + '/results/real'; cfg = json.load(open(R_ + '/configs/real_pancreas.json'))
        add(Job('real_prepare', ['scripts/run_real.py', '--prepare-only'], R_, O + '/prepared.npz', prio=0))
        for s in cfg['stability']['seeds']:
            add(Job(f'real_seed{s}', ['scripts/run_real.py', '--seeds', str(s), '--tag', 'real'], R_, f'{O}/real_none_seed{s}.json', ['real_prepare'], prio=2))
            add(Job(f'real_ctrl{s}', ['scripts/run_real.py', '--seeds', str(s), '--tag', 'real', '--control', 'shuffle', '--no-dyn'], R_, f'{O}/real_shuffle_seed{s}.json', ['real_prepare'], prio=4))
        for lam in cfg['stability']['lam_sweep']:
            add(Job(f'real_sweep{lam}', ['scripts/run_real.py', '--sweep', '--lam', str(lam), '--tag', 'real'], R_, f'{O}/real_sweep_lam{lam}.json', ['real_prepare'], prio=2))
        # Baselines VII-D : dynamo, CellRank, BoolODE (s'exécutent après le prétraitement)
        add(Job('real_baselines', ['scripts/run_real_baselines.py', '--config', 'real_pancreas.json', '--tag', 'real'], R_, f'{O}/baselines_real.json', ['real_prepare'], prio=3))
        add(Job('real_prepare_dg', ['scripts/run_real.py', '--config', 'real_dentategyrus.json', '--tag', 'real_dg', '--prepare-only'], R_, f'{O}/prepared_dg.npz', prio=0))
        for s in json.load(open(f'{R_}/configs/real_dentategyrus.json'))['stability']['seeds'][:3]: add(Job(f'real_dg_seed{s}', ['scripts/run_real.py', '--config', 'real_dentategyrus.json', '--tag', 'real_dg', '--seeds', str(s)], R_, f'{O}/real_dg_none_seed{s}.json', ['real_prepare_dg'], prio=3))
        for sp in ('module', 'latent'):                     # espaces d'etat alternatifs (II-C), 3 graines
            c2 = json.load(open(f'{R_}/configs/real_pancreas_{sp}.json')); cf = f'real_pancreas_{sp}.json'; pass
            add(Job(f'real_prepare_{sp}', ['scripts/run_real.py', '--config', cf, '--tag', f'real_{sp}', '--prepare-only'], R_, f'{O}/prepared_{sp}.npz', prio=0))
            for s in c2['stability']['seeds'][:3]: add(Job(f'real_{sp}_seed{s}', ['scripts/run_real.py', '--config', cf, '--tag', f'real_{sp}', '--seeds', str(s)], R_, f'{O}/real_{sp}_none_seed{s}.json', [f'real_prepare_{sp}'], prio=3))
    if 'synthetic' in a.only:
        SC = json.load(open(SY + '/configs/synth.json')); OS = SY + '/results/synth'
        for v in SC['variants']:
            for s in SC['seeds']:
                add(Job(f'synth_{v}_s{s}', ['scripts/run_synth.py', '--variant', v, '--seed', str(s)] + (['--free-form', 'gplearn'] if v == 'base' else []), SY, f'{OS}/synth_{v}_seed{s}.json', prio=3 if v in ('base', 'pseudo2', 'd8') else 5))
    if not a.no_pysr:       # PySR (recherche libre) : un seul import (installation des paquets Julia) AVANT les taches, pour eviter des installations concurrentes
        wk = HERE + '/results/pysr_warmup.ok'; PCt = json.load(open(P + '/configs/pysr.json')); PCr = json.load(open(R_ + '/configs/real_pysr.json'))
        add(Job('pysr_warmup', ['-c', "import os, pysr; os.makedirs('results', exist_ok=True); open('results/pysr_warmup.ok', 'w').write(pysr.__version__)"], HERE, wk, prio=0))
        if 'toggle' in a.only:
            for s in PCt['toggle']['seeds']: add(Job(f'toggle_pysr_s{s}', ['scripts/run_pysr_toggle.py', '--seeds', str(s)], P, f'{P}/results/phase1/pysr_toggle_seed{s}.json', ['pysr_warmup'], prio=6))
        if 'synthetic' in a.only:
            for s in json.load(open(SY + '/configs/synth.json'))['seeds']: add(Job(f'synthpysr_base_s{s}', ['scripts/run_synth.py', '--variant', 'base', '--seed', str(s), '--free-form', 'pysr', '--tag', 'synthpysr'], SY, f'{SY}/results/synth/synthpysr_base_seed{s}.json', ['pysr_warmup'], prio=6))
        if 'real' in a.only:
            for s in PCr['seeds']: add(Job(f'real_pysr_s{s}', ['scripts/run_real_pysr.py', '--seeds', str(s)], R_, f'{R_}/results/real/real_pysr_seed{s}.json', ['pysr_warmup', 'real_prepare'], prio=6))
            add(Job('real_pysr_dg_s0', ['scripts/run_real_pysr.py', '--config', 'real_dentategyrus.json', '--tag', 'real_pysr_dg', '--seeds', '0'], R_, f'{R_}/results/real/real_pysr_dg_seed0.json', ['pysr_warmup', 'real_prepare_dg'], prio=6))
    if a.filter: J = [j for j in J if re.search(a.filter, j.name)]
    names = {j.name for j in J}
    for j in J: j.deps &= names                 # dependances hors sous-ensemble : supposees deja satisfaites
    return J


def run_logged(cmd, cwd):
    """Lance une analyse/un controle : affiche la sortie ET l'enregistre dans logs/analyses.txt (sinon elle ne vivrait que dans la console)."""
    p = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, encoding='utf-8', errors='replace'); txt = (p.stdout or '') + (p.stderr or '')
    print(txt, end='', flush=True)
    with open(os.path.join(HERE, 'logs', 'analyses.txt'), 'a', encoding='utf-8') as f: f.write('\n########## ' + ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd[1:])) + f'  (code {p.returncode})\n' + txt)
    return p.returncode


def main():
    jobs = build()
    if not jobs: sys.exit('aucune tache')
    todo = [j for j in jobs if a.force or not j.done()]
    print(f'{len(jobs)} taches, {len(todo)} a executer, {a.workers} coeur(s) ; python = {PY}', flush=True)
    if a.dry_run:
        for j in sorted(jobs, key=lambda j: j.prio): print(('  [a faire] ' if j in todo else '  [existe]  ') + j.name + '  deps=' + ','.join(sorted(j.deps)))
        return 0
    pr = subprocess.run([PY, os.path.join(HERE, 'checks', 'prereg.py'), 'verify'], capture_output=True, text=True, env=ENV)
    if pr.returncode == 2:
        print('Aucun gel de preenregistrement : creation automatique (auto-declare ; ancrez-le ensuite dans git).', flush=True); print(subprocess.run([PY, os.path.join(HERE, 'checks', 'prereg.py'), 'freeze'], capture_output=True, text=True, env=ENV).stdout.strip())
    elif pr.returncode == 1:
        print(pr.stdout)
        if not a.allow_config_change: sys.exit("Une configuration differe du gel : annuler la modification, ou la declarer (python checks/prereg.py amend \"motif\"), ou relancer avec --allow-config-change.")
    if not a.no_pysr and not __import__('importlib.util').util.find_spec('pysr'): sys.exit("PySR n'est pas installe (le projet complet l'inclut) : python setup_all.py   ; ou relancer avec --no-pysr.")
    chk = subprocess.run([PY, os.path.join(HERE, 'setup_all.py'), '--check', '--no-venv'], capture_output=True, text=True, env=ENV)
    if chk.returncode != 0 and not a.filter: print(chk.stdout[-1500:]); sys.exit("Environnement incomplet : lancer d'abord  python setup_all.py")
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    pending = sorted(todo, key=lambda j: j.prio); done = {j.name for j in jobs if j not in todo}; failed = set(); running = {}; log = {}; t_all = time.time()
    try:
        while pending or running:
            for j in list(pending):
                if j.deps & failed: pending.remove(j); failed.add(j.name); print(f'[saute : dependance en echec] {j.name}', flush=True); continue
                if not j.deps <= done or len(running) >= a.workers: continue
                if j.group == 'sergio' and sum(1 for r in running.values() if r[0].group == 'sergio') >= a.sergio_parallel: continue
                lf = open(os.path.join(HERE, 'logs', j.name + '.log'), 'w', encoding='utf-8')
                for f in j.out:
                    if a.force and os.path.exists(f): os.remove(f)
                running[j.name] = (j, subprocess.Popen([PY, '-u'] + j.args, cwd=j.cwd, stdout=lf, stderr=subprocess.STDOUT, env=ENV), lf, time.time()); pending.remove(j)
                print(f'[debut ] {j.name}', flush=True)
            for n in list(running):
                j, p, lf, t0 = running[n]; rc = p.poll()
                if rc is None: continue
                lf.close(); del running[n]; dt = (time.time() - t0) / 60
                if rc == 0 and j.done(): done.add(n); print(f'[ok    ] {n} ({dt:.1f} min) ; restant {len(pending) + len(running)}', flush=True)
                else: failed.add(n); print(f'[ECHEC ] {n} (code {rc}) -> logs/{n}.log', flush=True)
                log[n] = dict(returncode=rc, minutes=round(dt, 2))
            if (pending or running) and not running and not any(j.deps <= done and not (j.deps & failed) for j in pending): break
            time.sleep(1)
    except KeyboardInterrupt:
        for j, p, lf, t0 in running.values(): p.terminate()
        print('Interrompu : relancer la meme commande pour reprendre.'); sys.exit(130)
    json.dump(dict(workers=a.workers, total_minutes=round((time.time() - t_all) / 60, 1), jobs=log, failed=sorted(failed)), open(os.path.join(HERE, 'run_everything_log.json'), 'w'), indent=1)
    print('\n=========== analyses ===========', flush=True)
    steps = []
    if 'toggle' in a.only: steps += [([PY, 'scripts/analyze_results.py', w], P) for w in ('phase1', 'phase6', 'misspec', 'detect')]
    if 'sergio' in a.only: steps += [([PY, 'analyze_sergio.py'], S)]
    if 'real' in a.only:
        steps += [([PY, 'scripts/analyze_real.py'], R_)] + [([PY, 'scripts/analyze_real.py', R_ + '/results/real', f'real_{sp}', f'real_pancreas_{sp}.json'], R_) for sp in ('module', 'latent')]
    if 'synthetic' in a.only: steps += [([PY, 'scripts/analyze_synth.py'], SY)]
    if 'real' in a.only: steps += [([PY, 'scripts/analyze_real.py', R_ + '/results/real', 'real_dg', 'real_dentategyrus.json'], R_)]
    steps += [([PY, 'make_figures.py'], HERE), ([PY, 'checks/analyze_cost.py'], HERE)] + ([([PY, 'checks/analyze_pysr.py'], HERE)] if not a.no_pysr else [])
    ctl = [([PY, 'checks/verify_results.py'] + (['--strict'] if a.strict else []), HERE), ([PY, 'checks/consistency.py'] + (['--strict'] if a.strict else []), HERE), ([PY, 'checks/make_report.py'], HERE)]
    open(os.path.join(HERE, 'logs', 'analyses.txt'), 'w', encoding='utf-8').write('Sorties des analyses et controles de ce lancement\n')
    bad_ctl = []
    for cmd, cwd in steps:
        print('\n##########', ' '.join(cmd[1:]), flush=True)
        if run_logged(cmd, cwd): bad_ctl.append(os.path.basename(cmd[1]) + ' (analyse)')
    print('\n=========== controles (integrite, coherence, rapport) ===========', flush=True)
    for cmd, cwd in ctl:
        print('\n##########', ' '.join(cmd[1:]), flush=True); rc = run_logged(cmd, cwd)
        if rc: bad_ctl.append(os.path.basename(cmd[1]))
    if bad_ctl: print('\nCONTROLES EN ECHEC :', bad_ctl, '-> NE PAS UTILISER les resultats avant d analyser ces echecs (voir verification_report.json, consistency_report.json).')
    print(f'\nTermine en {(time.time() - t_all) / 60:.1f} min ; echecs : {sorted(failed) or "aucun"}. Resultats : voir GUIDE_TEST.md section 5 (RAPPORT_RESULTATS.md, figures_article/, logs/analyses.txt, project/results, sergio_validation/result_*.json, real_data/results/real, synthetic/results/synth)')
    return 1 if (failed or bad_ctl) else 0


if __name__ == '__main__': sys.exit(main())
