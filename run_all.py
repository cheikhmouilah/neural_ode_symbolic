#!/usr/bin/env python3
"""Lanceur multi-plateformes. Exemples :
  python run_all.py toggle --quick        # test de bon fonctionnement (~2 min : Phase 1, graine 0)
  python run_all.py toggle                # toute la chaine toggle (~1 h sur 1 CPU)
  python run_all.py sergio                # validation SERGIO (~20 min ; clone SERGIO si absent)
  python run_all.py analyze               # tables d'analyse a partir des resultats existants"""
import os, json, sys, subprocess, argparse, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NODE_THREADS='1')   # Windows : UTF-8 ; 1 coeur par processus
print('NB : pour utiliser les 4 coeurs, preferer  python run_everything.py (taches paralleles).')
R_ = os.path.join(HERE, 'real_data'); P = os.path.join(HERE, 'project'); S = os.path.join(HERE, 'sergio_validation')
ap = argparse.ArgumentParser(); ap.add_argument('part', choices=['toggle', 'sergio', 'real', 'analyze']); ap.add_argument('--quick', action='store_true'); ap.add_argument('--force', action='store_true'); a = ap.parse_args()


def run(args, cwd, log=None, outcheck=None):
    if outcheck and os.path.exists(outcheck) and not a.force: print(f"[saute] {outcheck} existe (--force pour recalculer)"); return
    print('>>', ' '.join(args), flush=True)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, 'w') as fh: rc = subprocess.call([PY, '-u'] + args, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    else: rc = subprocess.call([PY, '-u'] + args, cwd=cwd)
    if rc != 0: sys.exit(f"ECHEC (code {rc}) : {' '.join(args)}" + (f" ; voir {log}" if log else ''))


if a.part == 'toggle':
    R = os.path.join(P, 'results', 'phase1'); sc = 'scripts/'
    if a.quick:
        run([sc + 'run_phase1.py', '--seeds', '0', '--tag', 'quick'], P, R + '/run_quick.log', R + '/phase1_quick_seed0.json')
        ref = json.load(open(os.path.join(HERE, 'reference_results', 'toggle', 'phase1_traj_seed0.json'))); got = json.load(open(R + '/phase1_quick_seed0.json')); bad = []
        for lab, g, r_ in (('P1 NRMSE champ', got['P1_node_field_nrmse_test'], ref['P1_node_field_nrmse_test']), ('erreur max des parametres', got['distilled']['max_rel_err'], ref['distilled']['max_rel_err'])):
            rel = abs(g - r_) / max(abs(r_), 1e-12); print(f'  {lab} : obtenu {g:.5f} ; reference {r_:.5f} ; ecart relatif {rel:.2e}'); bad += [lab] if rel > 5e-3 else []
        if not got['distilled']['structure_exact']: bad.append('structure non exacte')
        print('\nVERIFICATION AUTOMATIQUE (tolerance relative 5e-3) :', 'ECHEC -> ' + ', '.join(bad) if bad else 'OK, identique a la reference a la tolerance pres'); sys.exit(1 if bad else 0)
    else:
        run([sc + 'run_phase1.py', '--seeds', '0', '1', '2', '--tag', 'traj'], P, R + '/run_traj.log', R + '/phase1_traj_seed2.json')
        run([sc + 'run_phase1.py', '--seeds', '0', '1', '2', '--velocity', '1.0', '--tag', 'vel'], P, R + '/run_vel.log', R + '/phase1_vel_seed2.json')
        for tag, noise, nt in (('n05', '0.05', '40'), ('n10', '0.10', '40'), ('N15', '0.0', '15')):
            run([sc + 'run_phase6.py', '--seeds', '0', '1', '2', '--tag', tag, '--noise', noise, '--ntrain', nt], P, R + f'/run_{tag}.log', R + f'/phase1_{tag}_seed2.json')
        run([sc + 'run_misspec.py'], P, R + '/run_misspec.log', R + '/misspec_gauss_seed2.json')
        run([sc + 'run_detect_test.py'], P, R + '/run_detect.log', R + '/detect_M4_seed2.json')
        print("\nTermine. Analyses : python run_all.py analyze")
elif a.part == 'sergio':
    if not os.path.isdir(os.path.join(S, 'SERGIO', 'SERGIO')): sys.exit("SERGIO absent : lancer d'abord  python setup_all.py")
    run(['run_sim.py'], S, S + '/log_sim.txt', S + '/sergio_clean.npz')                 # ~5 min, ~2 Go de RAM
    for w in ('clean', 'noisy_mild', 'noisy_severe', 'control'): run(['validate.py', w], S, S + f'/log_validate_{w}.txt', S + f'/result_{w}.json')
    run(['identifiability.py'], S, S + '/log_ident.txt', S + '/result_identifiability.json')
    run(['steady_state_diagnostic.py'], S, S + '/log_ss.txt', S + '/result_steady_state_diag.json')
    run(['eiv_synthetic.py'], S, S + '/log_eiv.txt', S + '/result_eiv.json')
    run(['eiv_test_sergio.py'], S, S + '/log_eiv_sergio.txt', S + '/result_eiv_sergio.json')
elif a.part == 'real':
    O = os.path.join(R_, 'results', 'real'); sc = 'scripts/'    # telecharge le jeu pancreas (52 Mo) depuis GitHub ; ~2,5 min par graine sur 1 CPU
    for s_ in range(5): run([sc + 'run_real.py', '--seeds', str(s_), '--tag', 'real'], R_, O + f'/log_seed{s_}.txt', O + f'/real_none_seed{s_}.json')
    for s_ in range(5): run([sc + 'run_real.py', '--seeds', str(s_), '--tag', 'real', '--control', 'shuffle', '--no-dyn'], R_, O + f'/log_ctrl{s_}.txt', O + f'/real_shuffle_seed{s_}.json')
    run([sc + 'run_real.py', '--sweep', '--tag', 'real'], R_, O + '/log_sweep.txt', O + '/real_sweep_lam1.0.json')
    subprocess.call([PY, sc + 'analyze_real.py'], cwd=R_)
else:
    for w in ('phase1', 'phase6', 'misspec', 'detect'):
        print(f"\n########## {w}"); subprocess.call([PY, 'scripts/analyze_results.py', w], cwd=P)
    print("\n########## sergio"); subprocess.call([PY, 'analyze_sergio.py'], cwd=S)
