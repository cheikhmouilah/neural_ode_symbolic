"""Distillation LIBRE par PySR du champ neuronal sur DONNEES REELLES (chaine C) : meme pretraitement, meme etat d'etat, meme Neural ODE (appariement de vitesse) que run_real.py ; les cibles de PySR sont
les valeurs du champ du Neural ODE sur D_f ; evaluation sur cellules de test (cosinus avec le champ neuronal et avec la vitesse mesuree). Necessite Julia et PySR : voir GUIDE_PYSR.md.
   python scripts/run_real_pysr.py [--config real_pancreas.json] [--seeds 0] [--equations 0 1 2] [--iterations 40] [--maxsize 15] [--quick]
Une recherche PySR par equation (8 genes : 8 recherches) : lancer d'abord avec --equations 0 pour estimer la duree."""
import os, sys, json, time, argparse, re, hashlib, multiprocessing, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT + '/src')
from real_data import download, prepare, save_prepared, load_prepared
from real_node import train_vel_node, field_np
from sklearn.neighbors import NearestNeighbors

if __name__ == '__main__':
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='real_pancreas.json'); ap.add_argument('--seeds', type=int, nargs='+', default=[0]); ap.add_argument('--equations', type=int, nargs='+', default=None)
    PC = json.load(open(ROOT + '/configs/real_pysr.json')); ap.add_argument('--iterations', type=int, default=PC['iterations']); ap.add_argument('--maxsize', type=int, default=PC['maxsize']); ap.add_argument('--quick', action='store_true'); ap.add_argument('--tag', default='real_pysr'); ap.add_argument('--data-dir', default=ROOT + '/data'); args = ap.parse_args()
    try: from pysr import PySRRegressor
    except Exception as e: sys.exit(f"PySR indisponible ({e}). Voir GUIDE_PYSR.md : python setup_all.py --with-pysr")
    CP = ROOT + '/configs/' + args.config; cfg = json.load(open(CP)); sha = hashlib.sha256(open(CP, 'rb').read()).hexdigest()[:16]
    if args.quick: cfg['node'].update(epochs=40, patience=20); cfg['distillation']['M'] = 1200; args.iterations = min(args.iterations, 2); sha = 'QUICK-' + sha
    OUT = ROOT + ('/results/real_quick' if args.quick else '/results/real'); os.makedirs(OUT, exist_ok=True)
    prep = OUT + '/' + cfg['data'].get('prepared_name', 'prepared.npz')
    if os.path.exists(prep): P = load_prepared(prep)
    else:
        path = download(cfg['data']['url'], os.path.join(args.data_dir, cfg['data']['file']))
        if cfg['data'].get('sha256') and hashlib.sha256(open(path, 'rb').read()).hexdigest() != cfg['data']['sha256']: sys.exit('[GARDE-FOU] empreinte SHA-256 du jeu de donnees differente de celle attendue. Arret.')
        P = prepare(cfg, path); save_prepared(P, prep)
    Z, V, W, genes, split = P['Z'], P['V'], P['w'], P['genes'], P['split']; tr, va, te = [np.where(split == k)[0] for k in (0, 1, 2)]; d = Z.shape[1]; eqs = args.equations if args.equations else list(range(d))
    cosine = lambda A, B: float(np.mean(np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)))
    for seed in args.seeds:
        t0 = time.time(); rng = np.random.default_rng(seed); print(f'=== PySR reel ({args.config}), graine {seed}, equations {eqs} ===', flush=True)
        f, info = train_vel_node(Z, V, W, tr, va, cfg['node'], seed, log=lambda s: None); F = field_np(f, Z)
        nn = NearestNeighbors(n_neighbors=cfg['distillation']['knn'] + 1).fit(Z[tr]); dk = nn.kneighbors(Z[tr])[0][:, -1]; M = cfg['distillation']['M']; i_ = rng.choice(len(tr), M)
        cand = np.maximum(Z[tr][i_] + rng.normal(size=(M, d)) * (cfg['distillation']['jitter_frac'] * dk[i_] / np.sqrt(d))[:, None], 0); keep = nn.kneighbors(cand, n_neighbors=1)[0][:, 0] <= np.quantile(dk, cfg['distillation']['reject_quantile'])
        Zf = np.vstack([Z[tr][rng.choice(len(tr), M // 2)], cand[keep][:M - M // 2]]); Ff = field_np(f, Zf); rows = []; Fs = F[te].copy(); t1 = time.time()
        for i in eqs:
            m = PySRRegressor(niterations=args.iterations, binary_operators=['+', '-', '*', '/'], unary_operators=[], maxsize=args.maxsize, random_state=seed * 100 + i, deterministic=True, parallelism='serial',
                              progress=False, verbosity=0, model_selection='best', temp_equation_file=False, output_directory=os.path.join(ROOT, 'results', 'pysr_runs'))
            m.fit(Zf[:min(len(Zf), 3000)], Ff[:min(len(Zf), 3000), i]); b = m.get_best(); expr = str(b['equation']); Fs[:, i] = np.asarray(b['lambda_format'](Z[te])).ravel(); used = sorted({int(v) for v in re.findall(r'x(\d+)', expr)})
            rows.append(dict(eq=i, gene=genes[i], expression=expr, complexity=int(b['complexity']), loss=float(b['loss']), used_variables=used, used_genes=[genes[k] for k in used], pareto=[dict(complexity=int(r['complexity']), loss=float(r['loss'])) for _, r in m.equations_.iterrows()]))
            print(f"   eq{i} {genes[i]} (complexite {rows[-1]['complexity']}) : {expr}", flush=True)
        sub = list(eqs); r = dict(seed=seed, tag=args.tag, config_sha=sha, config=args.config, genes=genes, space=cfg['state'].get('space', 'gene'), equations_fitted=sub, iterations=args.iterations, maxsize=args.maxsize, equations=rows,
                                  node_cos_test=cosine(F[te], V[te]), pysr_vs_node_cos_test=cosine(Fs[:, sub], F[te][:, sub]), pysr_vs_measured_cos_test=cosine(Fs[:, sub], V[te][:, sub]), node_vs_measured_cos_same_genes=cosine(F[te][:, sub], V[te][:, sub]), seconds=time.time() - t0, seconds_pysr=time.time() - t1)
        json.dump(r, open(f'{OUT}/{args.tag}_seed{seed}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
        print(f"graine {seed} : cosinus PySR~NODE {r['pysr_vs_node_cos_test']:.3f} ; PySR~vitesse {r['pysr_vs_measured_cos_test']:.3f} ; NODE~vitesse (memes genes) {r['node_vs_measured_cos_same_genes']:.3f} ; {r['seconds']:.0f} s")
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
