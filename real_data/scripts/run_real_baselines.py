"""Script de comparaison des baselines dynamo, CellRank et BoolODE (chaîne C, section VII-D).

Lance les trois baselines sur les données réelles préparées et produit un fichier JSON de résultats
comparable aux sorties de run_real.py.

Usage :
  <python du venv> real_data/scripts/run_real_baselines.py [--config real_pancreas.json] [--out-dir results/real] [--quick]

Prérequis :
  - Données préparées (prepared.npz) : lancer d'abord run_real.py --prepare-only
  - Le h5ad brut (endocrinogenesis_day15.h5ad) améliore dynamo et CellRank (velocity_graph)
    mais n'est pas obligatoire (fallback automatique).

Sorties :
  <out-dir>/baselines_<tag>.json  —  résultats des 3 baselines + métriques de comparaison

Référence théorique :
  - dynamo   : Chen et al. 2022, Nature Methods 19:1105-1117
  - CellRank : Lange et al. 2022, Nature Methods 19:159-170
  - BoolODE  : Pratapa et al. 2020, Nature Methods 17:195-204
"""
import os, sys, json, time, argparse, multiprocessing, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/src')

from real_data import load_prepared
from baseline_dynamo import dynamo_velocity_cosine
from baseline_cellrank import cellrank_cosine
from baseline_boolode import boolode_cosine


if __name__ == '__main__':
    multiprocessing.freeze_support()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='real_pancreas.json')
    ap.add_argument('--out-dir', default=None,
                    help='répertoire de sortie (défaut: results/real_quick si --quick, sinon results/real)')
    ap.add_argument('--tag', default='real',
                    help='préfixe des fichiers (défaut: real)')
    ap.add_argument('--quick', action='store_true',
                    help='mode test rapide : charge prepared.npz depuis results/real_quick')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--data-dir', default=ROOT + '/data',
                    help='répertoire des données brutes h5ad (défaut: real_data/data)')
    ap.add_argument('--no-dynamo', action='store_true')
    ap.add_argument('--no-cellrank', action='store_true')
    ap.add_argument('--no-boolode', action='store_true')
    ap.add_argument('--boolode-from-symbolic', default=None,
                    help='chemin vers un fichier JSON de résultats run_real.py pour extraire '
                         'les modèles symboliques comme réseau BoolODE')
    args = ap.parse_args()

    CFG_PATH = ROOT + '/configs/' + args.config
    cfg = json.load(open(CFG_PATH))

    if args.out_dir is None:
        args.out_dir = ROOT + ('/results/real_quick' if args.quick else '/results/real')
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f'baselines_{args.tag}.json')
    if os.path.exists(out_path) and not args.force:
        print(f'[saute] {out_path} existe (--force pour recalculer)', flush=True)
        sys.exit(0)

    # --- Chargement des données préparées ---
    prep_name = cfg['data'].get('prepared_name', 'prepared.npz')
    prep_path = os.path.join(args.out_dir, prep_name)
    if not os.path.exists(prep_path):
        # Fallback : chercher dans results/real_quick
        alt = os.path.join(ROOT, 'results', 'real_quick', prep_name)
        if os.path.exists(alt):
            prep_path = alt
            print(f'[info] utilise {alt}', flush=True)
        else:
            sys.exit(f'[ERREUR] {prep_path} introuvable. Lancer d\'abord :\n'
                     f'  python scripts/run_real.py --config {args.config} --prepare-only')

    print(f'chargement : {prep_path}', flush=True)
    P = load_prepared(prep_path)
    split = P['split']
    te = np.where(split == 2)[0]
    genes = P['genes']
    n_cells = int(P['n_cells'])
    print(f'cellules={n_cells} gènes={genes} test={len(te)}', flush=True)

    h5ad_path = os.path.join(args.data_dir, cfg['data']['file'])

    # --- Modèles symboliques pour BoolODE (optionnel) ---
    symbolic_models = None
    if args.boolode_from_symbolic and os.path.exists(args.boolode_from_symbolic):
        try:
            sym_data = json.load(open(args.boolode_from_symbolic))
            # Les modèles sont sérialisés (sans objets Python) : reconstituer la structure minimale
            # (kind, j, typ, a, K, n) depuis le champ 'equations'... mais les objets modèles
            # internes ne sont pas dans le JSON. On cherche dans 'models' si présent.
            if 'models' in sym_data:
                symbolic_models = sym_data['models']
                print(f'[info] modèles symboliques chargés depuis {args.boolode_from_symbolic}', flush=True)
            else:
                print('[info] boolode_from_symbolic : champ "models" absent, réseau Granger utilisé', flush=True)
        except Exception as e:
            print(f'[warn] impossible de charger {args.boolode_from_symbolic}: {e}', flush=True)

    results = dict(
        config=args.config,
        tag=args.tag,
        n_cells=n_cells,
        genes=genes,
        n_test=int(len(te)),
        baselines={}
    )

    # --- dynamo ---
    if not args.no_dynamo:
        print('\n--- dynamo (analyse RNA velocity cinétique) ---', flush=True)
        t0 = time.time()
        try:
            r_dyn = dynamo_velocity_cosine(h5ad_path, cfg, P, te)
            r_dyn['seconds'] = round(time.time() - t0, 2)
            print(f"  cos test = {r_dyn.get('cos_test', float('nan')):.4f} "
                  f"| nrmse = {r_dyn.get('nrmse_test', float('nan')):.4f} "
                  f"| source = {r_dyn.get('source', '?')} "
                  f"({r_dyn['seconds']:.1f}s)", flush=True)
            results['baselines']['dynamo'] = r_dyn
        except Exception as e:
            print(f'  [ERREUR] dynamo : {e}', flush=True)
            results['baselines']['dynamo'] = dict(error=str(e))

    # --- CellRank ---
    if not args.no_cellrank:
        print('\n--- CellRank (Markov, probabilités d\'absorption) ---', flush=True)
        t0 = time.time()
        try:
            progenitor = cfg.get('dynamics', {}).get('progenitor_cluster', 'Ngn3 low EP')
            r_cr = cellrank_cosine(h5ad_path, cfg, P, te, progenitor_cluster=progenitor)
            r_cr['seconds'] = round(time.time() - t0, 2)
            print(f"  cos test = {r_cr.get('cos_test', float('nan')):.4f} "
                  f"| nrmse = {r_cr.get('nrmse_test', float('nan')):.4f} "
                  f"| source = {r_cr.get('source', '?')} "
                  f"({r_cr['seconds']:.1f}s)", flush=True)
            if 'absorption' in r_cr:
                ab = r_cr['absorption']
                print(f"  états terminaux : {ab.get('terminal_states', [])} "
                      f"| pureté cluster test = {ab.get('cluster_purity_test', float('nan')):.3f}", flush=True)
            results['baselines']['cellrank'] = r_cr
        except Exception as e:
            print(f'  [ERREUR] CellRank : {e}', flush=True)
            results['baselines']['cellrank'] = dict(error=str(e))

    # --- BoolODE ---
    if not args.no_boolode:
        print('\n--- BoolODE (EDO booléenne continue avec Hill) ---', flush=True)
        t0 = time.time()
        try:
            r_bo = boolode_cosine(P, te, symbolic_models=symbolic_models)
            r_bo['seconds'] = round(time.time() - t0, 2)
            print(f"  cos test = {r_bo.get('cos_test', float('nan')):.4f} "
                  f"| nrmse = {r_bo.get('nrmse_test', float('nan')):.4f} "
                  f"| source réseau = {r_bo.get('network_source', '?')} "
                  f"({r_bo['seconds']:.1f}s)", flush=True)
            if 'cos_test_strict' in r_bo:
                print(f"  BoolODE strict (β=γ=1) : cos = {r_bo['cos_test_strict']:.4f}", flush=True)
            if 'network' in r_bo:
                print(f"  réseau inféré (d° entrant) : "
                      + ', '.join(f"{g}←{len(v)}" for g, v in r_bo['network'].items()), flush=True)
            results['baselines']['boolode'] = r_bo
        except Exception as e:
            print(f'  [ERREUR] BoolODE : {e}', flush=True)
            results['baselines']['boolode'] = dict(error=str(e))

    # --- Tableau comparatif ---
    print('\n=== Tableau comparatif (cosinus test vs vitesse scVelo mesurée) ===', flush=True)
    print(f"{'méthode':<25} {'cos_test':>10} {'nrmse_test':>12}", flush=True)
    print('-' * 50, flush=True)
    for name, r in results['baselines'].items():
        c = r.get('cos_test', float('nan'))
        n = r.get('nrmse_test', float('nan'))
        print(f"  {name:<23} {c:>10.4f} {n:>12.4f}", flush=True)

    # --- Sauvegarde ---
    json.dump(results, open(out_path, 'w'), indent=1,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'\n[ok] résultats → {out_path}', flush=True)
    sys.stdout.flush()
    os._exit(0)
