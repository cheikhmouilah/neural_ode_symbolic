"""
Phase 2 — Pipeline Real Data: Erythroid / Gastrulation (Pijuan-Sala et al. 2019)
Reproduit exactement le meme pipeline que run_real_impl.py mais pour le dataset
gastrulation_erythroid charge via scvelo.datasets (pas de URL directe connue).

Produit : real_data/results/real/real_erythroid_none_seed{0-4}.json
Thresholds pre-enregistres dans configs/real_erythroid.json :
  node cos test >= 0.50, sym~node fidelity >= 0.90
"""
import os, sys, json, time, hashlib, multiprocessing, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # real_data/
PROJ_ROOT = os.path.dirname(ROOT)                                      # c:\neural_ode_symbolic
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, PROJ_ROOT)

from real_data import prepare, save_prepared, load_prepared
from real_node import train_vel_node, field_np
from real_identify import (build_candidates, select_structure, predict_all,
                            structure_set, jaccard, describe)
from sklearn.neighbors import NearestNeighbors

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--force', action='store_true')
ap.add_argument('--no-dyn', action='store_true')
ap.add_argument('--quick', action='store_true',
                help='Test rapide : peu d\'epoques, bibliotheque reduite')
args = ap.parse_args()

CFG_PATH = os.path.join(ROOT, 'configs', 'real_erythroid.json')
cfg = json.load(open(CFG_PATH))
CFG_SHA = hashlib.sha256(open(CFG_PATH, 'rb').read()).hexdigest()[:16]
if args.quick:
    cfg['node'].update(epochs=40, patience=20)
    cfg['library'].update(k_max=2, n_starts=2,
                          K_grid=cfg['library']['K_grid'][::2],
                          n_grid=[1.0, 2.0, 3.0])
    cfg['distillation']['M'] = 1200
    CFG_SHA = 'QUICK-' + CFG_SHA

OUT = os.path.join(ROOT, 'results', 'real_quick' if args.quick else 'real')
os.makedirs(OUT, exist_ok=True)

# ---- 1. Prepare dataset (download + preprocess, cached as .npz) ----
prep_name = cfg['data'].get('prepared_name', 'prepared_erythroid.npz')
prep_path = os.path.join(OUT, prep_name)

if os.path.exists(prep_path) and not args.force:
    print(f'[LOAD cached] {prep_path}', flush=True)
    P = load_prepared(prep_path)
else:
    print('Loading gastrulation_erythroid via scVelo...', flush=True)
    import scvelo as scv, scanpy as sc

    adata = scv.datasets.gastrulation_erythroid()
    print(f'  Raw: {adata.n_obs} cells, {adata.n_vars} genes', flush=True)
    print(f'  Layers: {list(adata.layers.keys())}', flush=True)
    print(f'  Obs keys: {list(adata.obs.columns[:8])}', flush=True)

    # Save to temp h5ad so prepare() can read it (needs sc.read_h5ad)
    h5ad_path = os.path.join(OUT, 'erythroid_raw.h5ad')
    if not os.path.exists(h5ad_path) or args.force:
        print(f'  Writing temp h5ad -> {h5ad_path}', flush=True)
        adata.write_h5ad(h5ad_path)

    sha256 = hashlib.sha256(open(h5ad_path, 'rb').read()).hexdigest()
    print(f'  SHA-256: {sha256}', flush=True)

    # Update config SHA in preregistration (informational only)
    cfg['data']['sha256_computed'] = sha256

    # Build a minimal cfg compatible with prepare() signature
    # prepare() reads cfg['data']['min_shared_counts'], n_top_genes, n_pcs, n_neighbors,
    # velocity_mode, cluster_key; cfg['state'], cfg['split']
    t0 = time.time()
    P = prepare(cfg, h5ad_path)
    save_prepared(P, prep_path)
    print(f'  Preprocessing done in {time.time()-t0:.0f}s', flush=True)
    print(f'  Genes selected: {P["genes"]}', flush=True)
    print(f'  Cells: {P["n_cells"]}, velocity genes: {P["n_velocity_genes"]}', flush=True)

Z, V0, W = P['Z'], P['V'], P['w']
genes = P['genes']
clusters = P['clusters']
split = P['split']
tr, va, te = [np.where(split == k)[0] for k in (0, 1, 2)]
print(f'Train/val/test: {len(tr)}/{len(va)}/{len(te)}  genes={genes}', flush=True)
SPACE = cfg['state'].get('space', 'gene')


# ---- Helper functions (identical to run_real_impl.py) ----
def cosine(Fp, Ft, eps=1e-9):
    a, b = np.linalg.norm(Fp, axis=1), np.linalg.norm(Ft, axis=1)
    m = (a > eps) & (b > eps)
    if not np.any(m):
        return 0.0
    val = np.sum(Fp[m] * Ft[m], 1) / (a[m] * b[m])
    val = val[np.isfinite(val)]
    return float(np.mean(val)) if len(val) > 0 else 0.0


def nrmse(Fp, Ft):
    rng_ft = Ft.max() - Ft.min()
    return float(np.sqrt(np.mean((Fp - Ft) ** 2)) / (rng_ft if rng_ft > 1e-9 else 1.0))


def sample_Df(f, Ztr, rng, dist):
    M, k = dist['M'], dist['knn']
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Ztr)
    dk = nn.kneighbors(Ztr)[0][:, -1]
    thr = np.quantile(dk, dist['reject_quantile'])
    half = M // 2
    obs = Ztr[rng.choice(len(Ztr), half)]
    i = rng.choice(len(Ztr), M)
    cand = Ztr[i] + rng.normal(size=(M, Ztr.shape[1])) * (
        dist['jitter_frac'] * dk[i] / np.sqrt(Ztr.shape[1]))[:, None]
    cand = np.maximum(cand, 0)
    keep = nn.kneighbors(cand, n_neighbors=1)[0][:, 0] <= thr
    Zf = np.vstack([obs, cand[keep][:M - half]])
    return Zf, dict(n_obs=int(half), n_jitter=int(len(Zf) - half),
                    reject_thr=float(thr), rejected=int((~keep).sum()))


def make_cand(Zc):
    return build_candidates(Zc, cfg['library']['K_grid'], cfg['library']['n_grid'])


# ---- Main per-seed run ----
def run_one(seed, lam):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    lib = cfg['library']
    V = V0.copy()

    f, info = train_vel_node(Z, V, W, tr, va,
                             dict(cfg['node']), seed, lam_reg=lam,
                             log=lambda s: print(s, flush=True))
    F = field_np(f, Z)
    r = dict(seed=seed, lam_reg=lam, control='none', config_sha=CFG_SHA,
             genes=genes, space=SPACE, node=info, dataset='erythroid')

    r['node_cos_test'] = cosine(F[te], V[te])
    r['node_cos_val'] = cosine(F[va], V[va])
    r['node_cos_train'] = cosine(F[tr], V[tr])
    r['node_nrmse_test'] = nrmse(F[te], V[te])

    # Linear baseline
    A = np.column_stack([Z[tr], np.ones(len(tr))])
    Bl = np.linalg.lstsq(A, V[tr], rcond=None)[0]
    r['linear_baseline_cos_test'] = cosine(
        np.column_stack([Z[te], np.ones(len(te))]) @ Bl, V[te])

    # Distillation
    Zf, sinfo = sample_Df(f, Z[tr], rng, cfg['distillation'])
    Ff = field_np(f, Zf)
    r['Df'] = sinfo
    p = rng.permutation(len(Zf))
    nv = int(cfg['distillation']['val_fraction'] * len(Zf))
    sv, st = p[:nv], p[nv:]
    cand = make_cand(Zf[st])
    models, paretos = [], []
    for i in range(Z.shape[1]):
        ch, pa = select_structure(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i],
                                  i, lib, rng, cand=cand)
        models.append(ch['model'])
        paretos.append(pa)

    Fs = predict_all(models, Z[te])
    r['sym_cos_vs_node_test'] = cosine(Fs, F[te])
    r['sym_nrmse_vs_node_test'] = nrmse(Fs, F[te])
    r['sym_cos_vs_measured_test'] = cosine(Fs, V[te])
    r['R7_ratio'] = r['sym_cos_vs_measured_test'] / max(r['node_cos_test'], 1e-9)
    r['equations'] = describe(models, genes)
    r['structure'] = [sorted(map(str, s)) for s in structure_set(models)]

    # Direct baseline
    q = rng.permutation(len(tr))
    nv2 = int(0.2 * len(tr))
    dv, dt = tr[q[:nv2]], tr[q[nv2:]]
    cand2 = make_cand(Z[dt])
    dm = []
    for i in range(Z.shape[1]):
        ch, _ = select_structure(Z[dt], V[dt, i], Z[dv], V[dv, i],
                                 i, lib, rng, cand=cand2)
        dm.append(ch['model'])
    r['direct_cos_vs_measured_test'] = cosine(predict_all(dm, Z[te]), V[te])
    r['direct_equations'] = describe(dm, genes)
    r['direct_structure'] = [sorted(map(str, s)) for s in structure_set(dm)]
    r['distilled_vs_direct_jaccard'] = jaccard(structure_set(models), structure_set(dm))

    r['seconds_total'] = time.time() - t0
    print(
        f"seed {seed} lam={lam}: node cos test={r['node_cos_test']:.3f} "
        f"(lin {r['linear_baseline_cos_test']:.3f}) | "
        f"sym~node={r['sym_cos_vs_node_test']:.3f} | "
        f"sym~V={r['sym_cos_vs_measured_test']:.3f} | "
        f"direct~V={r['direct_cos_vs_measured_test']:.3f} | "
        f"{r['seconds_total']:.0f}s",
        flush=True
    )
    return r


def dump(r, name):
    json.dump(r, open(f'{OUT}/{name}.json', 'w'), indent=1,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    lam = cfg['node']['lam_reg']
    seeds = cfg['stability']['seeds']
    for s in seeds:
        name = f'real_erythroid_none_seed{s}'
        out_file = os.path.join(OUT, f'{name}.json')
        if os.path.exists(out_file) and not args.force:
            print(f'[SKIP] {out_file}')
            continue
        r = run_one(s, lam)
        dump(r, name)
        print(f'  Saved: {out_file}', flush=True)
    print('FIN erythroid pipeline', flush=True)
    sys.stdout.flush()
    os._exit(0)
