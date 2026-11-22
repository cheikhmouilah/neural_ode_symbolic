"""Chaine C : donnees reelles (endocrinogenese pancreatique). Protocole = partie theorique, sections V-B a V-E, VII-C, VII-D.
Seuils et choix fixes dans ../configs/real_pancreas.json AVANT execution (empreinte SHA-256 enregistree dans chaque resultat)."""
import os, sys, json, time, argparse, hashlib, multiprocessing, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/src')
from real_data import download, prepare, save_prepared, load_prepared
from real_node import train_vel_node, field_np
from real_identify import build_candidates, select_structure, select_structure_poly, predict_all, structure_set, jaccard, describe
from real_dyn import equilibria, cluster_correspondence, endpoints
from sklearn.neighbors import NearestNeighbors


def cosine(Fp, Ft, eps=1e-9):
    a, b = np.linalg.norm(Fp, axis=1), np.linalg.norm(Ft, axis=1); m = (a > eps) & (b > eps)
    if not np.any(m):
        return 0.0
    val = np.sum(Fp[m] * Ft[m], 1) / (a[m] * b[m])
    val = val[np.isfinite(val)]
    return float(np.mean(val)) if len(val) > 0 else 0.0


def nrmse(Fp, Ft): return float(np.sqrt(np.mean((Fp - Ft) ** 2)) / (Ft.max() - Ft.min()))


def sample_Df(f, Ztr, rng, dist):
    M, k = dist['M'], dist['knn']; nn = NearestNeighbors(n_neighbors=k + 1).fit(Ztr)
    dk = nn.kneighbors(Ztr)[0][:, -1]; thr = np.quantile(dk, dist['reject_quantile'])
    half = M // 2; obs = Ztr[rng.choice(len(Ztr), half)]
    i = rng.choice(len(Ztr), M); cand = Ztr[i] + rng.normal(size=(M, Ztr.shape[1])) * (dist['jitter_frac'] * dk[i] / np.sqrt(Ztr.shape[1]))[:, None]
    cand = np.maximum(cand, 0) if SPACE != 'latent' else cand; keep = nn.kneighbors(cand, n_neighbors=1)[0][:, 0] <= thr
    Zf = np.vstack([obs, cand[keep][:M - half]]); return Zf, dict(n_obs=int(half), n_jitter=int(len(Zf) - half), reject_thr=float(thr), rejected=int((~keep).sum()))


def stlsq_baseline(Zf, Ff, Zte, Fte_node, lib):
    import pysindy as ps
    d = Zf.shape[1]; names = ['1'] + [f'z{i}' for i in range(d)]; cols = lambda Z_: [np.ones(len(Z_))] + [Z_[:, i] for i in range(d)]
    quad = [(i, j) for i in range(d) for j in range(i, d)]
    def lib_cols(Z_):
        c = cols(Z_) + [Z_[:, i] * Z_[:, j] for i, j in quad]
        _, Phi = build_candidates(Z_, lib['K_grid'], lib['n_grid']); return np.column_stack(c + [Phi[:, 1:]])
    Th, Tt = lib_cols(Zf), lib_cols(Zte); nrm = np.linalg.norm(Th, axis=0) + 1e-12; out = []
    for thr in (0.01, 0.05, 0.1, 0.2, 0.5):
        coefs = []
        for i in range(d):
            o = ps.STLSQ(threshold=thr, alpha=1e-6); o.fit(Th / nrm, Ff[:, i:i + 1]); coefs.append(np.asarray(o.coef_).reshape(-1) / nrm)
        C = np.stack(coefs, 1); out.append(dict(threshold=thr, mean_terms=float(np.mean((C != 0).sum(0))), cos_vs_node_test=cosine(Tt @ C, Fte_node)))
    return out


def run_one(seed, lam, do_dyn, control):
    t0 = time.time(); rng = np.random.default_rng(seed); node_cfg = dict(cfg['node']); lib = cfg['library']
    V = V0.copy()
    if control == 'shuffle': V = V0[np.random.default_rng(1000 + seed).permutation(len(V0))]
    f, info = train_vel_node(Z, V, W, tr, va, node_cfg, seed, lam_reg=lam, log=lambda s: print(s, flush=True))
    F = field_np(f, Z); r = dict(seed=seed, lam_reg=lam, control=control, config_sha=CFG_SHA, genes=genes, space=SPACE, node=info)
    r['node_cos_test'], r['node_cos_val'], r['node_cos_train'] = cosine(F[te], V[te]), cosine(F[va], V[va]), cosine(F[tr], V[tr])
    r['node_nrmse_test'] = nrmse(F[te], V[te])
    A = np.column_stack([Z[tr], np.ones(len(tr))]); Bl = np.linalg.lstsq(A, V[tr], rcond=None)[0]
    r['linear_baseline_cos_test'] = cosine(np.column_stack([Z[te], np.ones(len(te))]) @ Bl, V[te])
    # --- distillation : D_f echantillonne dans la region supportee, cibles = champ du Neural ODE ---
    Zf, sinfo = sample_Df(f, Z[tr], rng, cfg['distillation']); Ff = field_np(f, Zf); r['Df'] = sinfo
    p = rng.permutation(len(Zf)); nv = int(cfg['distillation']['val_fraction'] * len(Zf)); sv, st = p[:nv], p[nv:]
    cand = make_cand(Zf[st]); models, paretos = [], []
    for i in range(Z.shape[1]):
        ch, pa = SEL(Zf[st], Ff[st, i], Zf[sv], Ff[sv, i], i, lib, rng, cand=cand); models.append(ch['model']); paretos.append(pa)
    Fs = predict_all(models, Z[te]); r['sym_cos_vs_node_test'] = cosine(Fs, F[te]); r['sym_nrmse_vs_node_test'] = nrmse(Fs, F[te])
    r['sym_cos_vs_measured_test'] = cosine(Fs, V[te]); r['R7_ratio'] = r['sym_cos_vs_measured_test'] / max(r['node_cos_test'], 1e-9)
    r['equations'] = describe(models, genes); r['models'] = models
    r['structure'] = [sorted(map(str, s)) for s in structure_set(models)]
    r['omega'] = [p_['omega'] for p_ in [min([q for q in pa if q['model'] is models[k]] or pa, key=lambda q: q['omega']) for k, pa in enumerate(paretos)]]
    r['pareto'] = [[dict(k=q['k'], omega=q['omega'], mse_val=q['mse_val']) for q in pa] for pa in paretos]
    r['refine_spread'] = [[(q['refine'] or {}).get('spread') for q in pa if q['model'] is models[k]] for k, pa in enumerate(paretos)]
    # --- identification directe (baseline) : cibles = vitesses mesurees ---
    q = rng.permutation(len(tr)); nv2 = int(0.2 * len(tr)); dv, dt = tr[q[:nv2]], tr[q[nv2:]]; cand2 = make_cand(Z[dt]); dm = []
    for i in range(Z.shape[1]):
        ch, _ = SEL(Z[dt], V[dt, i], Z[dv], V[dv, i], i, lib, rng, cand=cand2); dm.append(ch['model'])
    r['direct_cos_vs_measured_test'] = cosine(predict_all(dm, Z[te]), V[te]); r['direct_equations'] = describe(dm, genes)
    r['direct_structure'] = [sorted(map(str, s)) for s in structure_set(dm)]; r['distilled_vs_direct_jaccard'] = jaccard(structure_set(models), structure_set(dm))
    if control == 'none' and SPACE != 'latent': r['stlsq_baseline'] = stlsq_baseline(Zf[st], Ff[st], Z[te], F[te], lib)
    # --- dynamique : equilibres, stabilite, correspondance aux clusters, trajectoires depuis les progeniteurs ---
    if do_dyn:
        D = cfg['dynamics']; lo, hi = Z[tr].min(0), Z[tr].max(0); span = hi - lo; lo, hi = lo - D['box_margin'] * span, hi + D['box_margin'] * span
        starts = Z[rng.choice(len(Z), D['n_equilibria_starts'], replace=False)]; tol = 1e-3 * float(np.median(np.linalg.norm(F[tr], axis=1)))
        fn = lambda x: field_np(f, np.asarray(x)[None])[0]; fs = lambda x: predict_all(models, np.asarray(x)[None])[0]
        for name, fun in (('node', fn), ('symbolic', fs)):
            eq = equilibria(fun, starts, lo, hi, tol); rows, cen = cluster_correspondence(eq, Z, clusters)
            r[f'eq_{name}'] = dict(n=len(eq), n_stable=sum(e['stable'] for e in eq), points=[dict(x=e['x'].tolist(), stable=e['stable'], max_eig_real=e['max_eig_real']) for e in eq], clusters=rows)
        rate = float(np.median(np.linalg.norm(F[tr], axis=1) / (np.linalg.norm(Z[tr], axis=1) + 1e-9))); T = D['T_multiplier'] / rate
        prog = np.where(clusters == D['progenitor_cluster'])[0]; z0 = Z[rng.choice(prog, min(60, len(prog)), replace=False)]
        for name, fun in (('node', fn), ('symbolic', fs)):
            ends, spd = endpoints(fun, z0, T, cen); vals, cnt = np.unique(ends, return_counts=True)
            spd_clean = [float(s) for s in spd if np.isfinite(s)]
            med_spd = float(np.median(spd_clean)) if len(spd_clean) > 0 else 0.0
            r[f'endpoints_{name}'] = dict(T=T, distribution=dict(zip(vals.tolist(), cnt.tolist())), median_final_speed=med_spd)
    r['seconds_total'] = time.time() - t0
    print(f"seed {seed} lam={lam} ctrl={control}: node cos test={r['node_cos_test']:.3f} (lin {r['linear_baseline_cos_test']:.3f}) | sym~node={r['sym_cos_vs_node_test']:.3f} | sym~V={r['sym_cos_vs_measured_test']:.3f} | direct~V={r['direct_cos_vs_measured_test']:.3f} | omega={r['omega']} | {r['seconds_total']:.0f}s", flush=True)
    return r


def dump(r, name):
    json.dump(r, open(f'{OUT}/{name}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=None); ap.add_argument('--tag', default='real')
    ap.add_argument('--control', choices=['none', 'shuffle'], default='none'); ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--no-dyn', action='store_true'); ap.add_argument('--prepare-only', action='store_true'); ap.add_argument('--quick', action='store_true', help='test de fonctionnement : peu d epoques, bibliotheque reduite ; resultats dans results/real_quick (jamais melanges aux vrais)'); ap.add_argument('--config', default='real_pancreas.json'); ap.add_argument('--lam', type=float, default=None); ap.add_argument('--data-dir', default=ROOT + '/data'); ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    CFG_PATH = ROOT + '/configs/' + args.config; cfg = json.load(open(CFG_PATH))
    CFG_SHA = hashlib.sha256(open(CFG_PATH, 'rb').read()).hexdigest()[:16]
    if args.quick:
        cfg['node'].update(epochs=40, patience=20); cfg['library'].update(k_max=2, n_starts=2, K_grid=cfg['library']['K_grid'][::2], n_grid=[1.0, 2.0, 3.0]); cfg['distillation']['M'] = 1200; cfg['dynamics'].update(n_equilibria_starts=15); CFG_SHA = 'QUICK-' + CFG_SHA
    OUT = ROOT + ('/results/real_quick' if args.quick else '/results/real'); os.makedirs(OUT, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else cfg['stability']['seeds']

    SPACE = cfg['state'].get('space', 'gene'); SEL = select_structure_poly if SPACE == 'latent' else select_structure
    make_cand = (lambda Zc: None) if SPACE == 'latent' else (lambda Zc: build_candidates(Zc, cfg['library']['K_grid'], cfg['library']['n_grid']))
    prep_path = OUT + '/' + cfg['data'].get('prepared_name', 'prepared.npz' if SPACE == 'gene' else f'prepared_{SPACE}.npz')
    if os.path.exists(prep_path) and not args.force: P = load_prepared(prep_path)
    else:
        path = download(cfg['data']['url'], os.path.join(args.data_dir, cfg['data']['file'])); t0 = time.time()
        if cfg['data'].get('sha256') and hashlib.sha256(open(path, 'rb').read()).hexdigest() != cfg['data']['sha256']: sys.exit(f"[GARDE-FOU] empreinte SHA-256 de {path} differente de celle attendue : jeu de donnees modifie ou corrompu. Arret.")
        P = prepare(cfg, path); save_prepared(P, prep_path); print(f'pretraitement {time.time()-t0:.0f}s', flush=True)
    if args.prepare_only: print('pretraitement termine :', prep_path, flush=True); sys.stdout.flush(); os._exit(0)
    Z, V0, W, genes, clusters, split = P['Z'], P['V'], P['w'], P['genes'], P['clusters'], P['split']
    tr, va, te = [np.where(split == k)[0] for k in (0, 1, 2)]
    print(f'cellules={len(Z)} genes={genes} train/val/test={len(tr)}/{len(va)}/{len(te)} config={CFG_SHA}', flush=True)

    if args.sweep:
        for lam in ([args.lam] if args.lam is not None else cfg['stability']['lam_sweep']):
            n = f"{args.tag}_sweep_lam{lam}"
            if os.path.exists(f'{OUT}/{n}.json') and not args.force: print('[saute]', n); continue
            dump(run_one(0, lam, not args.no_dyn, 'none'), n)
    else:
        for s in seeds:
            n = f"{args.tag}_{args.control}_seed{s}"
            if os.path.exists(f'{OUT}/{n}.json') and not args.force: print('[saute]', n); continue
            dump(run_one(s, cfg['node']['lam_reg'], (not args.no_dyn) and args.control == 'none', args.control), n)
    print('FIN', flush=True); sys.stdout.flush(); os._exit(0)
