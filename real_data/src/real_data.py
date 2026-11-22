"""Donnees reelles : chargement, pretraitement scVelo, etat d'etat (espace des genes), vitesse, poids, decoupage par blocs.
Correspond a : partie theorique II-B (donnees, temps), II-C (espace d'etat), III-B (vitesse, poids w_i), V-B."""
import os, json, warnings, urllib.request, numpy as np
warnings.filterwarnings("ignore", message=".*chained_assignment.*")   # bruit de pandas 3 dans anndata : sans effet sur les calculs


def download(url, path, tries=3):
    """Telechargement avec reprises (erreurs reseau transitoires, ex. HTTP 504). Prefere : python setup_all.py (verifie aussi le SHA-256)."""
    import time
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        for k in range(1, tries + 1):
            try:
                print(f'telechargement {url} (tentative {k}/{tries})', flush=True); urllib.request.urlretrieve(url, path + '.part'); os.replace(path + '.part', path); break
            except Exception as e:
                print('  echec :', e, flush=True)
                if k == tries: raise SystemExit(f"Telechargement impossible apres {tries} tentatives ({e}). Lancer  python setup_all.py  plus tard, ou telecharger a la main : {url} -> {path}")
                time.sleep(5 * k)
    return path


def prepare(cfg, data_path):
    import scanpy as sc, scvelo as scv
    from sklearn.cluster import KMeans
    D, S, SP = cfg['data'], cfg['state'], cfg['split']
    a = sc.read_h5ad(data_path)
    # scVelo >= 0.3 n'a plus l'option n_top_genes : filtrage et normalisation par scVelo, genes variables par scanpy (sur copie log1p)
    scv.pp.filter_and_normalize(a, min_shared_counts=D['min_shared_counts'])
    tmp = a.copy(); sc.pp.log1p(tmp); sc.pp.highly_variable_genes(tmp, n_top_genes=D['n_top_genes'], flavor='seurat')
    a = a[:, tmp.var['highly_variable'].values].copy(); del tmp
    # Compute PCA explicitly and ensure C-contiguous layout before moments (avoids
    # numba TypingError "A->C" in UMAP nearest_neighbors on non-contiguous float32 arrays)
    if 'X_pca' not in a.obsm:
        sc.tl.pca(a, n_comps=D['n_pcs'], use_highly_variable=False)
    a.obsm['X_pca'] = np.ascontiguousarray(a.obsm['X_pca'], dtype=np.float32)
    scv.pp.moments(a, n_pcs=D['n_pcs'], n_neighbors=D['n_neighbors'])
    scv.tl.velocity(a, mode=D['velocity_mode'])
    
    scv.tl.velocity_graph(a, n_jobs=1, show_progress_bar=False)   # n_jobs=1 + show_progress_bar=False : evite le multiprocessing de scVelo (fragile sous Windows : spawn)
    scv.tl.velocity_confidence(a)
    Ms = np.asarray(a.layers['Ms'], float); Vel = np.asarray(a.layers['velocity'], float)
    ok = np.asarray(a.var['velocity_genes'].values, bool) & np.isfinite(Vel).all(0)
    space = S.get('space', 'gene')
    if S['genes']:
        genes = [g for g in S['genes'] if g in a.var_names]
    else:
        idx = np.where(ok)[0]; var = Ms[:, idx].var(0)
        genes = list(a.var_names[idx[np.argsort(-var)[:(S['d'] if space == 'gene' else S.get('n_pool', 40))]]])
    cols = [list(a.var_names).index(g) for g in genes]
    scale = np.quantile(Ms[:, cols], 0.99, axis=0)
    Z = Ms[:, cols] / scale; V = Vel[:, cols] / scale
    if space == 'module':      # modules : k-means des genes (profils standardises) ; z_k = moyenne des genes du module, v_k = moyenne de leurs vitesses
        Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); lab = KMeans(n_clusters=S['d'], n_init=5, random_state=0).fit_predict(Zs.T)
        Zm = np.stack([Z[:, lab == k].mean(1) for k in range(S['d'])], 1); Vm = np.stack([V[:, lab == k].mean(1) for k in range(S['d'])], 1)
        genes = [f"mod{k}[" + ','.join(np.array(genes)[lab == k][:3]) + ']' for k in range(S['d'])]; Z, V = Zm, Vm
    elif space == 'latent':    # latent lineaire (ACP) : z = (x - mu) W ; la vitesse se projette par la meme application lineaire : v = vel W
        mu = Z.mean(0); W_ = np.linalg.svd(Z - mu, full_matrices=False)[2][:S['d']].T; Z, V = (Z - mu) @ W_, V @ W_; genes = [f'PC{k+1}' for k in range(S['d'])]
    w = np.nan_to_num(np.asarray(a.obs['velocity_confidence'].values, float), nan=0.0).clip(0, 1); w = w / w.mean()
    clusters = np.asarray(a.obs[D.get('cluster_key', 'clusters')].astype(str).values)
    # decoupage par blocs de voisinage (limite la fuite via le lissage kNN des moments)
    pcs = np.asarray(a.obsm['X_pca'])[:, :10]
    blocks = KMeans(n_clusters=SP['n_blocks'], n_init=3, random_state=SP['seed']).fit_predict(pcs)
    rng = np.random.default_rng(SP['seed']); perm = rng.permutation(SP['n_blocks'])
    nb = [int(round(f * SP['n_blocks'])) for f in SP['fractions']]; nb[-1] = SP['n_blocks'] - nb[0] - nb[1]
    role = {}
    for k, b in enumerate(perm): role[b] = 0 if k < nb[0] else (1 if k < nb[0] + nb[1] else 2)
    split = np.array([role[b] for b in blocks])
    return dict(Z=Z, V=V, w=w, genes=genes, scale=scale, clusters=clusters, split=split,
                n_cells=int(a.n_obs), n_velocity_genes=int(ok.sum()), pcs=pcs)


def save_prepared(P, path):
    tmp = path[:-4] + '.tmp.npz'                                   # ecriture atomique (plusieurs processus liront ce fichier)
    np.savez_compressed(tmp, **{k: np.array(v) for k, v in P.items()}); os.replace(tmp, path)


def load_prepared(path):
    z = np.load(path, allow_pickle=True); P = {k: z[k] for k in z.files}
    P['genes'] = [str(g) for g in P['genes']]; P['n_cells'] = int(P['n_cells']); P['n_velocity_genes'] = int(P['n_velocity_genes'])
    return P
