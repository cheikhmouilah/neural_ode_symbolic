"""Baseline CellRank-like : probabilités d'absorption par état terminal via la matrice de Markov de vitesse.

Implémente la méthode centrale de CellRank (Lange et al. 2022, Nature Methods 19:159–170) :
1. Construction de la matrice de transition T_{ij} à partir du graphe de cosinus de vitesse scVelo
   (disponible dans velocity_graph de l'adata, ou recalculé depuis Z et V).
2. Identification des états terminaux (cellules dont la probabilité de sortie est minimale).
3. Résolution du système linéaire pour calculer les probabilités d'absorption par état terminal.
4. Évaluation : cosinus entre la vitesse induite par T et la vitesse scVelo mesurée.

Sans nécessiter l'installation du paquet `cellrank` — calcul direct sur (Z, V).

Usage :
    from baseline_cellrank import cellrank_cosine
    result = cellrank_cosine(h5ad_path, cfg, P, te)
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Construction de la matrice de transition (style CellRank / scVelo)
# ---------------------------------------------------------------------------

def _softmax_row(M, T=1.0):
    """Softmax ligne par ligne (température T) ; M sparse ou dense."""
    if hasattr(M, 'toarray'):
        M = M.toarray()
    M = np.array(M, dtype=float)
    # soustraction du max pour la stabilité numérique
    M -= M.max(axis=1, keepdims=True)
    E = np.exp(M / T)
    row_sum = E.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-15] = 1.0
    return E / row_sum


def _build_transition_matrix(Z, V, n_neighbors=30, temperature=1.0):
    """Matrice de transition T (n_cells × n_cells) calculée comme dans CellRank :
    T_{ij} ∝ exp(cosine(v_i, z_j - z_i) / T) pour j dans les k-NN de i.

    Returns
    -------
    T : ndarray (n_cells, n_cells), lignes normalisées (distributions de probabilité)
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    k = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(Z)
    dist, idx = nn.kneighbors(Z)   # (n, k+1) — inclut la cellule elle-même en idx[:,0]

    # Cosinus entre la vitesse de la cellule i et le déplacement vers chaque voisin j
    rows, cols, vals = [], [], []
    for i in range(n):
        vi = V[i]              # vitesse de la cellule i
        norm_vi = np.linalg.norm(vi)
        for t, j in enumerate(idx[i]):
            if j == i:
                continue
            dz = Z[j] - Z[i]  # direction vers j
            norm_dz = np.linalg.norm(dz)
            if norm_vi < 1e-12 or norm_dz < 1e-12:
                cos_ij = 0.0
            else:
                cos_ij = float(np.dot(vi, dz) / (norm_vi * norm_dz))
            rows.append(i); cols.append(j); vals.append(cos_ij)

    # Matrice sparse (cosinus)
    C = csr_matrix((vals, (rows, cols)), shape=(n, n))
    C_dense = C.toarray()
    # Exclure les 0 (non-voisins) → mettre à -inf avant softmax
    C_dense[C_dense == 0] = -1e9   # non-voisins non connectés
    # Softmax par ligne → T
    T = _softmax_row(C_dense, T=temperature)
    # Remettre à 0 les non-voisins
    mask = (C.toarray() == 0)
    # Rétablir les autoconnexions à 0
    np.fill_diagonal(mask, False)
    T[mask] = 0.0
    # Re-normaliser
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-15] = 1.0
    T /= row_sum
    return T


def _identify_terminal_states(T, clusters, progenitor_cluster='Ngn3 low EP', top_k=3):
    """Identifie les états terminaux comme les clusters avec la plus faible auto-transition
    (ou la plus faible entropie de sortie), à l'exclusion du cluster progéniteur.

    Returns
    -------
    terminal_indices : dict cluster_name → list[int] indices de cellules terminales
    """
    # Auto-transition T_{ii} pour chaque cellule → cellules terminales = haute auto-transition
    diag_T = np.diag(T) if T.ndim == 2 else np.array([T[i, i] for i in range(len(T))])
    unique_clusters = sorted(set(clusters))
    # Score de terminalité par cluster = moyenne de T_{ii}
    terminal_score = {}
    for c in unique_clusters:
        idx_c = np.where(clusters == c)[0]
        if len(idx_c) == 0:
            continue
        terminal_score[c] = float(diag_T[idx_c].mean())
    # Exclure le cluster progéniteur
    candidates = {c: s for c, s in terminal_score.items() if c != progenitor_cluster}
    # top_k clusters avec le score le plus élevé (interprétation : plus stable = terminal)
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:top_k]
    terminal_clusters = {c for c, _ in sorted_candidates}
    terminal_indices = {c: np.where(clusters == c)[0].tolist() for c in terminal_clusters}
    return terminal_indices


def _absorption_probabilities(T, terminal_groups):
    """Calcule les probabilités d'absorption par groupe terminal.
    Utilise le système linéaire classique de Markov pour les états absorbants.

    Returns
    -------
    abs_probs : ndarray (n_cells, n_terminals) — probabilité d'atteindre chaque terminal
    terminal_names : list[str]
    """
    n = T.shape[0]
    terminal_names = sorted(str(k) for k in terminal_groups.keys())
    # Reconstruire le dict avec des str (clés np.str_ → str)
    terminal_groups = {str(k): v for k, v in terminal_groups.items()}
    terminal_idx_flat = []
    for name in terminal_names:
        terminal_idx_flat.extend(terminal_groups[name])
    terminal_set = set(terminal_idx_flat)
    transient = [i for i in range(n) if i not in terminal_set]
    n_tr = len(transient)
    if n_tr == 0:
        # Toutes les cellules sont terminales
        return np.eye(n, len(terminal_names)), terminal_names
    # Réindexer
    tr_map = {old: new for new, old in enumerate(transient)}
    # Matrice de transition entre transients : Q = T[transient][:,transient]
    Q = T[np.ix_(transient, transient)]
    # Matrice d'absorption : R[i, k] = sum_{j in term_k} T[transient[i], j]
    R = np.zeros((n_tr, len(terminal_names)))
    for k, name in enumerate(terminal_names):
        t_idx = terminal_groups[name]
        R[:, k] = T[np.ix_(transient, t_idx)].sum(axis=1)
    # (I - Q) h = R  →  h = (I - Q)^{-1} R
    IQ = np.eye(n_tr) - Q
    try:
        from scipy.linalg import solve
        H = solve(IQ, R)   # (n_tr, n_terminals)
    except Exception:
        H = np.linalg.lstsq(IQ, R, rcond=None)[0]
    # Assembler la matrice complète
    abs_probs = np.zeros((n, len(terminal_names)))
    for i, ti in enumerate(transient):
        abs_probs[ti] = H[i]
    for k, name in enumerate(terminal_names):
        for j in terminal_groups[name]:
            abs_probs[j, k] = 1.0
    abs_probs = np.clip(abs_probs, 0, 1)
    # Normaliser
    row_sum = abs_probs.sum(axis=1, keepdims=True)
    row_sum[row_sum < 1e-9] = 1.0
    abs_probs /= row_sum
    return abs_probs, terminal_names


def _velocity_from_T(T, Z, V_ref=None):
    """Vitesse induite par T : v_T(i) = sum_j T_{ij} (z_j - z_i).
    Correspond à la direction privilégiée par la chaîne de Markov.
    """
    # v_T = T @ Z - Z (déplacement moyen pondéré)
    V_T = T @ Z - Z   # (n_cells, d)
    return V_T


# ---------------------------------------------------------------------------
# Interface principale
# ---------------------------------------------------------------------------

def cellrank_cosine(h5ad_path, cfg, P, te, n_neighbors=30, temperature=1.0,
                    progenitor_cluster='Ngn3 low EP', eps=1e-9):
    """Évalue la baseline CellRank-like sur le jeu de test.

    Returns
    -------
    dict avec cos_test, nrmse_test, absorption_probabilities (par cluster terminal),
         terminal_states, method, notes
    """
    Z = P['Z']  # (n_cells, d)
    V = P['V']  # vitesse scVelo mesurée
    clusters = np.array([str(c) for c in P['clusters']])
    split = P['split']
    te_arr = np.asarray(te)
    result = dict(method='cellrank_markov', n_neighbors=n_neighbors, temperature=temperature)

    # Essai d'utiliser la velocity_graph déjà calculée depuis l'h5ad
    vgraph = None
    try:
        import scanpy as sc, os, warnings
        if os.path.exists(h5ad_path):
            a = sc.read_h5ad(h5ad_path)
            if 'velocity_graph' in a.uns:
                vg = a.uns['velocity_graph']
                if hasattr(vg, 'toarray'):
                    vgraph = vg.toarray()
                else:
                    vgraph = np.asarray(vg)
                result['source'] = 'velocity_graph_scVelo'
    except Exception as e:
        result['source_warning'] = str(e)[:200]

    if vgraph is not None:
        # Construire T depuis la velocity_graph scVelo (cosinus → softmax)
        # Remplacer les -inf et nans
        vgraph_clean = np.where(np.isfinite(vgraph), vgraph, -1e9)
        T = _softmax_row(vgraph_clean, T=temperature)
        # Remettre à 0 les connexions absentes dans le graphe original
        absent = (vgraph == 0)
        np.fill_diagonal(absent, False)
        T[absent] = 0.0
        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-15] = 1.0
        T /= row_sum
        result['source'] = 'velocity_graph_scVelo'
    else:
        # Calculer T directement depuis (Z, V)
        # On utilise seulement les cellules d'entraînement+validation pour l'ajustement kNN
        # mais on calcule sur toutes les cellules
        result['source'] = 'computed_from_Z_V'
        T = _build_transition_matrix(Z, V, n_neighbors=n_neighbors, temperature=temperature)

    # Vitesse induite par T
    V_T = _velocity_from_T(T, Z)   # (n_cells, d)

    # Métriques sur le jeu de test
    V_T_te = V_T[te_arr]
    V_ref_te = V[te_arr]
    cos = _cosine(V_T_te, V_ref_te, eps)
    nrmse = _nrmse(V_T_te, V_ref_te)
    result.update(cos_test=float(cos), nrmse_test=float(nrmse))

    # États terminaux et probabilités d'absorption
    try:
        terminal_groups = _identify_terminal_states(T, clusters,
                                                     progenitor_cluster=progenitor_cluster)
        abs_probs, term_names = _absorption_probabilities(T, terminal_groups)
        # Distribution des probabilités d'absorption sur les cellules de test
        abs_te = abs_probs[te_arr]  # (n_te, n_terminals)
        # Cluster dominant prédit par CellRank pour chaque cellule de test
        pred_cluster = [term_names[int(np.argmax(row))] for row in abs_te]
        true_cluster = clusters[te_arr].tolist()
        # Accord entre cluster CellRank et cluster annoté (pureté)
        correct = sum(p == t for p, t in zip(pred_cluster, true_cluster))
        result['absorption'] = {
            'terminal_states': term_names,
            'terminal_groups': {k: len(v) for k, v in terminal_groups.items()},
            'cluster_purity_test': float(correct / max(len(pred_cluster), 1)),
            'mean_abs_prob_top_test': float(abs_te.max(axis=1).mean()),
        }
    except Exception as e:
        result['absorption_error'] = str(e)[:300]

    return result


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _cosine(A, B, eps=1e-9):
    a = np.linalg.norm(A, axis=1); b = np.linalg.norm(B, axis=1)
    m = (a > eps) & (b > eps)
    if not m.any():
        return float('nan')
    return float(np.mean(np.sum(A[m] * B[m], 1) / (a[m] * b[m])))


def _nrmse(Fp, Ft):
    rng = Ft.max() - Ft.min()
    return float(np.sqrt(np.mean((Fp - Ft) ** 2)) / (rng if rng > 1e-12 else 1.0))
