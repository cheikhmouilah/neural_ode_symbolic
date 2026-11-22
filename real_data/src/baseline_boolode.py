"""Baseline BoolODE-like : simulation d'un réseau booléen continu par des EDO de Hill.

Implémente l'approche de BoolODE (Pratapa et al. 2020, Nature Methods 17:195–204) :
chaque gène g_i suit  dz_i/dt = β·prod(H(z_j; K, n, sign)) - γ·z_i
où les interactions sont déduites de la structure booléenne inférée depuis les données.

Le réseau d'interactions est inféré par régression de Granger (causalité temporelle) sur
les vitesses scVelo — ou importé depuis la structure symbolique si disponible.

Comparaison avec notre approche : notre bibliothèque Hill + sélection symbolique vs BoolODE
(interactions binaires fixes, paramètres K=0.5, n=2 uniformes, pas de raffinement).

Usage :
    from baseline_boolode import boolode_cosine
    result = boolode_cosine(P, te, symbolic_models=None)
"""
import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Inférence du réseau booléen
# ---------------------------------------------------------------------------

def _infer_granger_network(Z, V, threshold=0.0, max_in_degree=3):
    """Infère un réseau d'interactions signé par régression de Granger.
    Pour chaque gène i : on régresse V[:, i] sur Z[:, j] (j != i) ;
    les coefficients significativement positifs → activation, négatifs → répression.

    Returns
    -------
    network : dict  i → list[(j, sign)]   où sign ∈ {+1, -1}
    """
    d = Z.shape[1]
    network = {i: [] for i in range(d)}
    for i in range(d):
        # régression V[:, i] ~ sum_j a_j * Z[:, j]  (tous sauf autoterme)
        others = [j for j in range(d) if j != i]
        if not others:
            continue
        X = Z[:, others]
        y = V[:, i]
        # Moindres carrés ordinaires
        coef, _, _, _ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)
        coef_genes = coef[:len(others)]
        # Seuillage
        abs_coef = np.abs(coef_genes)
        order = np.argsort(-abs_coef)[:max_in_degree]
        for rank in order:
            c = coef_genes[rank]
            if abs(c) > threshold:
                j = others[rank]
                sign = +1 if c > 0 else -1
                network[i].append((j, sign))
    return network


def _infer_network_from_symbolic(symbolic_models):
    """Extrait le réseau d'interactions directement des modèles symboliques.

    Parameters
    ----------
    symbolic_models : list de modèles (dict) issus de real_identify.select_structure
        Chaque modèle contient 'i' (gène cible), 'terms' avec 'kind', 'j', 'typ'.

    Returns
    -------
    network : dict i → list[(j, sign)]
    """
    network = {}
    for m in symbolic_models:
        i = m['i']
        edges = []
        for t in m.get('terms', []):
            if t['kind'] == 'hill':
                j = t['j']
                sign = -1 if t['typ'] == 'rep' else +1
                if (j, sign) not in edges:
                    edges.append((j, sign))
        network[i] = edges
    return network


# ---------------------------------------------------------------------------
# Simulation BoolODE-like (EDO de Hill uniformes)
# ---------------------------------------------------------------------------

def _hill(x, K=0.5, n=2):
    """Fonction de Hill activatrice H+(x; K, n) = x^n / (K^n + x^n)."""
    xn = np.maximum(x, 0.0) ** n
    return xn / (K ** n + xn)


def _hill_rep(x, K=0.5, n=2):
    """Fonction de Hill répressive H-(x; K, n) = 1 - H+(x; K, n)."""
    return 1.0 - _hill(x, K, n)


def _boolode_field(Z, network, beta=1.0, gamma=1.0, K=0.5, n=2):
    """Calcule le champ de vitesse BoolODE-like sur les cellules Z.

    dz_i/dt = β * (prod des régulateurs de i, selon leur signe) - γ * z_i

    Returns
    -------
    V_bool : ndarray (n_cells, d)
    """
    n_cells, d = Z.shape
    V_bool = np.zeros_like(Z)
    for i in range(d):
        regs = network.get(i, [])
        if regs:
            # Produit des régulateurs (hypothèse BoolODE : AND logique)
            prod = np.ones(n_cells)
            for j, sign in regs:
                x_j = Z[:, j]
                if sign > 0:
                    prod *= _hill(x_j, K=K, n=n)
                else:
                    prod *= _hill_rep(x_j, K=K, n=n)
            V_bool[:, i] = beta * prod - gamma * Z[:, i]
        else:
            # Pas de régulateur connu → décroissance pure
            V_bool[:, i] = -gamma * Z[:, i]
    return V_bool


# ---------------------------------------------------------------------------
# Estimation des paramètres β, γ par gène (optionnel, améliore BoolODE)
# ---------------------------------------------------------------------------

def _fit_boolode_params(Z_tr, V_tr, network, K=0.5, n=2):
    """Ajuste β_i et γ_i par moindres carrés non-négatifs (style raffinement BoolODE).

    Returns
    -------
    params : dict  i → {'beta': float, 'gamma': float}
    """
    n_cells, d = Z_tr.shape
    params = {}
    for i in range(d):
        regs = network.get(i, [])
        if regs:
            prod = np.ones(n_cells)
            for j, sign in regs:
                prod *= _hill(Z_tr[:, j], K=K, n=n) if sign > 0 else _hill_rep(Z_tr[:, j], K=K, n=n)
            # Design : [prod, -z_i]  →  coef [β, γ]  (non-négatifs)
            A = np.column_stack([prod, Z_tr[:, i]])
            y = V_tr[:, i]
            coef, _ = nnls(A, y, maxiter=1000)
            beta_i = max(float(coef[0]), 1e-9)
            gamma_i = max(float(coef[1]), 1e-9)
        else:
            # Ajuster γ seulement
            gamma_i = max(float(np.dot(Z_tr[:, i], V_tr[:, i]) /
                                 max(np.dot(Z_tr[:, i], Z_tr[:, i]), 1e-15)), 1e-9)
            beta_i = 1e-9
        params[i] = dict(beta=beta_i, gamma=gamma_i)
    return params


def _boolode_field_fitted(Z, network, params, K=0.5, n=2):
    """Champ BoolODE avec β et γ ajustés par gène."""
    n_cells, d = Z.shape
    V_bool = np.zeros_like(Z)
    for i in range(d):
        regs = network.get(i, [])
        beta_i = params[i]['beta']
        gamma_i = params[i]['gamma']
        if regs:
            prod = np.ones(n_cells)
            for j, sign in regs:
                prod *= _hill(Z[:, j], K=K, n=n) if sign > 0 else _hill_rep(Z[:, j], K=K, n=n)
            V_bool[:, i] = beta_i * prod - gamma_i * Z[:, i]
        else:
            V_bool[:, i] = -gamma_i * Z[:, i]
    return V_bool


# ---------------------------------------------------------------------------
# Interface principale
# ---------------------------------------------------------------------------

def boolode_cosine(P, te, symbolic_models=None, K=0.5, n=2.0,
                   fit_params=True, eps=1e-9):
    """Évalue la baseline BoolODE-like sur le jeu de test.

    Parameters
    ----------
    P : dict (prépared data)
    te : indices de test
    symbolic_models : list[dict] de modèles symboliques (si disponibles, le réseau est extrait directement)
                      Si None : réseau inféré par régression de Granger sur (Z_tr, V_tr)
    K, n : paramètres Hill de BoolODE (par défaut : K=0.5, n=2 comme dans BoolODE original)
    fit_params : bool — si True, ajuste β_i et γ_i par moindres carrés (amélioré) ;
                        si False, utilise β=1, γ=1 (BoolODE strict)

    Returns
    -------
    dict avec cos_test, nrmse_test, network, params, method, notes
    """
    Z = P['Z']
    V = P['V']
    split = P['split']
    te_arr = np.asarray(te)
    tr = np.where(split == 0)[0]   # train seulement pour l'inférence du réseau
    genes = [str(g) for g in P['genes']]

    result = dict(method='boolode_hill', K=K, n=n, fit_params=fit_params, genes=genes)

    # --- Inférence du réseau ---
    if symbolic_models is not None:
        network = _infer_network_from_symbolic(symbolic_models)
        result['network_source'] = 'symbolic_models'
    else:
        network = _infer_granger_network(Z[tr], V[tr])
        result['network_source'] = 'granger_regression'

    # Description du réseau
    network_desc = {}
    for i, edges in network.items():
        if edges:
            network_desc[genes[i]] = [(genes[j], '+' if s > 0 else '-') for j, s in edges]
    result['network'] = network_desc

    # --- Paramètres ---
    if fit_params:
        params = _fit_boolode_params(Z[tr], V[tr], network, K=K, n=n)
        V_bool_all = _boolode_field_fitted(Z, network, params, K=K, n=n)
        result['params'] = {genes[i]: {'beta': params[i]['beta'], 'gamma': params[i]['gamma']}
                            for i in range(len(genes))}
    else:
        V_bool_all = _boolode_field(Z, network, beta=1.0, gamma=1.0, K=K, n=n)
        result['params'] = 'uniform_beta=1_gamma=1'

    # --- Métriques ---
    V_bool_te = V_bool_all[te_arr]
    V_ref_te = V[te_arr]
    cos = _cosine(V_bool_te, V_ref_te, eps)
    nrmse = _nrmse(V_bool_te, V_ref_te)
    result.update(cos_test=float(cos), nrmse_test=float(nrmse))

    # Variante sans ajustement (BoolODE strict) pour comparaison
    if fit_params:
        V_bool_strict_te = _boolode_field(Z[te_arr], network, beta=1.0, gamma=1.0, K=K, n=n)
        result['cos_test_strict'] = float(_cosine(V_bool_strict_te, V_ref_te, eps))

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
