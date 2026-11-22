# Annexe B — Pseudo-codes des algorithmes principaux

## B.1 Entraînement du Neural ODE (données réelles)

```
Algorithme: train_vel_node
Entrées: Z (n_cells × d), V (vitesses), W (poids), tr/va (indices train/val), cfg, seed
Sorties: f_θ (réseau entraîné), métriques d'entraînement

1. Fixer seed PyTorch et NumPy
2. Construire sous-ensemble train par sous-échantillonnage (taux 0.8)
3. Calculer norme de vitesse ν = mean(w_i * ||v_i||²)
4. Initialiser f_θ = MLP(d → 64 → 64 → d, activation Tanh)
5. Initialiser optimiseur Adam (lr=0.003)
6. Pour chaque époque e = 0 ... 599:
   a. pred = f_θ(z_train)
   b. L_vel = mean(w * ||v - pred||²) / ν
   c. R = mean(||pred||²) / ν
   d. loss = L_vel + λ * R
   e. Rétropropagation + clip_grad_norm(5.0) + Adam step
   f. Calculer L_val sur l'ensemble de validation
   g. Si L_val améliore le meilleur: sauvegarder poids
   h. Si patience > 80 sans amélioration: arrêt anticipé
7. Charger les meilleurs poids
Retourner f_θ
```

## B.2 Sélection de structure symbolique (bibliothèque Hill)

```
Algorithme: select_structure
Entrées: Z_tr/Z_va, y_tr/y_va (cibles du champ), i (gène cible), lib (config), rng
Sorties: modèle optimal, front de Pareto

1. Construire la bibliothèque de candidats Φ:
   - Colonne constante (1 candidat)
   - Pour chaque gène j, type (rep/act), K ∈ K_grid, n ∈ n_grid:
     → ajouter H±(z_j; K, n) (8 × 5 × 2 × d candidats)
   
2. Front de Pareto ← [{k=0, Ω=0, MSE_val=mean(y_va²)}]

3. Pour k = 1 à k_max (= 3):
   a. Sélection avide: pour chaque candidat non encore sélectionné
      (en évitant les doublons (j, type)):
      → Ajuster par NNLS(Φ[:,sel+c], -Z[:,i], y)
      → Garder le candidat minimisant le RSS
   b. Ajouter le meilleur candidat à sel
   c. Raffinement TRF (n_starts départs):
      → Paramétrer (a_k, K_k, n_k, γ_i) dans les bornes
      → Minimiser ||f_θ(Z) - pred(Z)||² par trust-region
      → Calculer dispersion des solutions proches (spread)
   d. Ajouter (k, Ω, MSE_val) au front de Pareto

4. Trouver MSE_min = min(front)
5. Choisir le modèle de plus petite complexité Ω tel que
   MSE_val ≤ (1 + 0.05) × MSE_min

Retourner modèle choisi, front complet
```

## B.3 Forme faible (Weak-SINDy)

```
Algorithme: weak_select
Entrées: X (traj × temps × d), t, bibliothèque Hill, cfg
Sorties: modèle symbolique par forme faible

1. Construire fonctions test φ_s(t) = [(t-a)(b-t)]² normalisées
   sur chaque fenêtre glissante (taille w=5, stride=1)

2. Calculer par fenêtre:
   A_s = ∫φ_s H_k(x) dt  (intégrale des candidats)
   D_s = ∫φ_s x_i dt     (intégrale de l'état)
   Y_s = -∫φ'_s x_i dt   (intégrale par parties de ẋ)
   → Évite les dérivées numériques directes

3. Concaténer toutes les fenêtres: A, D, Y

4. Sélection avide par NNLS:
   Pour chaque équation i:
   a. Ajuster [A[:,sel], -D] @ c = Y par NNLS
   b. Calculer MSE_val sur fenêtres de validation
   c. Front de Pareto (Ω, MSE_val)
   d. Choisir par règle Pareto préenregistrée (tol=5%)

Retourner modèles (un par gène)
```

## B.4 Distillation du champ Neural ODE

```
Algorithme: sample_Df (distillation)
Entrées: f_θ (réseau entraîné), Z0_train, t_end, dist_cfg
Sorties: M points pseudo-trajectoires pour identification symbolique

1. Intégrer les trajectoires depuis Z0_train sur [0, t_end]
   → n_fine=40 points → rec (trajectoires denses)

2. Calculer les distances kNN dans l'espace de l'état
   → thr = quantile(d_kNN, 0.95)  [seuil de densité]

3. Générer M=4000 candidats:
   a. Échantillonner des points i de rec
   b. Perturber: z_cand = rec[i] + N(0, 0.5 * d_kNN[i] / √d)
   c. Clipper à ≥ 0 (non-négativité des expressions)

4. Rejeter les candidats hors support (d_kNN > thr)

5. Combiner: M/2 points de rec + M/2 points cand acceptés

6. Calculer les cibles y = f_θ(Z_distill) [champ évalué]

Retourner Z_distill, y_distill
```

## B.5 Vérification de la Proposition 1

```
Algorithme: prop1_check
Entrées: F_true, F_hat, trajectoires de test, temps t, CI Z0
Sorties: fraction de trajectoires satisfaisant la borne

Pour chaque trajectoire de test r:
1. ε = max_t ||F_true(z_r(t)) - F_hat(z_r(t))||  [erreur de champ]

2. Intégrer la trajectoire symbolique depuis Z0[r] avec F_hat
   → Trajectoire Ẑ_r(t) [via solve_ivp LSODA]

3. E = max_t ||z_r(t) - Ẑ_r(t)||  [erreur de trajectoire]

4. Estimer L = max_pts σ_max(∇F_hat) par Jacobien FD (h=1e-4)

5. Calculer B = (ε/L)(e^{L·T} - 1)

6. Vérifier: holds = (E ≤ B + tolérance numérique 1e-12)

Retourner: fraction holds, médiane B/E, statistiques ε, L, E
```

## B.6 Validation interventionnelle in silico (Knock-out)

```
Algorithme: ko_eval
Entrées: F_true, F_sym, F_node, Z0 (CI de test), T (horizon)
Sorties: cosinus et accord de signe des décalages KO

1. Intégrer l'état sauvage (wild-type) pour F_true, F_sym, F_node
   → wt_true, wt_sym, wt_node (via RK4 pas fixe)

Pour chaque gène k:
2. Construire le champ KO: z_k = 0, dz_k = 0 (délétion)
3. Intégrer depuis Z0 avec z_k = 0
4. Calculer le décalage: Δ = finale_KO - finale_wt (sur gènes k' ≠ k)
5. Cosinus(Δ_sym, Δ_true) et accord de signe (entrées |Δ_true| > 0.05)

Retourner cosinus moyens et fractions de signe correct
```
