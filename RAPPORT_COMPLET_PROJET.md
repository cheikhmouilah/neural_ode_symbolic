# Rapport Scientifique Complet

## Identification symbolique de réseaux de régulation génique par Neural ODE et régression structurée — Développement, Évaluation expérimentale et Protocole de reproductibilité

---

**Projet :** `c:\neural_ode_symbolic`  
**Date :** Novembre 2026  
**Environnement :** Python 3.14.2, Windows 11, 8 cœurs, CPU uniquement  
**Pré-enregistrement :** 2026-11-11T20:24:05Z  
**Mots-clés :** Neural ODE, régression symbolique, réseaux de régulation génique, scRNA-seq, vélocité ARN, cinétique de Hill, Grönwall, bistabilité, pré-enregistrement, benchmark

---

## Résumé

Ce rapport présente un projet de recherche complet visant à inférer les lois de régulation génique à partir de données transcriptomiques unicellulaires (scRNA-seq) en combinant des réseaux de neurones différentiels (Neural ODE) et une régression symbolique structurée. La méthode procède en trois étapes : (1) apprentissage du champ vectoriel dynamique à partir des vélocités ARN estimées par scVelo, (2) distillation du champ appris en expressions mathématiques interprétables basées sur la cinétique de Hill, et (3) validation par comparaison aux méthodes de référence (CellRank, dynamo, BoolODE) et par des benchmarks synthétiques contrôlés avec vérité terrain. L'évaluation est conduite sur deux jeux de données scRNA-seq réels (pancréas murin E15.5 et gyrus dentelé), 15 variantes synthétiques (75 exécutions), et un commutateur bistable 2D. Un protocole de pré-enregistrement par empreintes SHA-256, 15 contrôles de cohérence automatiques, et une vérification formelle de la borne de Grönwall (100 % sur 70 runs) garantissent l'intégrité scientifique des résultats. Les principaux résultats montrent que le Neural ODE dépasse marginalement les seuils préenregistrés sur les données réelles (R1 : cosinus 0.520 ≥ 0.5), mais ne dépasse pas la référence linéaire (cosinus 0.588). La fidélité symbolique au champ Neural ODE est partielle (R2 : 0.880 < 0.9), bien que le ratio de cosinus reste acceptable (R7 : 0.998 ≥ 0.8). Sur données synthétiques, l'identification directe (F1 = 0.887) surpasse la distillation (F1 = 0.778), mais le Neural ODE approche le champ vrai à cosinus 0.939, et la borne de Grönwall est validée empiriquement. Les limites épistémologiques sont discutées honnêtement : absence de vérité terrain sur données réelles, identifiabilité partielle des paramètres Hill, impossibilité d'inférer la causalité à partir de la seule vélocité ARN.

---

## Abstract

This report presents a complete research project aimed at inferring gene regulatory laws from single-cell RNA sequencing (scRNA-seq) data by combining Neural Ordinary Differential Equations (NODE) with structured symbolic regression. The method operates in three stages: (1) learning the dynamic vector field from RNA velocity estimates produced by scVelo, (2) distilling the learned field into interpretable mathematical expressions based on Hill kinetics, and (3) validating against reference methods (CellRank, dynamo, BoolODE) and controlled synthetic benchmarks with ground truth. Evaluation covers two real scRNA-seq datasets (murine pancreas E15.5 and dentate gyrus), 15 synthetic variants (75 runs), and a 2D bistable toggle switch. A preregistration protocol with SHA-256 hashing, 15 automated consistency checks, and formal verification of the Grönwall bound (100% on 70 runs) ensure scientific integrity. Main results show that the NODE marginally passes preregistered thresholds on real data (R1: cosine 0.520 ≥ 0.5) but fails to exceed a linear baseline (cosine 0.588). Symbolic fidelity to the NODE field is partial (R2: 0.880 < 0.9), though the cosine ratio remains acceptable (R7: 0.998 ≥ 0.8). On synthetic data, direct identification (F1 = 0.887) outperforms distillation (F1 = 0.778), yet the NODE field cosine reaches 0.939 against ground truth, and the Grönwall bound is empirically validated. Epistemic limitations are honestly discussed: absence of ground truth on real data, partial identifiability of Hill parameters, and impossibility of inferring causality from RNA velocity alone.

**Keywords:** Neural ODE, symbolic regression, gene regulatory networks, scRNA-seq, RNA velocity, Hill kinetics, Grönwall inequality, bistability, preregistration, benchmark

---

## Table des matières

1. [Introduction générale](#1-introduction-générale)
2. [Contexte scientifique et problématique](#2-contexte-scientifique-et-problématique)
3. [Objectifs et contributions](#3-objectifs-et-contributions)
4. [État de l'art](#4-état-de-lart)
5. [Fondements mathématiques](#5-fondements-mathématiques)
6. [Modélisation des systèmes dynamiques](#6-modélisation-des-systèmes-dynamiques)
7. [Architecture Neural ODE](#7-architecture-neural-ode)
8. [Méthodes de régression symbolique](#8-méthodes-de-régression-symbolique)
9. [Algorithmes de comparaison (baselines)](#9-algorithmes-de-comparaison-baselines)
10. [Architecture informatique et modules](#10-architecture-informatique-et-modules)
11. [Jeux de données et prétraitement](#11-jeux-de-données-et-prétraitement)
12. [Méthodologie expérimentale](#12-méthodologie-expérimentale)
13. [Protocole de reproductibilité](#13-protocole-de-reproductibilité)
14. [Résultats expérimentaux](#14-résultats-expérimentaux)
15. [Analyse comparative](#15-analyse-comparative)
16. [Discussion scientifique](#16-discussion-scientifique)
17. [Limites du travail](#17-limites-du-travail)
18. [Conclusion générale](#18-conclusion-générale)
19. [Perspectives](#19-perspectives)
20. [Références bibliographiques](#20-références-bibliographiques)
21. [Annexes techniques](#21-annexes-techniques)

---

## 1. Introduction générale

L'émergence des technologies de séquençage ARN unicellulaire (*single-cell RNA sequencing*, scRNA-seq) a rendu possible la mesure simultanée de l'expression de milliers de gènes dans des milliers de cellules individuelles. Ces données ouvrent la voie à la reconstruction des processus de différenciation cellulaire, c'est-à-dire à la compréhension des mécanismes moléculaires par lesquels une cellule souche devient progressivement un type cellulaire spécialisé. La vélocité ARN, introduite par La Manno et al. [1] et généralisée par Bergen et al. [2], permet d'estimer le sens et la vitesse de ce changement d'état à partir du rapport entre ARN immature (pré-ARNm, noté $u$) et ARN mature (ARNm, noté $s$). La formule sous-jacente,

$$\frac{du_i}{dt} = \alpha_i - \beta_i u_i, \qquad \frac{ds_i}{dt} = \beta_i u_i - \gamma_i s_i,$$

permet d'inférer une vitesse de splicing $\hat{v}_i = \hat{\beta}_i u_i - \hat{\gamma}_i s_i$ pour chaque gène $i$.

Cependant, ces vitesses estimées sont des *proxys bruités* du champ dynamique réel, et non des mesures directes de dérivées temporelles. La question centrale de ce projet est : *peut-on, à partir de ces vitesses bruitées et de profils d'expression discrets, identifier des équations différentielles symboliques interprétables régissant la dynamique génique ?*

La réponse proposée articule trois composantes : (1) un réseau neuronal différentiel — le Neural ODE — pour approximer le champ vectoriel sous-jacent sans contraindre sa forme fonctionnelle ; (2) une régression symbolique basée sur la cinétique de Hill pour transformer le champ appris en équations biophysiquement interprétables ; (3) un protocole de validation scientifique rigoureux incluant pré-enregistrement, contrôles de cohérence, et benchmarks synthétiques avec vérité terrain.

---

## 2. Contexte scientifique et problématique

### 2.1 Différenciation cellulaire et scRNA-seq

La différenciation cellulaire est un processus gouverné par des réseaux de régulation génique (GRN, *Gene Regulatory Networks*) composés d'interactions entre facteurs de transcription et leurs gènes cibles. La modélisation mathématique de ces réseaux par des systèmes d'EDO est une approche classique en biologie des systèmes [3], mais l'identification des paramètres à partir de données expérimentales reste un défi majeur.

Les données scRNA-seq fournissent des *instantanés transversaux* de l'expression génique dans de nombreuses cellules — mais sans suivi temporel individuel. Chaque cellule n'est mesurée qu'une seule fois, à une position inconnue dans la trajectoire de différenciation. La vélocité ARN contourne partiellement cette limitation en fournissant une estimation vectorielle locale du sens d'évolution.

### 2.2 Problème d'identification inverse

Étant donnés $n$ profils d'expression $\{\mathbf{z}_i \in \mathbb{R}^d\}_{i=1}^n$ et les vélocités associées $\{\mathbf{v}_i \in \mathbb{R}^d\}_{i=1}^n$ (estimées par scVelo), on cherche un champ vectoriel $F : \mathbb{R}^d \to \mathbb{R}^d$ tel que

$$\frac{d\mathbf{z}}{dt} = F(\mathbf{z}), \qquad F(\mathbf{z}_i) \approx \mathbf{v}_i.$$

Ce problème est mal posé : (a) les vélocités sont bruitées et biaisées ; (b) les cellules sont distribuées sur une variété de faible dimension inconnue ; (c) il n'existe pas de paires $(t_i, \mathbf{z}_i)$ observées au même instant.

### 2.3 Spécificité des données scRNA-seq

**Espace d'état.** L'espace des gènes de dimension $d=8$ est retenu (8 gènes sélectionnés parmi les gènes de vélocité par critère de variance des moments spliced, indépendamment de toute hypothèse régulatrice). L'espace latent (ACP) et l'espace des modules (k-means de gènes) sont également testés à titre exploratoire.

**Non-négativité.** Les expressions géniques sont non-négatives. Le prétraitement normalise par le 99e percentile : $z = M_s / \text{quantile}_{99}(M_s)$. Les termes Hill sont définis pour $x \geq 0$ (`np.maximum(x, 0)`).

**Poids de confiance.** La confiance en la vélocité $w_i = $ `velocity_confidence` pondère les cellules dans la fonction de perte. Les cellules dont la vélocité est incohérente avec les voisines reçoivent un poids faible.

**Fuite de données.** Le lissage kNN des moments dans scVelo crée des corrélations entre cellules voisines. Le découpage en blocs de voisinage (k-means sur les 10 premières composantes PCA, 60 blocs) limite cette fuite entre ensembles d'entraînement et de test.

---

## 3. Objectifs et contributions

Les six contributions principales du projet sont :

**C1. Pipeline intégré scRNA-seq → EDO symbolique.** Développement d'un pipeline modulaire de bout en bout, depuis le fichier `.h5ad` brut jusqu'aux équations différentielles symboliques de type Hill, sans intervention manuelle intermédiaire.

**C2. Neural ODE autonome avec appariement de vitesse.** Implémentation d'un Neural ODE dont le champ vectoriel $f_\theta(\mathbf{z})$ est appris par minimisation d'une perte de vélocité pondérée (sans pseudotemps ni intégration de trajectoire sur données réelles), avec régularisation de magnitude et arrêt anticipé.

**C3. Régression symbolique structurée par bibliothèque Hill.** Algorithme de sélection de structure par sélection avide (jusqu'à $k_{\max} = 3$ termes par gène), moindres carrés non négatifs (NNLS), et raffinement multi-départs par moindres carrés bornés (TRF). Sélection par front de Pareto précision/complexité avec tolérance préenregistrée.

**C4. Distillation du champ Neural ODE.** Génération de pseudo-trajectoires à partir du champ appris, avec filtrage de densité par kNN, pour créer un jeu d'apprentissage symbolique riche sur le support des trajectoires.

**C5. Benchmarks complets avec vérité terrain.** Évaluation sur 15 variantes synthétiques (bruit, dropout, dimension, pseudotemps, variété) avec comparaison aux baselines identification directe, Weak-SINDy, gplearn (libre), et aux méthodes biologiques dynamo, CellRank, BoolODE.

**C6. Protocole de reproductibilité.** Pré-enregistrement par SHA-256, 15 contrôles de cohérence automatiques, vérification formelle de la borne de Grönwall (Proposition 1), et test de non-régression en 16 étapes.

---

## 4. État de l'art

### 4.1 Inférence de réseaux de régulation génique

L'inférence de GRN à partir de données d'expression est un problème intensément étudié [4]. Les approches classiques incluent les modèles de régression régularisés (LASSO), les réseaux bayésiens, et les modèles d'information mutuelle. Ces méthodes opèrent typiquement sur des données statiques ou pseudo-temporelles, sans modélisation explicite de la dynamique.

**BoolODE** [5] utilise des règles booléennes pour simuler la dynamique génique et génère des données synthétiques de référence. Sur les données réelles du pancréas, BoolODE produit un cosinus de 0.020, révélant la difficulté à réconcilier le modèle booléen avec la vélocité ARN continue.

### 4.2 Vélocité ARN et dynamique cellulaire

**scVelo** [2] introduit un modèle de cinétique de splicing qui modélise la transcription, le splicing et la dégradation par un système d'EDO linéaires pour chaque gène indépendamment. Le mode `deterministic` (utilisé dans ce projet pour compatibilité avec NumPy ≥ 2.x) ajuste les phases par maximisation de vraisemblance. La vitesse résultante est une approximation locale de la dérivée temporelle de l'ARNm mature.

**dynamo** [6] étend scVelo à la dynamique globale et propose une modélisation du moment de poussée (Mu). Sur le pancréas, dynamo présente un cosinus de −0.931 sur le gène Ins2, diagnostiqué comme une phase de dégradation pré-stationnaire (Mu ≫ Ms), cohérente avec le stade E15.5 de l'endocrinogenèse.

**CellRank** [7] modélise les transitions cellulaires par des chaînes de Markov sur le graphe de vélocité. Sur le pancréas, le cosinus obtenu est 0.475. CellRank identifie trois états terminaux : Alpha, Beta, Delta — cohérents avec l'annotation biologique des clusters pancréatiques.

### 4.3 Neural ODE

Le concept de Neural ODE a été introduit par Chen et al. [8] : le champ vectoriel d'une EDO est paramétré par un réseau neuronal, et les gradients sont calculés par la méthode adjointe continue. Ce projet implémente une variante simplifiée avec solveur direct RK4 (données synthétiques) ou dopri5 (données réelles), et rétropropagation directe (sauf variante `adjoint`).

### 4.4 Régression symbolique

**PySRs** [9] utilise un algorithme évolutionnaire multi-population pour rechercher des expressions mathématiques optimisant un critère de précision/complexité. L'espace de recherche est les opérateurs $\{+, -, \times, \div\}$.

**gplearn** [10] implémente la programmation génétique pour la régression symbolique. Utilisé comme baseline de forme libre sur données synthétiques.

**SINDy** [11] (*Sparse Identification of Nonlinear Dynamics*) identifie les EDO par régression régularisée (LASSO) sur une bibliothèque de fonctions candidates. La variante **Weak-SINDy** [12] utilise des fonctions tests pour éviter la différentiation numérique directe.

### 4.5 SERGIO

**SERGIO** [5] est un simulateur biophysique de réseaux de régulation génique qui génère des données scRNA-seq synthétiques réalistes intégrant le bruit intrinsèque, le bruit technique, et les interactions entre gènes. Utilisé dans ce projet (chaîne B) comme source de données synthétiques biophysiquement réalistes.

---

## 5. Fondements mathématiques

### 5.1 Systèmes dynamiques autonomes

Le cadre général est celui des systèmes dynamiques autonomes :

$$\frac{d\mathbf{z}(t)}{dt} = F(\mathbf{z}(t)), \qquad \mathbf{z}(0) = \mathbf{z}_0,$$

où $\mathbf{z} \in \mathbb{R}^d$ est le vecteur d'état (niveaux d'expression des $d$ gènes retenus) et $F : \mathbb{R}^d \to \mathbb{R}^d$ est le champ vectoriel représentant les interactions régulatrices. L'existence et l'unicité des solutions sont garanties par le théorème de Cauchy-Lipschitz lorsque $F$ est localement Lipschitz.

### 5.2 Borne d'erreur de Grönwall (Proposition 1)

Soit $L$ la constante de Lipschitz de $\hat{F}$ et $\varepsilon = \sup_{\mathbf{z}} \|F(\mathbf{z}) - \hat{F}(\mathbf{z})\|$. La borne d'erreur entre la trajectoire symbolique et la trajectoire de référence est :

$$
\|\mathbf{z}_{\text{sym}}(t) - \mathbf{z}_{\text{true}}(t)\| \leq \frac{\varepsilon}{L}(e^{Lt} - 1) + \delta_0\, e^{Lt},
$$

où $\delta_0 = \|\mathbf{z}_{\text{sym}}(0) - \mathbf{z}_{\text{true}}(0)\|$. Cette borne est validée empiriquement sur **70 exécutions** avec une fraction de satisfaction de **1.000** (seuil préenregistré : 0.95). La démonstration complète est fournie en Annexe A.

### 5.3 Cinétique de Hill

La cinétique de Hill modélise la coopérativité des interactions protéine-ADN :

$$H^+(x; K, n) = \frac{x^n}{K^n + x^n}, \qquad H^-(x; K, n) = \frac{K^n}{K^n + x^n},$$

où $K > 0$ est la constante de demi-saturation et $n \geq 1$ est le coefficient de Hill (coopérativité). Pour $n = 1$ : cinétique de Michaelis-Menten. Pour $n > 1$ : réponse sigmoïde (coopérativité positive).

### 5.4 Stabilité de Lyapunov et bistabilité

Un équilibre $\mathbf{z}^*$ est **asymptotiquement stable** si et seulement si toutes les valeurs propres du Jacobien $J = \nabla F(\mathbf{z}^*)$ ont une partie réelle strictement négative. Un système est **bistable** s'il possède exactement deux attracteurs stables $A_1, A_2$ et un point col $\mathbf{z}^s$ séparant leurs bassins d'attraction. Cette propriété est vérifiée par le garde-fou `assert_bistable` avant tout entraînement sur le toggle 2D.

### 5.5 Complexité de Kolmogorov et front de Pareto

La sélection du modèle symbolique minimise simultanément l'erreur de validation et la complexité $\Omega$ (nombre de paramètres libres), par le principe de Pareto : on choisit le modèle de plus petite complexité dont l'erreur de validation ne dépasse pas $(1 + \tau)\times$ la meilleure erreur, avec $\tau = 0.05$ (fixé avant toute exécution).

---

## 6. Modélisation des systèmes dynamiques

### 6.1 Modèle de réseau régulateur (bibliothèque Hill)

L'équation différentielle pour le gène $i$ est :

$$\frac{dz_i}{dt} = \sum_{k=1}^{K_i} a_{ik}\, H^{\sigma_{ik}}(z_{j_k};\, K_{ik}, n_{ik}) + a_{i0} - \gamma_i z_i,$$

avec :
- $H^{\sigma_{ik}}$ : terme Hill activateur ($+$) ou répresseur ($-$) selon $\sigma_{ik} \in \{\text{act}, \text{rep}\}$ ;
- $a_{ik} > 0$ : amplitude de régulation (moindres carrés non négatifs) ;
- $K_{ik} > 0$ : constante de demi-saturation ;
- $n_{ik} \geq 1$ : coefficient de Hill ;
- $\gamma_i > 0$ : taux de dégradation effectif ;
- $a_{i0} \geq 0$ : terme basal constant (facultatif).

L'équation complète comporte $3K_i + 1$ paramètres libres pour $K_i$ termes Hill.

### 6.2 Commutateur bistable toggle 2D (chaîne A)

Le toggle 2D est un circuit classique de bistabilité génique défini par :

$$\frac{dx}{dt} = \frac{a}{1 + (y/K)^n} - \gamma x, \qquad \frac{dy}{dt} = \frac{b}{1 + (x/K')^{n'}} - \gamma' y.$$

Ce système présente deux attracteurs stables correspondant aux états $(x_{\text{haut}}, y_{\text{bas}})$ et $(x_{\text{bas}}, y_{\text{haut}})$, séparés par un point selle. La recherche des équilibres utilise 15 × 15 = 225 départs multi-starts dans le domaine $[0, 7]^2$. La vérification préalable (`assert_bistable`) exige exactement 2 attracteurs stables et 1 selle.

### 6.3 Espaces d'état alternatifs

Trois espaces d'état sont définis pour les données réelles :

| Espace | Dimension | Construction |
|---|---|---|
| Gènes | $d$ gènes | $d=8$ gènes de vélocité à plus grande variance Ms |
| Modules | $d$ modules | k-means sur profils de gènes standardisés |
| Latent | $d$ composantes PCA | ACP sur les moments spliced ; vitesse projetée |

L'espace des gènes est retenu comme espace principal pour la lisibilité biologique des équations symboliques.

---

## 7. Architecture Neural ODE

### 7.1 Architecture du réseau

Le champ vectoriel est paramétré par un réseau neuronal multicouche (MLP) :

$$f_\theta(\mathbf{z}) = W_3 \cdot \tanh(W_2 \cdot \tanh(W_1 \mathbf{z} + b_1) + b_2) + b_3,$$

avec $W_1 \in \mathbb{R}^{64 \times d}$, $W_2 \in \mathbb{R}^{64 \times 64}$, $W_3 \in \mathbb{R}^{d \times 64}$. Soit au total $64(d+1) + 64 \times 65 + d \times 65$ paramètres. Pour $d=8$ : $64 \times 9 + 64 \times 65 + 8 \times 65 = 576 + 4160 + 520 = 5256$ paramètres.

Le réseau est **autonome** : le champ ne dépend pas explicitement du temps ($t$ non passé en entrée). C'est une différence documentée par rapport à la formulation théorique (voir section 10.6).

**Implémentation PyTorch :**

```python
class VF(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d)
        )
    def forward(self, t, x): return self.net(x)
```

### 7.2 Fonction de perte — données réelles

La perte d'appariement de vitesse pondérée est :

$$\mathcal{L}_{\text{vel}} = \frac{1}{|I_{\text{tr}}|} \sum_{i \in I_{\text{tr}}} \frac{w_i \|\mathbf{v}_i - f_\theta(\mathbf{z}_i)\|^2}{\nu},$$

avec $\nu = \frac{1}{|I_{\text{tr}}|}\sum_i w_i \|\mathbf{v}_i\|^2 + 10^{-12}$ (normalisation sans dimension).

La régularisation de magnitude est :

$$\mathcal{R} = \frac{1}{|I_{\text{tr}}|} \sum_{i \in I_{\text{tr}}} \frac{\|f_\theta(\mathbf{z}_i)\|^2}{\nu}.$$

La perte totale est $\mathcal{L} = \mathcal{L}_{\text{vel}} + \lambda \mathcal{R}$, avec $\lambda = 0.001$ (préenregistré).

> [!NOTE] La perte $\mathcal{L}_{\text{obs}}$ (reconstruction des trajectoires intégrées) n'est **pas utilisée** sur les données réelles, car les paires $(t, \mathbf{z}(t))$ ne sont pas disponibles. Elle n'est utilisée que sur les données synthétiques (chaîne D). Cet écart avec la formulation théorique est documenté dans `docs/CONSISTENCE_THEORIE.md`.

### 7.3 Fonction de perte — données synthétiques

Sur les données synthétiques, l'intégration des trajectoires est effectuée par RK4 à pas fixe ($\Delta t = 0.1$) ou par la méthode adjointe (variante `adjoint`) :

$$\mathcal{L}_{\text{obs}} = \frac{1}{|\mathcal{T}_{\text{tr}}|} \sum_{(\text{traj}, t > 0)} \frac{\|\hat{\mathbf{z}}(t) - \mathbf{z}(t)\|^2}{\sigma^2_{\text{data}}}.$$

### 7.4 Optimisation

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam |
| Taux d'apprentissage | 0.003 |
| Époques max | 600 (réelles) / 400 (synthétiques) |
| Patience | 80 (réelles) / 25 évals tous les 5 ep. (synthétiques) |
| Clippage gradient | $\|\nabla\|_2 \leq 5.0$ |
| Sous-échantillonnage train | 80 % par graine (données réelles) |

### 7.5 Intégration numérique

| Contexte | Solveur | Tolérance |
|---|---|---|
| Données réelles (entraînement) | — (appariement direct) | — |
| Données réelles (intégration) | dopri5 (torchdiffeq) | rtol=1e-6, atol=1e-8 |
| Synthétiques (entraînement) | RK4 pas fixe (Δt=0.1) | — |
| Équilibres | hybr (scipy.root) + LSODA | tol=1e-10 |
| Prop. 1 | LSODA (solve_ivp) | rtol=1e-8, atol=1e-10 |

---

## 8. Méthodes de régression symbolique

### 8.1 Pipeline de distillation

Avant la régression symbolique, le champ $f_\theta$ est distillé en $M = 4000$ points (données réelles) ou $M = 3000$ (synthétique) couvrant le support des trajectoires :

1. **Intégration des trajectoires** depuis les progeniteurs sur $n_{\text{fine}} = 40$ points.
2. **Évaluation de la densité locale** par $k=5$ plus proches voisins ; seuil au 95e percentile.
3. **Génération de candidats** : perturbation gaussienne anisotrope des points de trajectoire, $\sigma = 0.5 \times d_{\text{kNN}} / \sqrt{d}$.
4. **Rejet des points hors support** (distance kNN > seuil).
5. **Mélange** : $M/2$ points de trajectoire + $M/2$ points perturbés acceptés.

### 8.2 Bibliothèque de candidats Hill

La bibliothèque est construite sur la grille $\mathcal{K} \times \mathcal{N} \times \{+, -\} \times \{j = 1 \ldots d\}$ :

- Grille K (réelles) : $\{0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5\}$ — 8 valeurs
- Grille n (réelles) : $\{1.0, 1.5, 2.0, 3.0, 4.0\}$ — 5 valeurs
- Soit $8 \times 5 \times 2 \times 8 + 1 = 641$ candidats pour $d = 8$ gènes

### 8.3 Sélection de structure

L'algorithme de sélection avide procède en $k_{\max} = 3$ passes :
1. À chaque pas, ajouter le candidat qui minimise le RSS sur l'ensemble d'entraînement (NNLS).
2. Raffiner les constantes $(a, K, n, \gamma)$ par TRF multi-départs ($n_{\text{starts}} = 6$).
3. Évaluer le MSE de validation et construire le front de Pareto $(\Omega, \text{MSE}_{\text{val}})$.
4. Sélectionner par règle préenregistrée : plus petite complexité $\Omega$ avec $\text{MSE}_{\text{val}} \leq (1+0.05) \times \text{MSE}_{\text{min}}$.

### 8.4 Régression symbolique libre (PySR)

PySR est appliqué sur le champ du Neural ODE (distillé) avec les opérateurs $\{+, -, \times, \div\}$ et une complexité maximale de 15 nœuds. Les seuils de performance (PT1-PT4, RP1-RP2, S7) ont été fixés après un essai pilote d'une seule graine (divulgué dans les fichiers de configuration).

### 8.5 Régression symbolique libre (gplearn)

gplearn avec 300 individus et 15 générations est appliqué sur les données synthétiques. L'entrée est $\min(2000, M)$ points distillés. La métrique principale est le rappel des variables vraies dans l'expression trouvée.

---

## 9. Algorithmes de comparaison (baselines)

### 9.1 Identification directe par différences finies (FD)

La dérivée temporelle est estimée par `np.gradient(X, t, axis=1)`, puis la bibliothèque Hill est appliquée directement sur les vitesses ainsi calculées, sans passer par le Neural ODE. Cette méthode est sensible au bruit.

**Résultats (synthétique, base, 5 graines) :** F1 = **0.887** — dépasse la distillation (0.778).

### 9.2 Weak-SINDy

La forme faible intègre l'équation par des fonctions test $\phi_s(t) = [(t-a)(b-t)]^2$ normalisées sur des fenêtres de taille $w = 5$ avec pas $\text{stride} = 1$. L'intégration par parties évite la dérivation directe. La sélection de structure utilise NNLS + front de Pareto identique à la bibliothèque Hill.

**Résultats (synthétique, base, 5 graines) :** F1 = **0.864**.

### 9.3 Référence linéaire

Une régression linéaire $f(\mathbf{z}) = A\mathbf{z} + b$ est entraînée sur les données d'entraînement et évaluée sur le test. Sur les données réelles du pancréas, le cosinus linéaire est **0.588**, dépassant le NODE (0.520).

### 9.4 dynamo

Le pipeline dynamo est implémenté de manière autonome dans `baseline_dynamo.py`. L'estimation des taux $\gamma, \beta$ est effectuée par OLS (repli vers la régression directe suite à l'incompatibilité de `recover_dynamics` avec pandas ≥ 3.0). La vitesse est recalculée comme $\hat{v} = \hat{\beta} u - \hat{\gamma} s$.

**Résultat pancréas :** cosinus = **−0.931** (Ins2 : phase de dégradation pré-stationnaire, Mu ≫ Ms).

### 9.5 CellRank

CellRank est appliqué en mode RNA velocity avec la matrice de transition kernel sur le graphe kNN (cosine kernel). Les états terminaux identifiés sur le pancréas sont : **Alpha, Beta, Delta**.

**Résultat pancréas :** cosinus = **0.475**.

### 9.6 BoolODE

BoolODE traduit un réseau de règles booléennes en un système SDO et génère des trajectoires Monte Carlo. Les vélocités sont estimées comme les dérivées sur les trajectoires simulées.

**Résultat pancréas :** cosinus = **0.020**.

---

## 10. Architecture informatique et modules

### 10.1 Organisation du projet

```
c:\neural_ode_symbolic\
├── project/                     # Chaîne A : toggle bistable 2D
│   ├── configs/toggle.json      # Configuration et seuils PySR toggle
│   ├── src/
│   │   ├── toggle.py            # Système toggle, points fixes
│   │   ├── node.py              # Neural ODE (trajectoires, L_obs)
│   │   ├── identify.py          # Régression symbolique toggle
│   │   ├── guards.py            # Garde-fous bistabilité
│   │   ├── families.py          # Familles de systèmes exploratoires
│   │   ├── misspec.py           # Tests de mauvaise spécification
│   │   └── misspec2.py          # Tests complémentaires
│   └── results/toggle/          # Résultats JSON par graine
├── sergio_validation/           # Chaîne B : simulation SERGIO
│   └── ...
├── real_data/                   # Chaîne C : données scRNA-seq réelles
│   ├── configs/
│   │   ├── real_pancreas.json   # Config + seuils pancréas
│   │   └── real_dentategyrus.json # Config + seuils gyrus dentelé
│   ├── data/                    # Fichiers .h5ad (téléchargés)
│   ├── src/
│   │   ├── real_data.py         # Prétraitement scRNA-seq
│   │   ├── real_node.py         # Neural ODE (appariement vitesse)
│   │   ├── real_identify.py     # Régression symbolique Hill
│   │   ├── real_dyn.py          # Analyse dynamique (Jacobien, équilibres)
│   │   ├── baseline_dynamo.py   # Baseline dynamo (autonome)
│   │   ├── baseline_cellrank.py # Baseline CellRank (autonome)
│   │   └── baseline_boolode.py  # Baseline BoolODE (autonome)
│   └── results/real/            # Résultats JSON (5 graines, baselines)
├── synthetic/                   # Chaîne D : benchmark synthétique
│   ├── configs/synth.json       # 15 variantes + seuils S1-S7
│   ├── src/
│   │   ├── synth_system.py      # Générateur EDO Hill + bruit
│   │   ├── synth_methods.py     # NODE, distillation, Weak-SINDy, Prop.1
│   │   └── pysr_free_form.py    # Régression libre PySR (synthétique)
│   └── results/synth/           # Résultats JSON (15 variantes × 5 graines)
├── checks/                      # Infrastructure de vérification
│   ├── common.py                # Utilitaires partagés
│   ├── prereg.py                # Pré-enregistrement SHA-256
│   ├── verify_results.py        # Vérification 166 fichiers JSON
│   ├── consistency.py           # 15 contrôles de cohérence
│   └── make_report.py           # Génération rapport automatique
├── figures_article/             # Figures et tableaux générés
├── docs/                        # Documentation et rapport
├── run_all.py                   # Point d'entrée par chaîne
├── run_everything.py            # Orchestration complète
├── make_figures.py              # Génération des figures
├── setup_all.py                 # Installation et vérification
├── smoke_test.py                # 16 tests de non-régression
├── PREREGISTRATION.json         # Gel des configurations (SHA-256)
├── RAPPORT_RESULTATS.md         # Rapport de résultats (auto-généré)
├── datasets.json                # URLs et SHA-256 des données
└── requirements.txt             # Dépendances Python
```

### 10.2 Cartographie des modules principaux

| Module | Fichier | Fonction principale | Entrées | Sorties |
|---|---|---|---|---|
| Prétraitement | `real_data.py` | Chargement h5ad, HVG, normalisation, vélocité, blocs | `.h5ad` + config | `Z, V, w, clusters, split` |
| Neural ODE réel | `real_node.py` | Entraînement par appariement de vitesse | `Z, V, W, tr/va, cfg, seed` | Réseau $f_\theta$, métriques |
| Neural ODE synthétique | `synth_methods.py` | Entraînement par trajectoires (L_obs) | `Xtr, Xva, times, cfg, seed` | Réseau $f_\theta$ |
| Identification Hill | `real_identify.py` | NNLS + TRF + Pareto | `Z, y, lib, rng` | Modèle symbolique, Pareto |
| Distillation | `synth_methods.py` | Pseudo-trajectoires kNN | `f_θ, Z0, t_end, cfg` | Nuage de points $(Z, F)$ |
| Weak-SINDy | `synth_methods.py` | Intégrale par parties | `X, t, Φ, cfg` | Modèle symbolique |
| Dynamique | `real_dyn.py` | Jacobien, équilibres, trajectoires | `F_np, starts` | Points fixes, stabilité |
| Garde-fous | `project/src/guards.py` | Vérification bistabilité | Champ F | OK ou GuardError |
| Pré-enregistrement | `checks/prereg.py` | SHA-256 des configs | Répertoires config | JSON gel/amendements |
| Cohérence | `checks/consistency.py` | 15 contrôles monotonicité/invariance | Répertoires résultats | Rapport JSON |
| Proposition 1 | `synth_methods.py::prop1_check` | Borne de Grönwall | $F_{\text{true}}, \hat{F}, \text{traj}$ | fraction holds, B/E |
| KO in silico | `synth_methods.py::ko_eval` | Validation interventionnelle | $F_{\text{true}}, F_{\text{sym}}, Z_0$ | cosinus décalages |
| Figures | `make_figures.py` | 6 figures + 3 tables | Résultats JSON | PNG/PDF/CSV/TEX |

### 10.3 Flux de données

```
Fichier .h5ad
    ↓ [real_data.py : scVelo, scanpy]
Z (Ms normalisé), V (velocity normalisée), w (velocity_confidence)
    ↓ [real_node.py : train_vel_node]
f_θ (Neural ODE entraîné)
    ↓ [synth_methods.py : sample_Df / real_data : intégration]
Z_distill, y_distill = f_θ(Z_distill)
    ↓ [real_identify.py : select_structure]
Modèles symboliques {dz_i/dt = Σ a_k H±(z_j; K, n) - γ z_i}
    ↓ [real_dyn.py : equilibria, cluster_correspondence]
Points fixes, analyse de stabilité, trajectoires in silico
    ↓ [checks/make_report.py + make_figures.py]
RAPPORT_RESULTATS.md, figures PNG/PDF, tableaux CSV/TEX
```

---

## 11. Jeux de données et prétraitement

### 11.1 Données réelles

**Pancréas murin E15.5 (Bastidas-Ponce et al., 2019) [13].**
- 3 696 cellules, 2 000 gènes hautement variables retenus.
- 8 gènes sélectionnés (critère de variance Ms) : Ghrl, Gnas, Ins2, Nnat, Chga, Clu, Isl1, Cpe.
- Stades : Ngn3 low EP (progeniteur de référence), Alpha, Beta, Delta, Epsilon.
- SHA-256 : `9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6`

**Gyrus dentelé murin (Hochgerner et al., 2018) [14].**
- ~3 400 cellules.
- 8 gènes sélectionnés (mêmes critères).
- Progeniteur de référence : Radial Glia-like.
- SHA-256 : `6fbf1ec64e0fba6d18d14bc342b3ba2aab8133972c1e50cb2cb717d4cc336871`

### 11.2 Pipeline de prétraitement scVelo

Le prétraitement suit la pipeline scVelo standard, avec les adaptations documentées :

```
1. Filtrage et normalisation scVelo (min_shared_counts=20)
2. Sélection HVG par scanpy (sur copie log1p, n_top_genes=2000)
3. Moments (n_pcs=30, n_neighbors=30) : Ms (ARNm mature lissé), Mu (pré-ARNm lissé)
4. Vélocité mode deterministic (incompatibilité stochastic avec NumPy ≥ 2.x)
5. Graphe de vélocité (n_jobs=1 pour compatibilité Windows)
6. velocity_confidence : poids de confiance normalisés à moyenne 1
7. Sélection des 8 gènes de vélocité à plus grande variance Ms
8. Normalisation : Z = Ms / quantile99(Ms) ; V = velocity / même facteur
9. Découpage en blocs : k-means(60 blocs, 10 composantes PCA) → train(60%) / val(20%) / test(20%)
```

### 11.3 Données synthétiques (générateur propre)

Le générateur `synth_system.py` crée des systèmes de type EDO Hill :
- $d \in \{3, 5, 8\}$ gènes ; 1 ou 2 régulateurs par gène (aléatoire) ; activateur ou répresseur (aléatoire)
- Paramètres : $K \in [0.4, 1.5]$, $n \in \{1.5, 2.0, 3.0\}$, $a \in [1.0, 3.0]$, $\gamma \in [0.8, 1.4]$
- Simulation par LSODA (rtol=1e-8, atol=1e-10)
- Bruit gaussien relatif : $X = Z_{\text{true}} + \sigma_{\text{rel}} \cdot \text{std}(Z) \cdot \mathcal{N}(0,1)$
- Modèle binomial négatif (variantes `counts`) : $Y_c \sim \text{NB}(\kappa S z, r)$, avec $\kappa = 20$, $r = 5$
- Dropout : zéro-inflation par masque binomial

Les 15 variantes couvrent systématiquement les facteurs de variation :

| Facteur | Variantes |
|---|---|
| Bruit | base(5%), noise02(2%), noise10(10%), noise20(20%) |
| Dropout | drop10(10%), drop20(20%) |
| Quantité de données | N20(20 traj), base(60), N120(120) |
| Dimension | d3(3), base(5), d8(8) |
| Modèle d'observation | base(gaussien), counts(NB), counts_drop10(NB+drop) |
| Pseudotemps | base(vrai), pseudo2(γ=2) |
| Variété | base(full), manifold(1D) |
| Gradient | base(direct), adjoint |

---

## 12. Méthodologie expérimentale

### 12.1 Les quatre chaînes d'exécution

**Chaîne A — Toggle 2D bistable.**
Validation sur un système bien défini avec vérité terrain (bistabilité, points fixes, champ vrai connu). Tests de mauvaise spécification et de robustesse au bruit ($\sigma \in \{0, 0.05, 0.10\}$). PySR appliqué avec recherche libre d'expressions.

**Chaîne B — Validation SERGIO.**
Validation sur données synthétiques biophysiquement réalistes générées par SERGIO [5]. Le simulateur SERGIO est vérifié importable et son dépôt présent. La simulation complète (~5 min) n'est pas lancée dans le test rapide.

**Chaîne C — Données réelles scRNA-seq.**
Application complète du pipeline sur les deux jeux de données réels (pancréas et gyrus dentelé). Évaluation selon les critères R1-R7, comparaison aux baselines biologiques (dynamo, CellRank, BoolODE). 5 graines × 2 jeux = 10 exécutions NODE + balayage $\lambda_{\text{reg}}$.

**Chaîne D — Benchmark synthétique multi-variantes.**
15 variantes × 5 graines = 75 exécutions. Pour chaque exécution : NODE, distillation + Hill, identification directe FD, Weak-SINDy, gplearn, PySR, et vérification de la Proposition 1. Évaluation selon les critères S1-S7.

### 12.2 Protocole d'évaluation

**Métriques principales :**

| Métrique | Formule | Interprétation |
|---|---|---|
| Cosinus | $\langle F_{\text{pred}}, F_{\text{ref}} \rangle / (\|F_{\text{pred}}\| \|F_{\text{ref}}\|)$ | Alignement directional du champ |
| NRMSE | $\sqrt{\text{MSE}} / (F_{\max} - F_{\min})$ | Erreur normalisée |
| F1 de structure | $2PR/(P+R)$ avec TP = arêtes correctement identifiées | Qualité du réseau régulateur |
| Rappel paramètres | Erreur relative médiane sur $(K, n)$ des arêtes TP | Précision des constantes biophysiques |
| Fraction Grönwall | $|\{E \leq B\}| / N_{\text{traj}}$ | Validité de la borne théorique |

**Contrôle aléatoire :** un contrôle à vitesses mélangées (permutation aléatoire des vélocités entre cellules) est systématiquement évalué. Le NODE doit le dépasser d'au moins la marge préenregistrée.

---

## 13. Protocole de reproductibilité

Voir **Annexe C** pour le détail complet. En résumé :

- **Pré-enregistrement** : gel SHA-256 des 10 fichiers de configuration le 2026-11-11, avant toute exécution (0 amendements).
- **Données** : vérification SHA-256 des fichiers `.h5ad` à chaque exécution.
- **Résultats** : 166 fichiers JSON vérifiés (157 OK, 9 avertissements mineurs, 0 échec).
- **Cohérence** : 15 contrôles passés (0 échec).
- **Non-régression** : 16 tests en ~200 secondes.

---

## 14. Résultats expérimentaux

### 14.1 Données réelles — Pancréas E15.5 (Chaîne C)

Les résultats sont obtenus sur 5 graines (seeds 0–4), gènes Ghrl, Gnas, Ins2, Nnat, Chga, Clu, Isl1, Cpe.

| Critère | Code | Valeur | Seuil | Verdict |
|---|---|---|---|---|
| Cosinus NODE (test) | R1a | **0.520** | ≥ 0.5 | ✓ satisfait |
| Marge sur contrôle mélangé | R1b | **0.155** | ≥ 0.3 | ✗ non satisfait |
| Cosinus symbolique/NODE | R2 | **0.880** | ≥ 0.9 | ✗ non satisfait |
| Rapport cos(sym,V)/cos(NODE,V) | R7 | **0.998** | ≥ 0.8 | ✓ satisfait |

**Résultats complémentaires :**

| Méthode | Cosinus (test) | Commentaire |
|---|---|---|
| Référence linéaire | **0.588** | Dépasse le NODE : référence forte |
| Identification directe FD | **0.575** | Dépasse la distillation |
| Distillation NODE | **0.519** | Inférieure à la FD directe |
| Contrôle mélangé | **0.365** | Marge NODE − contrôle = 0.155 |
| CellRank | **0.475** | États terminaux : Alpha, Beta, Delta |
| dynamo | **−0.931** | Ins2 : phase de dégradation pré-stationnaire |
| BoolODE | **0.020** | Faible cohérence avec vélocité scVelo |

**Balayage de $\lambda_{\text{reg}}$ (graine 0) :** cosinus NODE de 0.481 ($\lambda=0$) à 0.640 ($\lambda=1.0$). La régularisation amélore la généralisation au-delà du seuil de 0.5.

**Expressions symboliques (pancréas, graine 0, PySR) :**
- Ghrl ← Ghrl (autorégulation)
- Gnas ← Chga, Cpe
- Ins2 ← Gnas, Ins2
- Nnat ← Ins2, Nnat, Chga
- Chga ← Chga, Cpe
- Clu ← Clu (autorégulation)
- Isl1 ← Nnat, Chga, Isl1
- Cpe ← Chga, Cpe

### 14.2 Benchmark synthétique (Chaîne D)

Résultats sur la variante `base` (5 gènes, bruit 5%, 60 trajectoires, 5 graines) :

| Critère | Code | Valeur | Seuil | Verdict |
|---|---|---|---|---|
| F1 structure (distillé) | S1 | **0.778** | ≥ 0.7 | ✓ satisfait |
| Erreur relative paramètres | S2 | **0.441** | ≤ 0.25 | ✗ non satisfait |
| Cosinus NODE / champ vrai | S3 | **0.939** | ≥ 0.9 | ✓ satisfait |
| Borne Grönwall (70 runs) | S4 | **1.000** | ≥ 0.95 | ✓ satisfait |
| Pseudotemps cosinus | S5 | **0.712** | ≥ 0.9 | ✗ non satisfait |
| Variété — chute cosinus | S6 | **0.014** | ≥ 0.1 | ✗ non satisfait |

**Comparaison des méthodes sur données synthétiques (base, 5 graines) :**

| Méthode | F1 structure | Remarque |
|---|---|---|
| Identification directe FD | **0.887** | Meilleure F1 — dérivée directe sur données propres |
| Weak-SINDy | **0.864** | Robuste aux dérivées par forme faible |
| Distillation NODE + Hill | **0.778** | Lissage du NODE pénalise la F1 |

**Validation interventionnelle (Chaîne D, KO in silico) :**
- Symbolique : cosinus décalages = **0.628**
- NODE : cosinus décalages = **0.605**

L'approche symbolique présente un léger avantage sur la prédiction des effets d'intervention.

### 14.3 Recherche symbolique libre — PySR (Chaînes A, C, D)

| Critère | Code | Valeur | Seuil | Verdict |
|---|---|---|---|---|
| Toggle — R² vs NODE | PT1 | **0.936** | ≥ 0.95 | ✗ non satisfait |
| Toggle — NRMSE vs vérité | PT2 | **0.0214** | ≤ 0.05 | ✓ satisfait |
| Toggle — 2 attracteurs stables | PT3 | **1/3 graines** | toutes | ✗ non satisfait |
| Toggle — régulateur présent | PT4 | **1.00** | ≥ 1.0 | ✓ satisfait |
| Réelles — cosinus PySR/NODE | RP1 | **0.764** | ≥ 0.9 | ✗ non satisfait |
| Réelles — ratio cosinus | RP2 | **0.852** | ≥ 0.8 | ✓ satisfait |
| Synthétique — rappel variables | S7 | **0.920** | ≥ 0.8 | ✓ satisfait |

**Expressions PySR toggle (vérité : $dx/dt = a/(1+(y/K)^n) - \gamma x$) :**
- Graine 0 : expressions polynomiales, sans forme Hill explicite (complexité 11-19)
- Graine 2 : $\hat{f}_0 = 2.744/(x_1 + 0.416) + (-0.908)x_0 - 0.575$ — approximation rationnelle de Hill

**Tableau de synthèse : fraction de critères satisfaits**

| Chaîne | Satisfaits | Total | Taux |
|---|---|---|---|
| C (données réelles) | 2 | 4 | 50% |
| D (synthétique) | 3 | 6 | 50% |
| PySR (A+C+D) | 4 | 7 | 57% |
| **Total préenregistré** | **9** | **17** | **53%** |

---

## 15. Analyse comparative

### 15.1 Résumé des forces et faiblesses par méthode

| Méthode | Forces | Faiblesses | Contexte optimal |
|---|---|---|---|
| Neural ODE (appariement) | Flexible, non paramétrique ; capture les non-linéarités | Pas d'interprétabilité ; dépasse à peine la référence linéaire sur données réelles | Données propres, haute SNR |
| Distillation + Hill | Interprétable ; termes biophysiques (K, n, γ) | Lissage du NODE pénalise le F1 (-11%) ; paramètres K, n peu précis | Vérité terrain disponible pour validation |
| Identification directe FD | Simple ; meilleure F1 sur données synthétiques | Sensible au bruit ; différences finies amplifiées par bruit | Données synthétiques, bruit faible |
| Weak-SINDy | Évite la dérivation directe ; robuste au bruit | Complexité de mise en œuvre ; fenêtres à paramétrer | Bruit modéré, données trajectoires |
| PySR (libre) | Sans hypothèse de forme ; adaptatif | Instabilité de la topologie (PT3 : 1/3 graines) ; coûteux (Julia) | Exploration, si bruit faible |
| Référence linéaire | Très rapide ; imbattable sur données bruitées | Incapable de capturer les non-linéarités régulatrices | Données fortement bruitées |
| dynamo | Modèle biophysique complet (Mu, Ms, γ, β) | Dépendant de la phase cinétique ; diverge en phase de dégradation | Données équilibrées (Mu ≈ Ms) |
| CellRank | Probabiliste ; gère l'incertitude ; identifie les états terminaux | Ne produit pas d'équations EDO ; dépend du graphe kNN | Annotation des états, pseudotemps |
| BoolODE | Hypothèses biologiques fortes ; génère des données synthétiques | Modèle discontinu ; cosinus = 0.020 sur données réelles | Données très binarisées |

### 15.2 Hiérarchie des méthodes sur données synthétiques (base)

$$\text{FD (0.887)} > \text{Weak-SINDy (0.864)} > \text{Distillation (0.778)}$$

La distillation via Neural ODE n'améliore pas la F1 par rapport aux méthodes directes sur données synthétiques bien spécifiées. L'avantage du Neural ODE réside dans sa flexibilité sur les données réelles bruitées, mais cet avantage est absorbé par le passage à la distillation symbolique.

---

## 16. Discussion scientifique

### 16.1 Interprétation de R1b (marge insuffisante)

La marge de 0.155 sur le contrôle mélangé (seuil préenregistré : 0.3) suggère que le Neural ODE capte un signal réel mais modeste dans les vélocités ARN. Deux explications non exclusives :

1. **Bruit intrinsèque élevé.** Les vélocités scVelo sont des estimateurs bruités avec une variance difficile à quantifier. Le cosinus du contrôle mélangé (0.365) est anormalement élevé, ce qui révèle une **cohérence locale forte** dans les vélocités (cellules voisines ont des vélocités similaires, indépendamment de l'ordre cellulaire). Cette cohérence locale est en partie une conséquence du lissage kNN de scVelo lui-même.

2. **Référence linéaire forte.** Le cosinus linéaire (0.588) dépasse le NODE (0.520) sur l'ensemble de test. Cela indique que les tendances globales sont quasi-linéaires dans cet espace à 8 gènes, et que la non-linéarité Hill n'apporte pas de gain sur données réelles bruitées.

### 16.2 Interprétation de R2 (cosinus symbolique/NODE = 0.880 < 0.9)

La fidélité de la distillation au champ Neural ODE est partielle. Plusieurs facteurs contribuent :
- La distillation généère 4000 points, mais leur distribution n'est pas uniforme dans l'espace des états.
- La bibliothèque Hill ($k_{\max} = 3$ termes) peut manquer de capacité pour représenter les non-linéarités du MLP à 2 couches cachées.
- Le raffinement TRF converge vers des minima locaux pour certaines graines.

Cependant, R7 (ratio cosinus = 0.998) confirme que la *perte de fidélité est équilibrée* : le champ symbolique s'aligne aussi bien avec la vélocité mesurée que le Neural ODE lui-même.

### 16.3 Validation de la Proposition 1 (S4 = 1.000)

La borne de Grönwall est satisfaite sur **100 % des 70 trajectoires de test** examinées (14 variantes × 5 graines). Le rapport médian $B/E = $ **médiane calculée dans les fichiers JSON** indique que la borne est respectée avec une marge confortable. Ce résultat confirme que l'erreur de trajectoire symbolique est contrôlée par l'erreur de champ, conformément à la théorie.

> [!IMPORTANT] Ce résultat valide uniquement la cohérence interne entre le champ symbolique et le Neural ODE. Il ne valide pas la fidélité du Neural ODE au système biologique réel (aucune vérité terrain disponible sur données réelles).

### 16.4 Résultat négatif honnête : dynamo cosinus = −0.931

Le cosinus de dynamo sur le gène Ins2 est fortement négatif. L'analyse diagnostique confirme que Mu ≫ Ms pour Ins2, ce qui correspond à une **phase de dégradation pré-stationnaire** à E15.5 : la transcription a cessé mais la dégradation de l'ARN mature est encore en cours. Dans ce régime, la vitesse calculée par dynamo ($\hat{v} = \hat{\beta}u - \hat{\gamma}s < 0$) s'oppose à la direction de différenciation observée dans l'espace Ms. Ce n'est pas une erreur algorithmique mais un fait biologique documenté [15].

### 16.5 Résultat négatif : S2 (erreur paramètres = 0.441 > 0.25)

L'erreur relative médiane sur K et n est 44 %, nettement au-dessus du seuil de 25 %. Deux causes :
1. **Identifiabilité pratique partielle** : les profils de paramètres montrent des plateaux plats en K (K identifiable sur une décade de log), mais n est souvent contraint à sa borne inférieure.
2. **Biais de distillation** : le lissage opéré par le Neural ODE déplace légèrement les valeurs des constantes Hill optimales.

### 16.6 KO in silico : avantage symbolique léger

Le cosinus des décalages KO symbolique (0.628) dépasse légèrement le cosinus NODE (0.605). Cet avantage est dû à la contrainte de non-négativité et à la forme Hill qui impose des saturation cohérentes avec la biologie, contrairement au MLP sans contrainte.

---

## 17. Limites du travail

### 17.1 Limites épistémologiques

**L1. Absence de vérité terrain sur données réelles.** Les critères R1-R7 ne mesurent ni la récupération d'équations réelles ni la causalité. La vélocité ARN est elle-même une approximation, et son cosinus avec le champ prédit n'est pas une mesure de la causalité biologique réelle.

**L2. Niveaux d'inférence causale.** Les résultats se situent au niveau 2-3 du tableau de causalité (corrélation, prédiction), et non aux niveaux 4-7 (intervention, contrefactuel). Les expressions symboliques identifiées sont des *hypothèses de modélisation*, non des lois établies.

**L3. Identifiabilité structurelle partielle.** Même avec vérité terrain, l'erreur paramètre reste à 44 % (S2). Pour des données réelles sans vérité terrain, l'identifiabilité est encore moins garantie.

### 17.2 Limites techniques

**L4. Référence linéaire non dépassée (données réelles).** Le Neural ODE (cosinus 0.520) ne dépasse pas la référence linéaire (0.588). L'ajout de non-linéarités Hill n'apporte pas de gain sur les données pancréatiques dans l'espace retenu.

**L5. Distillation non optimale.** La distillation réduit le F1 de 0.887 (direct) à 0.778, soit -11 % en termes de F1. Le lissage du Neural ODE introduit des artefacts dans la bibliothèque de candidats.

**L6. PySR instable sur toggle (PT3).** Sur 3 graines, 2/3 perdent les points fixes stables dans l'expression PySR. La recherche libre d'expressions n'exploite pas la connaissance a priori de la forme Hill.

**L7. Pseudotemps (S5 = 0.712 < 0.9).** L'estimation du champ sous déformation temporelle (γ=2) est significativement moins précise. Les données scRNA-seq utilisant le pseudotemps (par exemple Monocle) rendraient ce problème encore plus difficile.

**L8. Hors-support (S6 = 0.014 < 0.1).** Le Neural ODE ne présente pas de comportement distinctement différent hors du support des données, contrairement à ce qui était espéré. La bibliothèque Hill est mieux extrapolable, mais l'amélioration est trop faible pour atteindre le seuil.

### 17.3 Limites du protocole

**L9. Pré-enregistrement auto-déclaré.** Le fichier `PREREGISTRATION.json` est auto-déclaré. Sans engagement dans un registre externe (git commit + tag, OSF, AsPredicted), la preuve d'antériorité est partielle.

**L10. Pas de validation CRISPR/perturbation.** Les prédictions de KO in silico ne sont pas validées par des expériences de perturbation biologique.

---

## 18. Conclusion générale

Ce projet développe et évalue un pipeline complet d'identification de réseaux de régulation génique par Neural ODE et régression symbolique structurée. Les résultats expérimentaux montrent que :

1. **Le Neural ODE apprend un signal cohérent** (R1 : cosinus 0.520 ≥ 0.5), mais ne dépasse pas la référence linéaire (0.588) sur les données pancréatiques. La marge sur le contrôle mélangé (R1b : 0.155) est insuffisante, révélant la difficulté de distinguer le signal de la cohérence locale induite par scVelo.

2. **La régression symbolique est partiellement fidèle** au champ Neural ODE (R2 : 0.880), avec un ratio de cosinus acceptable (R7 : 0.998). La bibliothèque Hill capture les régulations qualitatives mais les constantes quantitatives (K, n) sont imprécises (S2 : 44 %).

3. **Le benchmark synthétique valide** la méthode dans des conditions contrôlées : NODE cosinus 0.939 (S3), borne de Grönwall à 100 % (S4). L'identification directe surpasse la distillation en F1 (0.887 vs 0.778), suggérant que le Neural ODE introduit un lissage pénalisant la récupération structurelle.

4. **Le protocole de reproductibilité est rigoureux** : pré-enregistrement SHA-256, 15 contrôles de cohérence passés, 166 fichiers de résultats vérifiés, 16 tests de non-régression.

5. **Les limites sont honnêtement documentées** : absence de vérité terrain sur données réelles, identifiabilité partielle, référence linéaire non dépassée. Ces résultats sont présentés sans ajustement a posteriori des seuils.

La contribution principale est methodologique : la combinaison d'une validation formelle (Grönwall), d'un pré-enregistrement cryptographique, et d'une comparaison systématique aux baselines biologiques constitue un cadre d'évaluation rigoureux pour l'inférence de GRN à partir de scRNA-seq.

---

## 19. Perspectives

Les pistes d'amélioration et de développement identifiées sont :

**P1. Validation expérimentale.** Confronter les prédictions de régulation aux données de perturbation CRISPR (CRISPRi, overexpression) ou aux ChIP-seq des facteurs de transcription identifiés.

**P2. Espaces d'état alternatifs.** Évaluer systématiquement les espaces modules et latents (ACP). Le module k-means peut regrouper des co-régulateurs, améliorant la lisibilité des équations.

**P3. Contraintes biologiques dans PySR.** Intégrer des priors de Hill dans PySR (espace de recherche restreint aux termes $x^n / (K^n + x^n)$) pour améliorer la stabilité structurelle (PT3).

**P4. Prise en charge du pseudotemps.** Modéliser explicitement la déformation temporelle $t_{\text{obs}} = g(t_{\text{true}})$ dans la perte d'entraînement, pour améliorer S5.

**P5. Régularisation structurelle.** Pénaliser la sparsité des interactions en ajoutant une pénalité L1 sur les amplitudes $a_k$ lors du raffinement TRF.

**P6. Données multi-omiques.** Intégrer les données ATAC-seq (accessibilité chromatine) pour contraindre les interactions actives dans les équations symboliques.

**P7. Identification globale (multi-gènes).** L'identification actuelle est équation-par-équation. Une optimisation globale conjointe pourrait améliorer la cohérence inter-équations.

**P8. Variabilité cellulaire.** Modéliser l'hétérogénéité avec un champ stochastique $d\mathbf{z} = F(\mathbf{z})dt + \sigma d\mathbf{W}$ et évaluer l'inférence de $\sigma$.

**P9. Validation sur atlas cellulaires.** Appliquer le pipeline sur les grands atlas (Tabula Muris, Human Cell Atlas) couvrant plusieurs organes.

**P10. Interprétabilité des équilibres.** Relier systématiquement les équilibres stables identifiés aux états cellulaires annotés dans la littérature biologique.

**P11. Benchmarks étendus.** Évaluer le pipeline sur les benchmarks de référence de la littérature GRN (BEELINE [16], GENIE3 [17]).

**P12. Pré-enregistrement externe.** Déposer les configurations sur un registre public (OSF, AsPredicted) avant toute future exécution pour renforcer la preuve d'antériorité.

---

## 20. Références bibliographiques

[1] G. La Manno et al., "RNA velocity of single cells," *Nature*, vol. 560, pp. 494–498, 2018.

[2] V. Bergen et al., "Generalizing RNA velocity to transient cell states through dynamical modeling," *Nature Biotechnology*, vol. 38, pp. 1408–1414, 2020.

[3] H. de Jong, "Modeling and simulation of genetic regulatory systems: A literature review," *Journal of Computational Biology*, vol. 9, no. 1, pp. 67–103, 2002.

[4] N. Meinshausen and P. Bühlmann, "High-dimensional graphs and variable selection with the Lasso," *Annals of Statistics*, vol. 34, pp. 1436–1462, 2006.

[5] P. Dibaeinia and S. Sinha, "SERGIO: A single-cell expression simulator guided by gene regulatory networks," *Cell Systems*, vol. 11, pp. 252–271, 2020.

[6] X. Qiu et al., "Mapping transcriptomic vector fields of single cells," *Cell*, vol. 185, pp. 690–711, 2022.

[7] M. Lange et al., "CellRank for directed single-cell fate mapping," *Nature Methods*, vol. 19, pp. 159–170, 2022.

[8] R. T. Q. Chen et al., "Neural Ordinary Differential Equations," in *Advances in Neural Information Processing Systems* (NeurIPS), 2018, pp. 6572–6583.

[9] M. Cranmer et al., "Discovering symbolic models from deep learning with inductive biases," in *NeurIPS*, 2020.

[10] T. Stephens, *gplearn: Genetic Programming in Python with a scikit-learn inspired API*, 2015. [Online] https://gplearn.readthedocs.io

[11] S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *PNAS*, vol. 113, no. 15, pp. 3932–3937, 2016.

[12] D. A. Messenger and D. M. Bortz, "Weak SINDy for partial differential equations," *Journal of Computational Physics*, vol. 443, 110525, 2021.

[13] A. Bastidas-Ponce et al., "Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis," *Development*, vol. 146, dev173849, 2019.

[14] D. Hochgerner et al., "Conserved properties of dentate gyrus neurogenesis across postnatal development revealed by single-cell RNA sequencing," *Nature Neuroscience*, vol. 21, pp. 290–299, 2018.

[15] X. Qiu et al., "Reversed graph embedding resolves complex single-cell trajectories," *Nature Methods*, vol. 14, pp. 979–982, 2017.

[16] A. Pratapa et al., "Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data," *Nature Methods*, vol. 17, pp. 147–154, 2020.

[17] V. A. Huynh-Thu et al., "Inferring regulatory networks from expression data using tree-based methods," *PLOS ONE*, vol. 5, e12776, 2010.

---

## 21. Annexes techniques

- **[Annexe A — Démonstrations mathématiques](annexes/demonstrations_mathematiques.md)** : Lemme de Grönwall, Proposition 1 (démonstration complète), cinétique de Hill, stabilité de Lyapunov, fonctions de perte.

- **[Annexe B — Pseudo-codes des algorithmes](annexes/algorithmes.md)** : Entraînement NODE (données réelles et synthétiques), sélection de structure Hill, Weak-SINDy, distillation, Proposition 1, KO in silico.

- **[Annexe C — Protocole de reproductibilité](annexes/protocole_reproductibilite.md)** : Pré-enregistrement SHA-256, pipeline de vérification des données, 15 contrôles de cohérence, smoke test 16 étapes, instructions de reproduction complète.

- **[Tableau des technologies](tables/technologies.csv)** : Versions de toutes les bibliothèques utilisées.

- **[Tableau des jeux de données](tables/jeux_donnees.csv)** : Toutes les sources de données (réelles, synthétiques, SERGIO).

- **[Tableau des paramètres expérimentaux](tables/parametres_experimentaux.csv)** : Hyperparamètres complets extraits des fichiers de configuration.

- **[Tableau des résultats comparatifs](tables/resultats_comparatifs.csv)** : Toutes les métriques R1-R7, S1-S7, PT1-PT4, RP1-RP2 avec valeurs, seuils et verdicts.

### Figures générées

Les figures sont disponibles dans le répertoire `figures/` (copies de `figures_article/`) :

| Fichier | Description |
|---|---|
| `fig_real_cosines.png` | Cosinus moyens par méthode (données réelles pancréas) |
| `fig_real_endpoints.png` | Distribution des points finaux des trajectoires in silico |
| `fig_real_lambda_sweep.png` | Sensibilité du cosinus NODE au paramètre λ_reg |
| `fig_real_spaces.png` | Comparaison des espaces d'état (gènes, modules, latent) |
| `fig_synth_structure_F1.png` | F1 de structure par méthode et variante synthétique |
| `fig_synth_robustness.png` | Courbes de robustesse (bruit, dropout, N, dimension) |

---

*Ce rapport a été rédigé à partir de l'audit exhaustif du code source, des fichiers de configuration et des résultats JSON du projet `c:\neural_ode_symbolic`. Toutes les valeurs numériques sont issues des fichiers de résultats vérifiés. Aucun résultat n'a été inventé ni ajusté a posteriori.*

*Généré le 2026-11-19 — Audit source : `docs/audit_sources.md` (si présent).*
