# Rapport de résultats (généré automatiquement)

Ce document est produit par `checks/make_report.py` à partir des fichiers de résultats ; chaque verdict est recalculé depuis les valeurs brutes et comparé à celui des scripts d'analyse. Aucune phrase n'est écrite à la main.

## Provenance

| Jeu de données | Fichier | SHA-256 conforme | Empreinte attendue |
|:--|:--|:--|:--|
| pancreas | présent | oui | `9e3e459eca00ba06…` |
| dentategyrus | présent | oui | `6fbf1ec64e0fba6d…` |

- **using_real_data : true** ; la chaîne C vérifie le SHA-256 du fichier h5ad avant de calculer et n'a aucun repli sur des données simulées (une erreur interrompt l'exécution).
- Empreintes de configuration des résultats réels : `a606d8c808d3a02b`
- Gel du préenregistrement : présent (2026-11-11T20:24:05Z)
- Python 3.14.2, Windows 11

## Données réelles (chaîne C)

5 graines ; gènes de l'espace d'état : Ghrl, Gnas, Ins2, Nnat, Chga, Clu, Isl1, Cpe.

| Critère | Valeur | Seuil | Verdict |
|:--|:--|:--|:--|
| R1 cosinus NODE (test) | 0.520 | ≥ 0.5 | satisfait |
| R1 marge sur le contrôle à vitesses mélangées (0.365) | 0.155 | ≥ 0.3 | **NON satisfait** |
| R2 cosinus symbolique/NODE | 0.880 | ≥ 0.9 | **NON satisfait** |
| R7 rapport cos(sym,V)/cos(NODE,V) | 0.998 | ≥ 0.8 | satisfait |

Référence linéaire : cosinus 0.588 — le Neural ODE ne dépasse pas cette référence (écart -0.068). Identification directe : 0.575 contre 0.519 pour la distillation (cosinus avec la vitesse mesurée).
Balayage de λ_reg (graine 0) : cosinus NODE de 0.481 à 0.640 selon λ_reg.

Aucune vérité terrain : ces critères ne mesurent ni la récupération d'équations ni la causalité.

## Benchmark synthétique (chaîne D)

Variantes disponibles : N120 (5 graine(s)), N20 (5 graine(s)), adjoint (5 graine(s)), base (5 graine(s)), counts_drop10 (5 graine(s)), counts (5 graine(s)), d3 (5 graine(s)), d8 (5 graine(s)), drop10 (5 graine(s)), drop20 (5 graine(s)), manifold (5 graine(s)), noise02 (5 graine(s)), noise10 (5 graine(s)), noise20 (5 graine(s)), pseudo2 (5 graine(s)).

| Critère | Valeur | Seuil | Verdict |
|:--|:--|:--|:--|
| S1 F1 de structure (distillé, base, 5 graines) | 0.778 | ≥ 0.7 | satisfait |
| S2 erreur relative médiane des paramètres | 0.441 | ≤ 0.25 | **NON satisfait** |
| S3 cosinus NODE/champ vrai | 0.939 | ≥ 0.9 | satisfait |
| S4 proposition 1 : borne tenue (70 runs) | 1.000 | ≥ 0.95 | satisfait |
| S5 pseudotemps : cosinus de direction (5 graine(s)) | 0.712 | ≥ 0.9 | **NON satisfait** |
| S6 variété : chute de cosinus hors support (5 graine(s)) | 0.014 | ≥ 0.1 | **NON satisfait** |

F1 de structure sur la variante de base : distillation 0.778, identification directe 0.887, forme faible 0.864 — la distillation ne dépasse pas les deux bases de référence.
Interventions in silico (délétion d'un gène) : cosinus des décalages prédits/vrais 0.628 (symbolique) contre 0.605 (Neural ODE).

## Recherche symbolique libre (PySR)

Recherche d'expressions sans famille imposée (opérateurs + − × ÷) sur le champ du Neural ODE ; seuils lus dans `project/configs/pysr.json`, `real_data/configs/real_pysr.json`, `synthetic/configs/synth.json`. Chaque seuil a été fixé après un essai pilote d'une graine (divulgué dans les fichiers de configuration).

| Critère | Valeur | Seuil | Verdict |
|:--|:--|:--|:--|
| PT1 toggle : fidélité R² au Neural ODE (3 graines) | 0.936 | ≥ 0.95 | **NON satisfait** |
| PT2 toggle : NRMSE de champ contre la vérité | 0.0214 | ≤ 0.05 | satisfait |
| PT3 toggle : 2 points fixes stables conservés dans chaque graine | 1/3 graines | toutes | **NON satisfait** |
| PT4 toggle : le gène régulateur apparaît dans l'équation | 1.00 | ≥ 1.0 | satisfait |
| RP1 données réelles : cosinus PySR/NODE (3 graines) | 0.764 | ≥ 0.9 | **NON satisfait** |
| RP2 données réelles : cosinus (PySR, vitesse) / cosinus (NODE, vitesse) | 0.852 | ≥ 0.8 | satisfait |
| S7 synthétique : rappel moyen des variables vraies (5 runs) | 0.920 | ≥ 0.8 | satisfait |

Expressions trouvées sur le toggle (une ligne par graine et équation ; vérité : dx/dt = a/(1+(y/K)^n) − g·x) :

- graine 0 : éq. 0 (complexité 11) : `((x1 * x0) - 1.1347333) * ((x1 * 0.10667803) - 0.6722486)` ; éq. 1 (complexité 19) : `(((x1 - -1.4594864) * ((x0 - -1.160297) * -0.28224394)) + 2.1078637) * (((x0 - -0.9331503) * -0.34177265) + 2.1078637)`
- graine 1 : éq. 0 (complexité 17) : `((((x1 * 0.17949091) * ((x0 * x1) - 1.8254955)) - (x0 * x1)) * 0.76227784) + 1.0863893` ; éq. 1 (complexité 15) : `(((x0 - 3.157891) * ((x0 * 0.4022738) + -1.1285005)) + 0.27275082) - (x1 * 0.8686954)`
- graine 2 : éq. 0 (complexité 11) : `((2.7440832 / (x1 + 0.4158853)) + (x0 * -0.9076368)) + -0.57479084` ; éq. 1 (complexité 13) : `3.7830539 - ((x0 * ((x0 * -0.40335006) - -2.3945103)) - (x1 * -0.8558735))`

Gènes utilisés par chaque équation (données réelles, graine 0) : Ghrl ← Ghrl ; Gnas ← Chga, Cpe ; Ins2 ← Gnas, Ins2 ; Nnat ← Ins2, Nnat, Chga ; Chga ← Chga, Cpe ; Clu ← Clu ; Isl1 ← Nnat, Chga, Isl1 ; Cpe ← Chga, Cpe.

Les expressions sont des hypothèses de modélisation (non uniques, VI-A) ; PySR ne fournit ni mécanisme ni causalité.

## Limites

- Chaînes A et B (toggle 2D, SERGIO) : leurs verdicts sont produits par leurs propres scripts d'analyse (`analyze_results.py`, `analyze_sergio.py`) ; ce rapport ne les recalcule pas.
- dynamo, CellRank, BoolODE non exécutés ; aucune perturbation expérimentale.
- Les verdicts ci-dessus portent sur les seuils préenregistrés ; toute variante ajoutée après coup est exploratoire.
