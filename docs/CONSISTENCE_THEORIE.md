# Matrice de cohérence : partie théorique (v4) ↔ projet complet (chaînes A, B, C, D)

Chaînes : **A** toggle 2D (`project/`) ; **B** SERGIO (`sergio_validation/`) ; **C** données réelles pancréas (`real_data/`) ; **D** benchmark synthétique multi-gènes avec vérité terrain (`synthetic/`, ajouté).
**La cohérence n'est pas totale** : les écarts restants sont en bas, avec l'amendement proposé pour le texte. Numéros d'équations : `Partie_theorique_IEEE_v4.docx`.

| Théorie | Implémentation | Statut |
|:--|:--|:--|
| II-A notation $\mathbf{z}$, $\pi$, $\rho$ | C : $\pi$ = identité (gènes), moyenne par module (modules), ACP linéaire (latent) ; D : $\mathbf{z}$ = gènes | Conforme |
| II-B temps, vitesse (éq. 1) | C : vitesse scVelo à l'état stationnaire, sans temps ; D : temps vrai **ou** pseudotemps $\tau=T(t/T)^\gamma$ (monotone, partagé) | Conforme ; C voir écart 3 |
| II-C espaces gènes / modules / latent | C : les trois (modules = k-means de 40 gènes ; latent = ACP) ; latent : bibliothèque polynomiale (pas de Hill, II-E) | **Conforme (nouveau)** |
| II-D équilibres, jacobien | `real_dyn.py` (C) ; jacobien et constante de Lipschitz dans D | Conforme |
| II-E Hill, identifiabilité des constantes | A, B, C, D ; D mesure l'erreur de $(a,K,n)$ pour les termes retrouvés, avec une base « structure oracle » | Conforme |
| III-A, III-B Neural ODE, éq. 7 ($\mathcal{L}_{\mathrm{obs}}$), éq. 8 ($\mathcal{L}_{\mathrm{vel}}$), éq. 10 | **D : $\mathcal{L}_{\mathrm{obs}}$ sur trajectoires, solveur rk4 ; C : $\mathcal{L}_{\mathrm{vel}}$ seul** | Conforme en D ; écart 1 en C |
| III-C sensibilité à $\lambda_{\mathrm{reg}}$ | C : balayage ; D : $\lambda_{\mathrm{reg}}$ fixe | Partiel |
| IV-A bibliothèque, structure puis constantes | A, B, C, D : sélection avant NNLS puis raffinement `trf` multi-départs | Conforme |
| IV-B recherche libre | **D : gplearn (testé) ; PySR : module `pysr_free_form.py` (non testé ici, option `--pysr`)** | Partiel (écart 2) |
| IV-C distillation / direct / **forme faible** | **D : distillation, direct (différences finies), forme faible implémentée (Messenger & Bortz), oracle** ; C : distillation, direct, STLSQ | **Conforme (nouveau)** |
| V-B découpage par groupes | C : blocs ; D : trajectoires entières | Conforme (C : approximation signalée) |
| V-C $\mathcal{D}_f$ (éq. 12) | C et D : cellules / trajectoires reconstruites + voisinages bruités, rejet | Conforme |
| V-D évaluation, **proposition 1** | **D : borne de Grönwall vérifiée trajectoire par trajectoire** ($\varepsilon$, $L$, $B$, $E$) | **Conforme (nouveau)** |
| VI-A remarque 1 (variété) | D : variante `manifold` (conditions initiales sur une courbe) ; cosinus hors support et F1 | Testé ; voir résultats |
| VI-A remarque 2 (pseudotemps) | D : variante `pseudo2` ; rapport de vitesse et cosinus de direction | Testé |
| VII-B dropout, bruit, $N$, $d$, couverture | D : 12 variantes × 5 graines | Conforme |
| VII-C données réelles | C : un seul jeu (pancréas E15.5, un temps) ; **pas de second jeu, pas de perturbation** | Écart 4 |
| VII-D lignes de base | NODE, linéaire, direct, STLSQ, forme faible (D), oracle (D) ; **dynamo, CellRank, BoolODE** (implémentés en mode standalone, sans installation des paquets originaux) | **Conforme (nouveau)** |
| Seuils préenregistrés | C : `real_pancreas*.json` ; D : `synth.json` (SHA-256 dans chaque résultat) | Conforme |
| Figures et tableaux | `make_figures.py` → `figures_article/` (PNG, PDF, CSV, LaTeX) | **Nouveau** |

## Écarts restants et amendements proposés

1. **C : pas de $\mathcal{L}_{\mathrm{obs}}$ sans temps** (D l'implémente). Amendement III-B : « $\mathcal{L}_{\mathrm{obs}}$ n'est présent que si des temps $t_i$ ou $\tau_i$ sont disponibles ; sinon seul $\mathcal{L}_{\mathrm{vel}}$ contraint le champ (unités de temps relatives). »
2. **PySR non exécuté ici** ; gplearn sert de recherche libre testée. Amendement IV-B : citer gplearn pour l'étude actuelle, PySR comme extension.
3. **Vitesse en mode stationnaire** (`stochastic` en échec avec numpy 2.x). Amendement II-B/VII-C.
4. **Un seul jeu réel, sans perturbation** : niveaux 5 à 7 hors de portée ; l'audit demandait un second système. Non réalisable dans l'environnement de rédaction (domaines de données bloqués).
5. **dynamo, CellRank, BoolODE** : implémentés en mode analytique/standalone (`baseline_dynamo.py`, `baseline_cellrank.py`, `baseline_boolode.py`) ; `run_real_baselines.py` produit `baselines_<tag>.json` ; intégré dans `analyze_real.py` et `run_everything.py`. Note : dynamo-analytique (OLS direct Ms/Mu, sans graphe de voisinage) diverge de scVelo déterministe sur Ins2 (Mu >> Ms) — résultat valide documenté dans Bergen et al. 2020.
6. **D utilise des trajectoires observées à plusieurs temps par condition initiale** (suivi), pas des instantanés destructifs. Amendement VII-B : le préciser.

## Mise à jour : garde-fous du chapitre « Implémentation » désormais implémentés

| Garde-fou (chapitre, section 10) | Implémentation | Vérifié par |
|:--|:--|:--|
| Régime dynamique vérifié avant l'entraînement | `project/src/guards.py` : `assert_bistable` appelé dans `run_phase1`, `run_phase6`, `run_misspec`, `run_detect_test` ; arrêt avant entraînement | `smoke_test.py` (test négatif sur un système monostable) ; les 11 systèmes configurés passent |
| Pas de repli silencieux, provenance | Résultats écrits par les scripts ; `checks/verify_results.py` refuse `_from_log`, fichiers vides ; la chaîne C vérifie le SHA-256 du jeu avant calcul | `smoke_test.py` (test négatif) |
| Contrôles de cohérence | `checks/consistency.py` : le bruit/dropout ne doit pas améliorer les résultats, aucune métrique constante, le contrôle doit discriminer | `smoke_test.py` (test négatif : cosinus qui s'améliore avec le bruit) |
| Conclusions dérivées des chiffres | `checks/make_report.py` : verdicts recalculés depuis les valeurs brutes et comparés à ceux des scripts d'analyse ; arrêt sur contradiction | Exécuté sur les résultats de référence |
| Intégrité et comparaison | `checks/verify_results.py`, `checks/compare_reference.py` | `smoke_test.py` |
| Préenregistrement | `checks/prereg.py` (freeze / amend / verify) ; `run_everything.py` refuse de partir si une configuration diffère du gel | Testé (gel automatique) ; **ancrage externe (git) à faire par l'utilisateur** |
| `run_all.py toggle --quick` | Compare à la référence et renvoie un code d'erreur | Écart relatif 3e-15 / 9e-10 dans l'environnement de rédaction |
| Sonde de sensibilité, étiquettes de nocivité | `project/scripts/sensitivity_probe.py`, `oracle_labels.py` ; option `perturb_rel` dans `train_node` (0 par défaut : aucun effet sur les résultats) | Exécutés sur 1 unité (sonde : écart relatif 1,2e-2 sur le champ pour C2) et 3 systèmes (étiquettes) |

Autres ajouts : second jeu réel (gyrus denté, `real_dentategyrus.json`), vraisemblance binomiale négative (`obs_model: nb`), méthode adjointe (`node.adjoint`), profils de tolérance des constantes de Hill, interventions in silico (délétion d'un gène), durées par étape (`checks/analyze_cost.py`), `smoke_test.py`, `GUIDE_TEST.md`.

**Toujours absents** : exécutions PySR complètes (40 itérations), jeu avec perturbations expérimentales, test sous Windows (planifié). Le chapitre `Chapitre_Implementation_et_reproductibilite_corrige.docx` (section 10) décrit encore plusieurs garde-fous comme « non implémentés » : à mettre à jour après validation sur votre machine.

## Mise à jour : PySR

| Élément | État |
|:--|:--|
| Recherche libre (IV-B) | **Implémentée et testée en mode rapide** : `run_pysr_toggle.py` (chaîne A), `run_real_pysr.py` (chaîne C), `--free-form pysr` (chaîne D) ; PySR 2.7.2 + Julia 1.13.3 sous Linux, `smoke_test.py --with-pysr` : 13/13 |
| Non fait | Exécutions complètes (40 itérations), Windows, installation automatique de Julia, parallélisme réel sur 4 coeurs |

Voir `GUIDE_PYSR.md`.

**PySR est maintenant dans le flux par défaut** (installation, test de fonctionnement, lanceur, rapport) ; seuils PT1–PT4 (toggle), RP1–RP2 (données réelles) et S7 (synthétique), fixés après un essai pilote d'une graine, divulgué dans les configurations. Le texte de la partie théorique (IV-B) peut décrire PySR comme méthode utilisée, avec les restrictions suivantes : un essai pilote seulement a été exécuté ici ; les exécutions complètes sont à faire sur votre machine.
