# Commandes essentielles

```
py -3 setup_all.py                                       # installation + PySR/Julia + jeux de donnees + SERGIO (une fois)
.venv\Scripts\python.exe smoke_test.py                    # test de fonctionnement (11 tests, 5-10 min)
.venv\Scripts\python.exe checks\prereg.py freeze         # gel des seuils AVANT les calculs
.venv\Scripts\python.exe run_everything.py --workers 4    # tout : calculs, analyses, figures, controles, rapport
```
Guide complet, lecture des résultats et dépannage : **`GUIDE_TEST.md`**. PySR et Julia (pourquoi, installation, exécution) : **`GUIDE_PYSR.md`**. Linux/macOS : `python3` et `.venv/bin/python`.

---

# Contenu complété (chaîne D, espaces d'état, figures)

- **`synthetic/`** : benchmark multi-gènes avec vérité terrain (12 variantes x 5 graines : bruit, dropout, N, d, pseudotemps, variété ; bases : identification directe, **forme faible**, structure oracle, recherche libre **gplearn**) ; vérifie la **proposition 1** trajectoire par trajectoire. Seuils S1-S6 dans `synthetic/configs/synth.json`.
- **Données réelles** : espaces `gene`, `module`, `latent` (`real_data/configs/real_pancreas*.json`).
- **PySR (optionnel, non testé ici)** : `python setup_all.py --with-pysr` puis `run_everything.py --pysr`.
- **Figures et tableaux** : `make_figures.py` (appelé en fin de `run_everything.py`) -> `figures_article/` (PNG, PDF, CSV, LaTeX).
- Temps total estimé sur 4 coeurs : 1 h 15 à 1 h 45 (138 tâches), non mesuré d'un bout à l'autre.
- Résultats du pilote de la session : `reference_results/` (`synthetic_pilot/`, `real_session_claude_spaces/`) et `docs/RESULTATS_SYNTHETIQUE_PILOTE.md`.
- Ce qui manque encore : second jeu réel / perturbations, dynamo, CellRank, BoolODE, un test sous Windows (voir `docs/CONSISTENCE_THEORIE.md`).

---

# Windows + CPU 4 coeurs : démarrage

**Prérequis** : Python 3.12 (python.org, cocher « Add python.exe to PATH »), Git for Windows (`winget install Git.Git`), ~8 Go libres, 8 Go de RAM recommandés.
**Installer le dossier à un chemin court** (ex. `C:\neural_ode_symbolic`) : la limite de 260 caractères de Windows peut casser le jeu de données SERGIO.

```bat
LANCER_WINDOWS.bat          :: double-clic ; équivaut aux deux commandes ci-dessous
py -3 setup_all.py          :: venv + torch CPU + dépendances + jeu pancréas (SHA-256) + SERGIO (commit figé, sans conversion CRLF) + vérification
.venv\Scripts\python.exe run_everything.py --workers 4
```

Linux/macOS : `python setup_all.py` puis `.venv/bin/python run_everything.py`.

Ce qui a été adapté :
- **4 coeurs** : `run_everything.py` lance 70 tâches indépendantes (une graine ou une configuration par processus, 1 coeur chacun, threads BLAS/torch/numba fixés à 1), jusqu'à `--workers` en parallèle ; la simulation SERGIO est limitée à 2 tâches simultanées (`--sergio-parallel`, mémoire). Reprise automatique, journaux dans `logs/`.
- **Windows** : aucun multiprocessing interne (scVelo `n_jobs=1` ; sous Windows, `spawn` relancerait les scripts), UTF-8 forcé (`PYTHONUTF8=1`) pour la console et les fichiers, clone SERGIO avec `core.autocrlf=false` (sinon le correctif ne s'applique pas), suppression robuste des fichiers `.git` en lecture seule.
- Options utiles : `--dry-run` (liste des tâches), `--only real`, `--filter "real_seed"`, `--workers 3` si la RAM est limitée, `python setup_all.py --check`.

**Non testé sous Windows** (développé et testé sous Linux, 1 CPU) : création du venv, index PyTorch CPU, `.bat`, `git` sous Windows, PySR/Julia. Si une étape échoue, `setup_report.json` et `logs/<tâche>.log` indiquent laquelle. Durée : estimation de 40 à 60 min sur 4 coeurs (≈ 3 h de calcul cumulé), non mesurée d'un bout à l'autre ; les valeurs peuvent différer à la 3e–4e décimale des résultats de référence (nombre de threads, versions).
Résultats de référence de la session : `reference_results/` (`real_session_claude/` pour les données réelles). Cohérence avec la partie théorique : `docs/CONSISTENCE_THEORIE.md`.

---

# Exécuter le projet en local

Deux chaînes indépendantes, un lanceur unique (`run_all.py`) et des scripts d'analyse qui reproduisent les tables du rapport.

```
neural_ode_symbolic/
├── run_all.py                 # lanceur multi-plateformes : toggle | sergio | analyze
├── requirements.txt
├── project/                   # chaîne A : toggle switch (Neural ODE -> distillation -> identification)
│   ├── configs/*.json         # configurations ET seuils préenregistrés (ne pas modifier : voir « Préenregistrement »)
│   ├── src/                   # toggle.py, node.py, identify.py, families.py, misspec.py, misspec2.py
│   └── scripts/               # run_phase1.py, run_phase6.py, run_misspec.py, run_detect_test.py, analyze_results.py
├── sergio_validation/         # chaîne B : validation sur données SERGIO (MLP + régression symbolique)
│   └── run_sim.py, validate.py, identifiability.py, steady_state_diagnostic.py,
│       eiv_synthetic.py, eiv_test_sergio.py, analyze_sergio.py, sergio_numpy_compat.patch
└── reference_results/         # résultats obtenus pendant la session (pour comparaison)
```

## 1. Installation

Testé avec **Python 3.12**. Aucun GPU requis (PyTorch CPU suffit).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch CPU (plus léger) si besoin :  pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Ressources : ~1 Go de disque pour la chaîne A ; la chaîne B télécharge un **clone partiel de SERGIO (~200 Mo)** (le dépôt complet pèse ~2,5 Go : ne pas le cloner en entier) et la simulation consomme **~2 Go de RAM**. Les entraînements Neural ODE utilisent 1 thread ; vous pouvez lancer plusieurs commandes en parallèle sur une machine multi-cœurs.

## 2. Test de bon fonctionnement (≈ 2 minutes)

```bash
python run_all.py toggle --quick
```
Exécute la Phase 1, graine 0. **Valeurs de référence** (doivent être retrouvées à 4 chiffres sur la même pile logicielle) : NRMSE du champ du Neural ODE = 0,012407 ; NRMSE de trajectoire symbolique = 0,007985 ; erreur relative maximale des paramètres = 0,090344 ; structure exacte = True.

## 3. Chaîne A — toggle switch (≈ 1 h sur 1 CPU)

`python run_all.py toggle` enchaîne les commandes suivantes (chacune se lance aussi seule depuis `project/`). Les sorties vont dans `project/results/phase1/` ; une étape dont le résultat existe est sautée (`--force` pour recalculer).

| Ordre | Commande (depuis `project/`) | Durée | Ce qu'elle fait | Configuration |
|---|---|---|---|---|
| 1 | `python scripts/run_phase1.py --seeds 0 1 2 --tag traj` | ~5 min | Phase 1 : Neural ODE → distillation → identification, protocole préenregistré (trajectoire seule) | `phase1.json` |
| 2 | `python scripts/run_phase1.py --seeds 0 1 2 --velocity 1.0 --tag vel` | ~5 min | Ablation : terme de vitesse (vitesses **exactes**) | `phase1.json` |
| 3 | `python scripts/run_phase6.py --seeds 0 1 2 --tag n05 --noise 0.05 --ntrain 40` | ~5 min | Grille bruit σ = 0,05 | `phase6_grid.json` |
| 4 | `... --tag n10 --noise 0.10 --ntrain 40` | ~5 min | Grille bruit σ = 0,10 | idem |
| 5 | `... --tag N15 --noise 0.0 --ntrain 15` | ~5 min | Grille : 15 trajectoires | idem |
| 6 | `python scripts/run_misspec.py` | ~11 min | Mauvaise spécification (hill / leaky / gauss) | `misspec.json` |
| 7 | `python scripts/run_detect_test.py` | ~26 min | Test préenregistré des règles de détection (7 systèmes) | `detect_test.json` |

Tables d'analyse (verdicts sur les critères préenregistrés) :
```bash
python run_all.py analyze                      # toutes les tables
cd project && python scripts/analyze_results.py {phase1|phase6|misspec|detect}
# comparer à la référence :  ... --dir ../reference_results/toggle
```

## 4. Chaîne B — validation SERGIO (≈ 20 min, ~2 Go de RAM)

```bash
python run_all.py sergio
```
Clone partiel de SERGIO, application du correctif de compatibilité numpy (`sergio_numpy_compat.patch`, 22 lignes : alias `np.int`/`np.float` supprimés dans numpy ≥ 1.24), puis dans l'ordre : `run_sim.py` (~5 min) → `validate.py clean | noisy_mild | noisy_severe | control` → `identifiability.py` → `steady_state_diagnostic.py` → `eiv_synthetic.py` → `eiv_test_sergio.py` (~3 min). Sorties dans `sergio_validation/`. Tables : `python sergio_validation/analyze_sergio.py` (ou `--dir ../reference_results/sergio`).

Les fichiers `sergio_clean.npz` doivent être **identiques bit à bit** à ceux de la session (même graine, même SERGIO) : SHA-256 de `sergio_clean.npz` = `833674ef2c876e07…`. Un écart indique une pile numpy/scipy différente, pas forcément une erreur.

## 5. Résultats de référence (à retrouver)

| Étape | Résultat attendu |
|---|---|
| Phase 1, critères P1, P2, P3a, P4, P5 | satisfaits ; **P3b échoue pour la graine 2** (erreur max 0,110 > 0,10) → H1 non satisfaite au sens strict |
| Phase 1, variante vitesse | erreur max 0,043 en moyenne (hors protocole) |
| Phase 6 (R1–R3) | satisfaits dans les 4 conditions ; erreur max < 0,10 pour 2/3, 2/3, 1/3, 0/3 graines (réf., N=15, σ=0,05, σ=0,10) |
| Mauvaise spécification | D1 ✘, S1 ✘ (BIC choisit « leaky » sur le contrôle), S2 ✔ |
| Règles de détection | DT1 ✔ (1/9), DT2 ✘ (3/10), SG1 ✘ (15/18), SG2 ✔ (12/12) → échec global |
| SERGIO, données propres | type correct 14/14 ; répressions n̂ = 1,86 ; activations n̂ = 1,21 (7 premières cibles) mais 1,79 sur les 32 activations |
| SERGIO, contrôle négatif | R² médian 0,89 (vrai régulateur) contre 0,39 (non lié) |
| Test EIV sur SERGIO | hypothèse non confirmée (gain de R² vrai < contrôle) |

**Précaution de reproductibilité.** Les entraînements sont déterministes sur la pile utilisée (rejeu identique bit à bit). Sur une autre machine, des différences numériques (BLAS, version de PyTorch) peuvent modifier les derniers chiffres. Comme le verdict P3b (11,0 % contre 10 %) et d'autres critères sont proches du seuil, **comparez les ordres de grandeur et non le verdict exact**. Les seuils eux-mêmes ne doivent pas être modifiés après coup.

## 6. Préenregistrement

Les fichiers de `project/configs/` contiennent les seuils fixés **avant** exécution. Empreintes SHA-256 (16 premiers caractères) à conserver si vous voulez prouver que vous n'avez pas ajusté les critères :

```
737d58006004a1ff…  detect_test.json
6ec38a0463718864…  misspec.json
7c8f7058a9cbf830…  phase1.json
d642b2db1d0c5c37…  phase6_grid.json
```
Si vous modifiez un seuil, enregistrez-le comme une **nouvelle** analyse (copier la configuration sous un autre nom) et notez que c'est post hoc.

## 7. Ce qui n'est PAS inclus ni testé

- **PySR** : non exécuté (Julia a saturé le disque du bac à sable). La « découverte » dans la chaîne A repose sur une **forme imposée** (un terme de Hill + dégradation) ; le STLSQ générique échoue. À tenter sur Colab ou en local (`pip install pysr`, Julia installé automatiquement au premier import).
- **BoolODE** : code lu mais jamais exécuté (dépendances de 2019). Ses ODE ont un coefficient de Hill par défaut de 10 : la grille de départ (n ≤ 4) est à étendre avant tout test.
- **Données réelles** (pancréas, sci-FATE2, scPerturb), **Phase 2** (SERGIO avec Neural ODE), dropout, dimension > 2 : non réalisés.
- **Figures** : les figures du rapport ont été produites par des extraits non sauvegardés ; les tables (équivalentes) sont reproduites par les scripts d'analyse.
- Les règles de détection exigent des **trajectoires de test** (temps connu) : non transposable telle quelle au scRNA-seq.

## 8. Dépannage

| Symptôme | Cause probable / remède |
|---|---|
| `No space left on device` pendant le clone de SERGIO | Clone complet tenté : utiliser `python run_all.py sergio` (clone partiel) |
| Un calcul s'arrête sans message | Manque de RAM (SERGIO, ~2 Go) ou de disque ; relancer : les étapes terminées sont sautées. Vérifier qu'un JSON de résultat n'a pas 0 octet (fichier corrompu : le supprimer et relancer) |
| `AttributeError: module 'numpy' has no attribute 'int'` | Correctif SERGIO non appliqué : `cd sergio_validation/SERGIO && git apply ../sergio_numpy_compat.patch` |
| Résultats légèrement différents | Voir « Précaution de reproductibilité » |
| Windows | Les scripts Python sont portables ; `run_all.py` ne dépend pas de bash. WSL2 recommandé pour SERGIO |
