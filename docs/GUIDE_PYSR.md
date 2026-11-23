# Guide PySR et Julia : pourquoi, installation, codes, exécution

> **PySR fait partie du projet complet** : `setup_all.py` l'installe par défaut (option `--no-pysr` pour l'éviter), `smoke_test.py` le teste (`--skip-pysr` pour l'éviter), `run_everything.py` lance ses tâches (`--no-pysr` pour les éviter) et `RAPPORT_RESULTATS.md` contient une section PySR avec des seuils (PT1–PT4, RP1–RP2, S7) lus dans `project/configs/pysr.json`, `real_data/configs/real_pysr.json` et `synthetic/configs/synth.json`.

## 1. Pourquoi PySR et Julia dans ce projet

**Le besoin scientifique.** La partie théorique (section IV-B) décrit une *régression symbolique libre* : chercher des expressions mathématiques sans imposer leur forme. Les chaînes A, B et C du projet font autre chose : elles choisissent, dans une **famille imposée** (un terme de Hill d'un régulateur plus une dégradation linéaire), le meilleur régulateur, le signe et les constantes sur une grille. Elles ne peuvent donc retrouver que ce qui est déjà dans la famille. PySR cherche dans l'espace de toutes les expressions construites avec les opérateurs que vous autorisez (+, −, ×, ÷, éventuellement ^) et rend un **front de Pareto** précision/complexité par équation : c'est le complément « libre » de la distillation décrite en IV-C (le Neural ODE fournit le champ, PySR cherche une expression qui l'approche). Ce principe de distillation d'un réseau de neurones en équation est celui de Cranmer *et al.* (arXiv 2006.11287), cité par la documentation de PySR.

**Pourquoi Julia.** PySR est une interface Python ; son moteur de recherche est la bibliothèque Julia **SymbolicRegression.jl** (documentation officielle : « SymbolicRegression.jl forms the powerful search engine of PySR » ; à l'appel de `fit`, un processus Julia mène « a multithreaded search »). Julia est un langage compilé et multithreadé ; c'est ce qui rend la recherche évolutive rapide. Python appelle Julia via les paquets `juliacall` et `juliapkg` : **vous n'écrivez pas de code Julia**, mais Julia doit être installé et ses paquets téléchargés (au premier import de PySR, d'après la documentation).

**Ce que PySR ne fait pas.** La recherche est stochastique : le résultat dépend de la graine, des opérateurs et du budget (nombre d'itérations). Il ne connaît pas les contraintes biologiques (une fonction de Hill avec exposant non entier est difficile à exprimer sans l'opérateur `^`). Une expression qui approche bien le champ n'est ni unique (section VI-A) ni un mécanisme (VI-C). Les résultats sont à lire comme des **hypothèses**.

## 2. Installation

**Versions testées dans l'environnement de rédaction (Linux, 1 coeur)** : PySR 2.7.2, juliacall 0.9.36, juliapkg 0.1.28, Julia 1.13.3 (installé depuis conda-forge). **Non testé** : l'installation automatique de Julia par `juliapkg` (réseau bloqué chez moi) et tout sous Windows.

**Place disque mesurée** : 723 Mo de paquets Julia (dossier des dépôts Julia) ; l'environnement conda-forge de Julia faisait 1,4 Go (l'installateur officiel de Julia est probablement plus léger : non mesuré). Prévoir 2 à 3 Go libres.

### Voie recommandée (une commande)
```
py -3 setup_all.py
```
Fait `pip install pysr`, déclenche l'installation de Julia et de ses paquets au premier import (internet requis), puis **lance une mini-recherche pour prouver que ça marche** (`PySR fonctionne : …`). Durée : plusieurs minutes ; dans mon test, la précompilation des paquets Julia a demandé environ 12 à 14 minutes en trois reprises sur un seul coeur. Vous pouvez interrompre et relancer la même commande : l'installation reprend.

### Voie manuelle
```
pip install pysr              (ou : conda install -c conda-forge pysr)
python -c "import pysr"       (le premier import installe Julia et ses paquets)
```

### Si l'installation automatique de Julia échoue (pare-feu, réseau d'entreprise)
1. Installer Julia vous-même : https://julialang.org/downloads (installateur Windows ou `juliaup`), **ou** via conda : `conda create -n jl -c conda-forge julia` (voie que j'ai testée sous Linux).
2. Indiquer à Python où le trouver, **avant** de lancer Python :
   - cmd : `set PYTHON_JULIAPKG_EXE=C:\chemin\vers\julia.exe`
   - PowerShell : `$env:PYTHON_JULIAPKG_EXE="C:\chemin\vers\julia.exe"`
   - bash : `export PYTHON_JULIAPKG_EXE=/chemin/vers/julia`
3. Relancer `py -3 setup_all.py`.

### Variables d'environnement utiles
| Variable | Rôle |
|:--|:--|
| `PYTHON_JULIAPKG_EXE` | Chemin d'un Julia déjà installé |
| `PYTHON_JULIACALL_THREADS` | Nombre de threads Julia (la documentation précise que `JULIA_NUM_THREADS` n'est pas la bonne variable avec juliacall) ; à fixer **avant** d'importer pysr. Les scripts du projet les laissent à 1 : le parallélisme vient du lanceur, une tâche par coeur |
| `JULIA_DEPOT_PATH` | Dossier où Julia stocke ses paquets (par défaut dans votre profil utilisateur) |

## 3. Les codes qui utilisent PySR

| Script | Ce qu'il fait | Sorties |
|:--|:--|:--|
| `project/scripts/run_pysr_toggle.py` | Chaîne A : mêmes données et même Neural ODE que la Phase 1 ; PySR sur le champ du Neural ODE aux états d'entraînement ; erreur de champ vs vérité, trajectoires, points fixes | `project/results/phase1/pysr_toggle_seed<s>.json` |
| `synthetic/scripts/run_synth.py --free-form pysr --tag synthpysr` | Chaîne D : PySR sur le champ distillé d'un système synthétique ; rappel/précision des variables utilisées par rapport aux vraies | `synthetic/results/synth/synthpysr_base_seed<s>.json` |
| `real_data/scripts/run_real_pysr.py` | Chaîne C : PySR sur le champ neuronal des données réelles (une recherche par gène) ; cosinus avec le champ neuronal et la vitesse mesurée, gènes utilisés | `real_data/results/real/real_pysr_seed<s>.json` |
| `checks/analyze_pysr.py` | Tableau de synthèse | `figures_article/tab_pysr.csv` |

Réglages communs : opérateurs binaires `+ − × ÷` (option `--ops pow` pour ajouter `^` au toggle), `maxsize` (complexité maximale), `niterations` (budget), recherche **déterministe et en série** (`parallelism='serial'`, `deterministic=True`, graine fixée), pas de fichier temporaire. L'API utilisée (paramètres `parallelism`, `deterministic`, `random_state`, `progress`, `verbosity`, `model_selection`) a été vérifiée dans le code source de PySR 2.7.2 et a fonctionné ; avec une version plus ancienne, mettre à jour : `pip install -U pysr`.

## 4. Exécuter, dans l'ordre

```
:: 1. installer et vérifier
py -3 setup_all.py

:: 2. test de fonctionnement avec PySR (13 tests, PySR inclus ; les valeurs --quick ne sont pas interpretables)
.venv\Scripts\python.exe smoke_test.py

:: 3. un essai reel, une graine, pour mesurer la duree
cd project
..\.venv\Scripts\python.exe scripts\run_pysr_toggle.py --seeds 0 --iterations 40 --maxsize 20
cd ..

:: 4. tout, en parallele sur 4 coeurs (calculs PySR inclus)
.venv\Scripts\python.exe run_everything.py --workers 4
```
- Le lanceur crée d'abord une tâche `pysr_warmup` (un seul import de PySR) pour éviter que plusieurs tâches installent Julia en même temps. Les tâches PySR : toggle × 3 graines, synthétique × 5 graines, pancréas × 3 graines (8 recherches chacune) et gyrus denté × 1 graine.
- Pour ne lancer que les tâches PySR d'une chaîne : `run_everything.py --only toggle --filter "pysr"`. Pour tout lancer **sans** PySR : `run_everything.py --no-pysr`.
- Pour les données réelles, limiter d'abord : `cd real_data & ..\.venv\Scripts\python.exe scripts\run_real_pysr.py --seeds 0 --equations 0 --iterations 40`.
- **Durées mesurées dans mon environnement (Linux, 1 coeur, recherche en série)** : toggle, une graine, 40 itérations : 112 s au total (dont l'entraînement du Neural ODE, ~90 s) ; données réelles, 2 équations, 40 itérations : 19 s de recherche PySR (27 s au total avec l'entraînement), soit ~80 s pour les 8 équations. Ces mesures portent sur une graine ; elles ne garantissent pas la durée sur votre machine. Augmenter `--iterations` améliore la recherche mais allonge le temps.

## 5. Lire les résultats

- **Toggle** (`pysr_toggle_seed*.json`) : `equations[i].expression` (l'expression trouvée), `complexity`, `pareto` (le front complet), `uses_other_gene` (le régulateur de l'autre gène apparaît-il ?), `field_nrmse_vs_truth` à comparer à `field_nrmse_node_vs_truth` (une distillation ne peut pas faire mieux que le Neural ODE qu'elle approche, sauf par compensation d'erreurs), `fidelity_r2_vs_node`, `stable_fixed_points` (la bistabilité est-elle conservée ?).
- **Réel** : `pysr_vs_node_cos_test` (fidélité au champ neuronal), `pysr_vs_measured_cos_test` à comparer à `node_vs_measured_cos_same_genes` ; `used_genes` (quels gènes entrent dans chaque équation).
- **Synthétique** : `var_recall` et `var_precision` (variables vraies utilisées) ; PySR ne rend pas de structure Hill identifiable, donc seule l'utilisation des bonnes variables est mesurée.
- **Bonnes pratiques** : comparer plusieurs graines (une expression qui change d'une graine à l'autre est une information de stabilité), regarder le front de Pareto et pas seulement l'expression « best », et rappeler que l'absence de vérité terrain sur données réelles interdit toute conclusion de mécanisme.

## 6. Dépannage

| Symptôme | Cause probable | Action |
|:--|:--|:--|
| `PySR indisponible` au lancement d'un script | pysr non installé dans le venv utilisé | `py -3 setup_all.py` ; vérifier que le script est lancé avec `.venv\Scripts\python.exe` |
| Premier import très long | Installation et précompilation des paquets Julia | Attendre ; relancer en cas d'interruption (reprise automatique) |
| Échec d'installation de Julia | Réseau filtré | Voie manuelle de la section 2 (`PYTHON_JULIAPKG_EXE`) |
| Plantage à l'import, message `GLIBCXX_… not found` (Linux) | Bibliothèque `libstdc++` chargée par une autre dépendance | Voir la section « Troubleshooting » du README officiel de PySR (ajuster `LD_LIBRARY_PATH` vers les bibliothèques de Julia) |
| Erreur sur un paramètre `parallelism` | Version de PySR plus ancienne | `pip install -U pysr` |
| Plusieurs tâches qui installent Julia en même temps | Lancement parallèle sans préchauffage | Utiliser `run_everything.py --pysr` (tâche `pysr_warmup`) |
| Antivirus ou pare-feu bloque Julia | Binaires téléchargés | Autoriser le dossier des dépôts Julia et `julia.exe` |
| Résultats différents d'une machine à l'autre | PySR/Julia versions, flottants 32 bits (précision par défaut) | Noter les versions (`pip show pysr juliacall`) ; la graine et le mode série rendent les rejeux d'une même machine reproductibles |
| Machine hors ligne | Installation impossible | Installer sur une machine connectée, copier le dossier des dépôts Julia et définir `JULIA_DEPOT_PATH` (non testé) |

## 7. Honnêteté sur ce qui est testé

**Testé (Linux, 1 coeur, PySR 2.7.2 + Julia 1.13.3 de conda-forge)** : import et mini-recherche, les trois scripts en mode `--quick`, la synthèse des résultats, `smoke_test.py` (13/13).
**Non testé** : Windows ; l'installation automatique de Julia par `juliapkg` ; les exécutions complètes sur toutes les graines ; le parallélisme réel sur 4 coeurs ; le coût mémoire de plusieurs recherches simultanées.

**Blocage à la sortie** : lors d'un essai, un script a fini son calcul puis est resté bloqué ~270 s à l'arrêt du processus (non reproduit). Les trois scripts PySR se terminent maintenant par une sortie forcée après avoir écrit leurs résultats.

**Résultats du pilote (une graine, à ne pas généraliser)** : voir `docs/RESULTATS_PYSR_PILOTE.md`.
