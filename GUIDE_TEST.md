# Guide de test sur votre machine (Windows, 4 coeurs)

Objectif : vérifier que tout fonctionne **avant** de lancer l'exécution complète, puis lancer celle-ci et lire les résultats sans se tromper. Chaque étape dit quoi taper, ce qui est attendu et quoi faire en cas d'échec.
Les commandes `py -3` et `.venv\Scripts\python.exe` sont pour Windows (cmd ou PowerShell) ; sous Linux/macOS : `python3` et `.venv/bin/python`.

## Étape 0 — Prérequis (5 min)

- Python **3.12** (python.org, cocher « Add python.exe to PATH »), Git for Windows (`winget install Git.Git`), ~8 Go d'espace libre, 8 Go de RAM recommandés.
- Dossier du projet à un **chemin court**, par exemple `C:\neural_ode_symbolic` (limite de 260 caractères de Windows).
- Ouvrir un terminal dans ce dossier. Pour forcer l'UTF-8 : `set PYTHONUTF8=1` (cmd) ou `$env:PYTHONUTF8=1` (PowerShell) ; `LANCER_WINDOWS.bat` le fait seul.

## Étape 1 — Installation et téléchargements (10 à 25 min selon le réseau)

```
py -3 setup_all.py
```
Télécharge : PyTorch CPU et les dépendances, les deux jeux de données scVelo (77 Mo, SHA-256 vérifié), SERGIO au commit figé (~200 Mo) avec son correctif.
**Attendu** : tout en `[OK]`, puis `TOUT EST PRET.` et la ligne de commande à utiliser ensuite. Rapport : `setup_report.json`.
**Si échec** : voir le dépannage en bas de page, puis relancer la même commande (elle reprend où elle s'est arrêtée).

## Étape 2 — Test de fonctionnement (5 à 10 min)

```
.venv\Scripts\python.exe smoke_test.py
```
Lance chaque chaîne en mode rapide (valeurs **non interprétables**) et vérifie que les garde-fous **refusent** des entrées fautives (système monostable, JSON vide ou `_from_log`, résultat qui s'améliore avec le bruit).
**Attendu** : `13/13 tests réussis` (11 si vous avez choisi `--skip-pysr`). Le test du toggle compare le résultat rapide à la référence : sur Linux de rédaction, l'écart relatif était de l'ordre de 1e-9 ; **sur votre machine, un écart jusqu'à ~1e-3 est plausible** (la tolérance du test est 5e-3).
**Si un test échoue** : ne pas continuer ; envoyer `smoke_report.json` et la sortie du terminal.

## Étape 3 — Geler le préenregistrement (1 min, **avant** les calculs)

```
.venv\Scripts\python.exe checks\prereg.py freeze
git init & git add PREREGISTRATION.json & git commit -m "gel des seuils" & git tag prereg
```
Les seuils des critères sont dans `project\configs`, `real_data\configs` et `synthetic\configs`. Le gel enregistre leur SHA-256 ; le commit git est l'ancrage horodaté. Si vous devez modifier une configuration après coup : `checks\prereg.py amend "motif"` (amendement daté et motivé).

## Étape 4 — Exécution complète (estimation : 1 h 30 à 2 h sur 4 coeurs, PySR inclus, non mesurée)

```
.venv\Scripts\python.exe run_everything.py --workers 4
```
170 tâches, reprise automatique si interrompu (relancer la même commande). Journaux : `logs\<tâche>.log`. Variantes :
- `--workers 3` si la RAM est limitée (la simulation SERGIO demande ~2 Go) ; `--sergio-parallel 1` pour la mémoire.
- `--only real` (ou `toggle`, `sergio`, `synthetic`) pour une seule chaîne ; `--dry-run` pour lister les tâches.
- `--strict` : un contrôle de cohérence non évaluable compte comme un échec.
En fin d'exécution, le lanceur enchaîne les analyses, les figures et les **contrôles** (intégrité, cohérence, rapport). Message final attendu : `Termine en … ; echecs : aucun`.

## Étape 5 — Lire les résultats (et savoir quand ne pas les utiliser)

| Fichier | Contenu |
|:--|:--|
| `RAPPORT_RESULTATS.md` | Rapport **généré à partir des chiffres** : provenance des données, verdicts R1–R7 et S1–S6 recalculés, limites |
| `verification_report.json` | Intégrité : JSON lisibles, pas de `_from_log`, pas de NaN, empreinte de configuration |
| `consistency_report.json` | Cohérence : le bruit ne doit pas améliorer les résultats, pas de métrique constante, le contrôle doit discriminer |
| `figures_article\` | Figures (PNG, PDF), tableaux (CSV, LaTeX), coût de calcul |
| `logs\analyses.txt` | **Toute la sortie texte des analyses et des contrôles** (tableaux des chaînes A et B, verdicts, contrôles) |
| `logs\<tâche>.log` | Journal de chaque calcul (une graine ou une configuration) |
| `run_everything_log.json`, `setup_report.json`, `smoke_report.json` | Durées et états du lancement, de l'installation et du test de fonctionnement |
| `project\results\phase1\` | Chaîne A (toggle) : `phase1_*`, `misspec_*`, `detect_*` (ignorer `phase1_quick_seed0.json`, issu du test de fonctionnement) |
| `sergio_validation\result_*.json` | Chaîne B (SERGIO) |
| `real_data\results\real\` | Chaîne C (données réelles) : un JSON par graine, `verdicts*.json` |
| `synthetic\results\synth\` | Chaîne D (benchmark synthétique) : un JSON par variante et graine, `verdicts_synth.json` |
| `real_data\results\real\verdicts*.json`, `synthetic\results\synth\verdicts_synth.json` | Verdicts par chaîne |

**Règle** : si le lanceur affiche `CONTROLES EN ECHEC`, ou si `verify_results.py` ou `consistency.py` renvoie un échec, **n'utilisez pas les résultats** avant d'en comprendre la cause. Un contrôle « NON EVALUE » signifie qu'il manque des graines : ce n'est pas un échec (sauf avec `--strict`).
Les verdicts négatifs (critères non satisfaits) sont des **résultats**, pas des erreurs : le rapport les affiche tels quels.

## Étape 6 — Comparer à la référence

```
.venv\Scripts\python.exe checks\compare_reference.py reference_results\toggle project\results\phase1 --tol 1e-3
.venv\Scripts\python.exe checks\verify_results.py
```
Chaîne A : attendu `identique` ou `numerique` (écart ≤ 1e-3). Les chaînes C et D ne sont pas comparables fichier à fichier avec `reference_results\real_session_claude*` et `synthetic_pilot*` (pilote partiel, versions différentes) ; comparez les **verdicts** (`RAPPORT_RESULTATS.md`).

## Étape 7 — Tests optionnels

- Sensibilité de l'entraînement (~3 min par unité) : `cd project & ..\.venv\Scripts\python.exe scripts\sensitivity_probe.py --systems C2 M1`
- Étiquettes de nocivité (sans entraînement) : `cd project & ..\.venv\Scripts\python.exe scripts\oracle_labels.py`
- PySR et Julia font partie du projet complet : voir **`GUIDE_PYSR.md`**. `setup_all.py` les installe (plusieurs minutes et ~2-3 Go la première fois) ; `--no-pysr` pour les éviter.

## Ce qu'il faut m'envoyer en cas de problème

`setup_report.json`, `smoke_report.json`, le dernier `logs\<tâche>.log` en échec, `run_everything_log.json`, `verification_report.json`, `consistency_report.json`, et la sortie du terminal (copier-coller).

## Dépannage

| Symptôme | Cause probable | Action |
|:--|:--|:--|
| `git requis` | Git absent | `winget install Git.Git`, rouvrir le terminal |
| Échec du clone SERGIO ou du correctif | Chemin trop long ; conversion CRLF | Dossier court (`C:\neural_ode_symbolic`) ; supprimer `sergio_validation\SERGIO` et relancer `setup_all.py` |
| Échec de torch CPU | Index PyTorch inaccessible | `py -3 setup_all.py --cuda-torch` (plus lourd) |
| `[GARDE-FOU] ... SHA-256 ... differente` | Fichier de données modifié ou corrompu | Supprimer le fichier dans `real_data\data` et relancer `setup_all.py` |
| `[GARDE-FOU] ... regime non bistable` | Une configuration a été modifiée | Annuler la modification ; sinon `prereg.py amend` |
| `Une configuration differe du gel` | Configuration modifiée après `freeze` | Idem ; ou `--allow-config-change` (à documenter) |
| `MemoryError` / tâche tuée pendant SERGIO | RAM insuffisante | `--workers 2 --sergio-parallel 1` |
| Erreur scVelo sur `stochastic` | numpy ≥ 2 ; mode fixé à `deterministic` dans les configurations | Ne pas changer ce mode |
| Résultats « différents » de la session de rédaction | Nombre de threads, versions de bibliothèques | Comparer les verdicts ; écarts numériques ≤ 1e-3 normaux pour la chaîne A |
