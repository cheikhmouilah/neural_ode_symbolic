# Annexe C — Protocole de reproductibilité

## C.1 Vue d'ensemble

Ce projet applique un protocole de reproductibilité structuré en cinq couches :

| Couche | Mécanisme | Fichier |
|---|---|---|
| 1. Pré-enregistrement | SHA-256 des configurations avant exécution | `PREREGISTRATION.json` |
| 2. Intégrité des données | SHA-256 des jeux scRNA-seq | `datasets.json` |
| 3. Cohérence des résultats | 15 contrôles automatiques | `checks/consistency.py` |
| 4. Vérification des fichiers | Lecture et recalcul des verdicts | `checks/verify_results.py` |
| 5. Test de non-régression | 16 tests de bout en bout | `smoke_test.py` |

---

## C.2 Pré-enregistrement (checks/prereg.py)

### C.2.1 Principe

Avant toute exécution produisant des résultats, les seuils de décision sont figés par empreinte cryptographique :

```bash
python checks/prereg.py freeze   # crée PREREGISTRATION.json (refus si déjà présent)
python checks/prereg.py verify   # compare l'état actuel au gel (code 1 si divergence)
python checks/prereg.py amend "motif motivé"  # enregistre un amendement daté
```

### C.2.2 Contenu de PREREGISTRATION.json

```json
{
  "frozen_utc": "2026-11-11T20:24:05Z",
  "git_head": "...",
  "configs": {
    "real_data/configs/real_pancreas.json": "<sha256>",
    "real_data/configs/real_dentategyrus.json": "<sha256>",
    "synthetic/configs/synth.json": "<sha256>",
    ...
  },
  "amendments": []
}
```

### C.2.3 Résultat de la vérification (2026-11-19)

- Gel du : 2026-11-11T20:24:05Z
- Amendements : 0
- Configurations conformes au gel ✓
- Fichier nouveau signalé (non bloquant) : `real_data/configs/real_pancreas2222.json`

---

## C.3 Intégrité des jeux de données (datasets.json)

Les deux jeux de données scRNA-seq réels sont vérifiés par SHA-256 avant tout calcul :

| Jeu | Fichier | Taille | SHA-256 (préfixe) |
|---|---|---|---|
| Pancréas E15.5 | `endocrinogenesis_day15.h5ad` | 50 MB | `9e3e459e` |
| Gyrus dentelé | `dentategyrus_10X43_1.h5ad` | 24 MB | `6fbf1ec6` |

La vérification est effectuée par `setup_all.py --check` et par le script de la chaîne C au démarrage. Toute divergence interrompt l'exécution (pas de repli sur données simulées).

---

## C.4 Contrôles de cohérence (checks/consistency.py)

Le script exécute 15 contrôles automatiques qui détectent les pathologies suivantes :

### C.4.1 Contrôles de monotonicité (synthétique)

| Contrôle | Variants | Tolérance | Description |
|---|---|---|---|
| Bruit → cosinus NODE | noise02/base/noise10/noise20 | 0.03 | Le cosinus ne doit pas s'améliorer quand le bruit augmente |
| Bruit → F1 distillé | noise02/base/noise10/noise20 | 0.15 | Idem pour le F1 de structure |
| Dropout → cosinus NODE | base/drop10/drop20 | 0.03 | Idem pour le dropout |
| Dropout → F1 distillé | base/drop10/drop20 | 0.15 | Idem pour le F1 |

### C.4.2 Contrôle de quantité de données

N120 − N20 ≥ −0.03 sur le cosinus NODE : augmenter les données par un facteur 6 ne doit pas dégrader le résultat au-delà de 0.03.

### C.4.3 Contrôle d'invariance (détection de valeurs constantes)

6 métriques vérifiées : cosinus NODE, NRMSE trajectoire, F1 distillé, cosinus distillé, F1 direct FD, F1 Weak-SINDy.
Un écart < 10⁻¹² sur l'ensemble des configurations signale une valeur de repli.

### C.4.4 Contrôles sur données réelles

| Contrôle | Description |
|---|---|
| Invariance entre graines | L'étendue du cosinus NODE entre graines doit être > 10⁻¹² |
| Balayage λ_reg | L'étendue sur 5 valeurs de λ_reg doit être > 10⁻¹² |
| Contrôle à vitesses mélangées | cosinus principal > cosinus contrôle |

### C.4.5 Contrôle sur le toggle 2D (chaîne A)

Le bruit ne doit pas réduire l'erreur de champ de plus de 0.005 entre σ=0, 0.05, 0.10.

### C.4.6 Résultat (2026-11-19)

**15/15 contrôles passés** — 0 échec, 0 non-évalué.

---

## C.5 Vérification des fichiers résultats (checks/verify_results.py)

| Statistique | Valeur |
|---|---|
| Fichiers lus | 166 |
| Fichiers OK | 157 |
| Avertissements (NaN dans colonnes inutilisées STLSQ) | 9 |
| Échecs | 0 |

Les 9 avertissements concernent des colonnes non utilisées dans les baselines STLSQ ; ils n'affectent pas les métriques principales.

---

## C.6 Test de non-régression (smoke_test.py)

Le script `smoke_test.py` exécute 16 tests couvrant les 4 chaînes d'exécution :

| Test | Chaîne | Durée | Résultat |
|---|---|---|---|
| Garde-fou bistabilité | A | 1 s | PASS |
| verify_results | Vérification | 0 s | PASS |
| Cohérence | D | 0 s | PASS |
| Pré-enregistrement | Méta | 0 s | PASS |
| Comparaison référence toggle | A | 0 s | PASS |
| Chaîne A toggle --quick | A | 0 s | PASS |
| Chaîne D synthétique base seed0 --quick | D | 11 s | PASS |
| Chaîne C pancréas seed0 --quick | C | 31 s | PASS |
| Chaîne C gyrus dentelé seed0 --quick | C | 23 s | PASS |
| Chaîne C baselines VII-D (dynamo/CellRank/BoolODE) | C | 15 s | PASS |
| Chaîne B SERGIO (vérification dépôt) | B | 2 s | PASS |
| PySR import + mini-recherche | A/C/D | 38 s | PASS |
| PySR chaîne A toggle --quick | A | 32 s | PASS |
| PySR chaîne D synthétique --quick | D | 23 s | PASS |
| PySR chaîne C données réelles --quick | C | 22 s | PASS |
| Figures et tableaux (make_figures.py) | Visualisation | 2 s | PASS |

**16/16 tests réussis** — les valeurs `--quick` ne sont pas interprétables scientifiquement ; seul le fonctionnement est vérifié.

---

## C.7 Reproductibilité complète

Pour reproduire l'intégralité des résultats depuis zéro :

```bash
# 1. Installer l'environnement
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Télécharger et vérifier les données
python setup_all.py

# 3. Vérifier le pré-enregistrement
python checks/prereg.py verify

# 4. Exécuter toutes les chaînes (durée : plusieurs heures)
python run_everything.py --all --seeds 0 1 2 3 4

# 5. Vérifier les résultats
python checks/verify_results.py
python checks/consistency.py
python checks/make_report.py

# 6. Générer les figures
python make_figures.py
```

**Reproductibilité partielle (test rapide) :**

```bash
python smoke_test.py  # ~200 s — vérifie le fonctionnement, pas les valeurs
```

**Limitations identifiées :**

- PySR installe Julia automatiquement au premier import (~2–5 min)
- scvelo 0.3.4 est incompatible avec le mode `stochastic` sous NumPy ≥ 2.x (mode `deterministic` utilisé)
- Les sorties de scvelo contiennent des caractères UTF-8 nécessitant `encoding='utf-8'` dans `subprocess.run`
- Windows Spawn multiprocessing : `n_jobs=1` imposé dans scVelo velocity_graph
