# Rapport d'audit final — Corrections appliquées au manuscrit

> Généré le 2026-11-22. Source de vérité : fichiers JSON dans `c:\neural_ode_symbolic\docs\` et `real_data/results/`.  
> Vérification d'intégrité : **168/168 fichiers OK, 0 échec** (`checks/verify_results.py --strict`)

---

## Chiffres réels issus des expériences (vérifiés depuis les JSON)

### Pancréas E15.5 — λ sweep 5 graines

| λ fixe | Cosinus test (brut) | Cosinus test (centré) | NRMSE |
|--------|--------------------|-----------------------|-------|
| 0.0 | 0.557 ± 0.042 | 0.619 ± 0.022 | 0.0362 |
| 0.001 | 0.520 ± 0.030 | 0.591 ± 0.020 | 0.0390 |
| 0.01 | 0.522 ± 0.065 | 0.591 ± 0.017 | 0.0386 |
| 0.1 | 0.536 ± 0.023 | 0.573 ± 0.043 | 0.0368 |
| 1.0 | 0.535 ± 0.120 | 0.576 ± 0.028 | 0.0385 |
| **Sélection gelée (val)** | **0.592 ± 0.038** | **0.607 ± 0.022** | **0.0365** |
| **Linéaire (baseline)** | **0.588** | **0.586** | **0.040** |

> **IMPORTANT — Aucun λ fixe ne bat le linéaire en cosinus brut.** Le 0.592 résulte de la sélection post-hoc par seed sur la validation — résultat exploratoire, pas confirmé pré-enregistré.

### Dentate Gyrus — 5 graines

| Seed | NODE cos | Linéaire | Sym~node (fidélité) | Direct FD |
|------|----------|----------|---------------------|-----------|
| 0 | 0.639 | 0.634 | 0.973 | 0.620 |
| 1 | 0.630 | 0.634 | 0.983 | 0.671 |
| 2 | 0.591 | 0.634 | 0.932 | 0.623 |
| 3 | 0.550 | 0.634 | 0.914 | 0.631 |
| 4 | 0.551 | 0.634 | 0.965 | 0.616 |
| **Moy** | **0.592 ± 0.038** | **0.634** | **0.953 ± 0.026** | **0.632 ± 0.020** |

> **ATTENTION — DG-NODE (0.592) < Linéaire (0.634)** et **< Direct FD (0.632)**. Le point fort réel est la fidélité de distillation (0.953, 100% ≥ 0.90), pas la performance directe du NODE.

### Robustesse bruit — Chute FD à σ=20%

| Condition | FD F1 moyen | σ |
|-----------|-------------|---|
| Base (σ=0) | 0.887 | 0.069 |
| Bruit σ=20% | 0.865 | 0.076 |
| **Chute réelle** | **0.022** | — |

> **CRITIQUE — Le DOCX Claude v6 uploadé mentionnait une chute de 0.21 — erreur de 10×.** La vraie chute est 0.022 (2.2 points de F1). Notre fichier source MD était déjà correct.

### Mauvaise spécification de n (5 graines)

| Condition | Discret F1 | Continu F1 | Différence | Significatif ? |
|-----------|-----------|-----------|------------|----------------|
| n_true=3.0 | 0.748 ± 0.068 | 0.695 ± 0.069 | +0.053 | ≈0.8 s.d. — NON |
| n_true=1.5 | 0.667 ± 0.074 | 0.660 ± 0.111 | +0.007 | <0.1 s.d. — NON |
| n_true=2.0 | 0.723 ± 0.060 | 0.718 ± 0.113 | +0.005 | <0.1 s.d. — NON |

---

## 6 Corrections appliquées au fichier source

| # | Problème (Claude v6) | Correction dans notre MD |
|---|---------------------|--------------------------|
| 1 | "beats linear decisively" (R1a) | Reformulé : "marginally exceeds (+0.004, ~0.1 s.d.); no fixed λ beats linear; exploratory result" |
| 2 | "actually outperforming continuous" (misspec) | Reformulé : "matching or marginally exceeding (within 1 s.d., not statistically conclusive)" |
| 3 | DG-NODE présenté sans baseline linéaire | Ajouté : "NODE below linear (0.634) and direct FD (0.632) on DG" |
| 4 | DG-SYM-MEAS framing optimiste | Ajouté "direct FD is better baseline on DG"; std corrigé à 0.022 |
| 5 | "D. Hochgerner" (prénom faux) | Corrigé → **"H. Hochgerner"** (Hannah) |
| 6 | SHA-256 DG absent du texte | Ajouté dans §3 Discussion : `6fbf1ec64e0fba6d18d14bc342b3ba2aab8133972c1e50cb2cb717d4cc336871` |

---

## Corrections NON requises dans notre source

| Sujet | Statut |
|-------|--------|
| Chute FD 0.21 | **Absent de notre MD** — erreur uniquement dans le DOCX Claude v6 uploadé |
| R2 sym~node = 0.890 < 0.90 | Déjà marqué "Marginal" dans Table III — honnête |
| Résultat 0.640 (seed 0 unique) | Déjà absent du manuscrit |

---

## Fichiers produits

- **Source MD corrigé** : `docs/IEEE_SECTION_IMPLEMENTATION_AND_CONCLUSION.md`
- **DOCX recompilé** : `docs/IEEE_SECTION_v2_corrected.docx` (2.08 MB)
- **Intégrité** : 168/168 OK, 0 avertissement

---

## Ce qui reste ouvert (non corrigeable sans nouvelles expériences)

| Insuffisance | Raison |
|-------------|--------|
| R2 sym~node = 0.890 (seuil 0.90 non atteint) | Marginal — besoin d'architecture plus fine |
| DG-NODE < linéaire sur données propres | Maintenant documenté honnêtement |
| Validation externe SERGIO | Expérience non lancée |
| Deuxième jeu de données réel | Hors scope session courante |
