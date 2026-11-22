# Résultats de la chaîne C (pancréas E15.5, scVelo, 8 gènes, 5 graines)

Configuration `real_data/configs/real_pancreas.json`, SHA-256 `0d0de6753cedfcf6`. Seuils fixés avant exécution. Un seul changement de configuration
(`velocity_mode` : `stochastic` → `deterministic`, après une erreur technique et avant tout résultat). Un essai de la graine 0 a été vu avant le lancement complet.
Verdicts produits par `scripts/analyze_real.py` (`results/real/verdicts.json`).

| Critère | Valeur | Seuil | Verdict |
|:--|:--|:--|:--|
| R1 cosinus NODE sur test (moyenne) | 0,489 | ≥ 0,50 | **Échec** |
| R1 marge sur contrôle à vitesses mélangées (0,367) | 0,122 | ≥ 0,30 | **Échec** |
| R2 cosinus symbolique/NODE (moyenne) | 0,860 | ≥ 0,90 | **Échec** |
| R3 Jaccard de support entre graines | 0,631 | ≥ 0,60 | Réussite (marge faible) |
| R4 Jaccard de support selon λ_reg | 0,52 à 0,61 | ≥ 0,60 | **Échec** |
| R7 rapport cos(sym,V)/cos(NODE,V) | 0,993 | ≥ 0,80 | Réussite |

Descriptif (sans seuil) : le modèle linéaire (cosinus 0,588) fait mieux que le NODE à λ_reg = 0,001 ; l'identification directe donne 0,576 contre 0,484 pour la distillation ;
le Jaccard de structure distillé/direct vaut 0,52 ; le NODE n'a trouvé aucun équilibre ; le symbolique en trouve 0 à 3 selon la graine ; les points d'arrivée depuis les progéniteurs
varient fortement d'une graine à l'autre. Le balayage de λ_reg (graine 0 seulement) donne des cosinus de 0,64 (λ=0), 0,50 (0,001), 0,55 (0,01), 0,61 (0,1), 0,54 (1,0) : résultat post hoc,
la valeur de référence et les seuils n'ont pas été modifiés.

**Lecture honnête.** Sur ce jeu, avec ces 8 gènes et ces réglages, la chaîne ne démontre pas que les équations sélectionnées reflètent la dynamique : le champ neuronal ne dépasse pas
nettement une référence linéaire, la structure dépend de λ_reg et de la graine, et il n'existe pas de vérité terrain. Rien ici n'établit un mécanisme ni une causalité.
