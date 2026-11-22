# Benchmark synthétique multi-gènes : résultats du pilote (session de rédaction)

Configuration `synthetic/configs/synth.json`, SHA-256 `711a030b8817bd95`, seuils fixés avant la première exécution et jamais modifiés.
**Pilote partiel** : `base` sur 5 graines ; les 11 autres variantes sur **1 seule graine (0)**. L'exécution complète (12 variantes × 5 graines) est prévue par `run_everything.py`.
Générateur bien spécifié (même famille que la bibliothèque), trajectoires observées à 10 temps, 60 trajectoires, $d=5$.

| Critère | Valeur | Seuil | Verdict |
|:--|:--|:--|:--|
| S1 F1 de structure (distillé, base) | 0,778 (précision 0,65 ; rappel 0,98) | ≥ 0,70 | Réussite |
| S2 erreur relative médiane des paramètres retrouvés | 0,441 | ≤ 0,25 | **Échec** |
| S3 cosinus NODE / champ vrai | 0,939 | ≥ 0,90 | Réussite |
| S4 proposition 1 : borne tenue (15 runs) | 100 % | ≥ 95 % | Réussite |
| S5 pseudotemps ($\gamma=2$), cosinus de direction | 0,840 (rapport de vitesse 2,69) | ≥ 0,90 | **Échec** |
| S6 chute de cosinus hors support (variété) | −0,02 | ≥ 0,10 | **Échec** |

Observations (descriptives, souvent 1 graine) :
- **La distillation ne bat pas les bases de référence sur la structure** : F1 direct 0,887, forme faible 0,864, distillé 0,778 (base, 5 graines). La précision du distillé est faible : il sur-sélectionne des termes.
- **Les constantes de Hill sont mal estimées même avec la structure exacte** (erreur médiane 0,37 pour la base « oracle »), ce qui concorde avec la discussion d'identifiabilité de la section II-E.
- **Le bruit gaussien jusqu'à 20 % dégrade peu le champ neuronal** (cosinus 0,97) mais fait baisser le F1 distillé (0,70 à 20 %).
- **Le dropout est très destructeur** : F1 distillé 0,44 à 10 %, 0,17 à 20 % ; cosinus du champ neuronal 0,74 puis 0,65 (modèle d'erreur gaussien mal adapté aux zéros).
- **Pseudotemps (remarque 2)** : la direction du champ reste correcte (cosinus 0,84), mais la vitesse est faussée d'un facteur d'environ 2,7, comme le prévoit la théorie ; la forme faible s'effondre (F1 0,35).
- **Variété (remarque 1)** : le F1 s'effondre (0,17) quand les conditions initiales sont sur une courbe, mais le cosinus hors support ne baisse pas (−0,02) : la métrique préenregistrée S6 ne capte pas le phénomène. Une graine seulement.

À ne pas conclure : rien ne montre ici que les résultats se transfèrent aux données réelles, ni que les variantes à une graine sont stables.
