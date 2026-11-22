# Essai pilote PySR (session de rédaction, UNE graine)

PySR 2.7.2, Julia 1.13.3, Linux, 1 coeur ; 40 itérations, recherche en série et déterministe. Fichiers : `reference_results/pysr_pilot/`. **Ces valeurs viennent d'une seule graine et ont été vues avant de fixer les seuils PT, RP** (divulgué dans les configurations).

**Toggle (graine 0), vérité : dx/dt = a/(1+(y/K)^n) − g·x avec a = (4,0 ; 3,6), K = (0,8 ; 1,0), n = 2, g = (1,0 ; 0,9).**

| Équation | Expression trouvée | Complexité | Commentaire |
|:--|:--|:--|:--|
| x0 | `2,607 / (x1² + 0,6647) − x0` | 9 | Forme d'une répression de Hill avec n = 2 ; la vérité s'écrit 2,56 / (x1² + 0,64) − x0 |
| x1 | `0,3664·x0² + (x0 − (−0,3723·x1 + 1,6327))·(−2,2757)` | 15 | Polynôme du second degré en x0 et linéaire en x1 : **n'a pas la forme de Hill** ; il approche la fonction sur le domaine étudié |

Mesures : NRMSE du champ contre la vérité 0,0116 (Neural ODE seul : 0,0124) ; R² au Neural ODE 0,977 et 0,978 ; les deux points fixes stables sont conservés (erreur relative maximale 0,018) ; NRMSE de trajectoire 0,0136. Durée : 112 s dont ~90 s pour le Neural ODE.

**Données réelles, pancréas (graine 0, 2 équations sur 8).** Cosinus PySR/NODE 0,652 (le seuil RP1 de 0,90 n'est pas atteint) ; rapport cosinus(PySR, vitesse)/cosinus(NODE, vitesse) 0,82 ; équation Ghrl : expression linéaire en Ghrl seul (complexité 5) ; équation Gnas : Chga et Cpe (complexité 7).

À ne pas conclure : une graine, pas de vérité terrain sur les données réelles, et une expression exacte sur une équation du toggle ne démontre pas que PySR retrouve la bonne forme en général (la seconde équation ne l'a pas retrouvée).
