# Annexe A — Démonstrations mathématiques

## A.1 Théorème de Grönwall et Proposition 1

### A.1.1 Lemme de Grönwall (forme intégrale)

**Énoncé.** Soient $u : [0, T] \to \mathbb{R}_+$ continue et $\alpha, \beta \geq 0$ tels que

$$
u(t) \leq \alpha + \beta \int_0^t u(s)\,ds, \quad \forall t \in [0, T].
$$

Alors

$$
u(t) \leq \alpha\, e^{\beta t}, \quad \forall t \in [0, T].
$$

**Démonstration.** Posons $v(t) = \alpha + \beta \int_0^t u(s)\,ds$. On a $v(0) = \alpha$,
$v'(t) = \beta\, u(t) \leq \beta\, v(t)$. L'inégalité différentielle donne $v(t) \leq v(0)\,e^{\beta t} = \alpha\,e^{\beta t}$.
Puisque $u(t) \leq v(t)$, le résultat est établi. $\square$

### A.1.2 Proposition 1 — Borne d'erreur entre trajectoire symbolique et trajectoire de référence

**Hypothèses.**

- $\mathbf{z}_{\text{true}}$ est la trajectoire du champ vrai $F$ et $\mathbf{z}_{\text{sym}}$ celle du champ appris $\hat{F}$.
- $\hat{F}$ est $L$-Lipschitz sur $\mathbb{R}^d$ : $\|\hat{F}(x) - \hat{F}(y)\| \leq L\,\|x - y\|$ pour tout $x, y$.
- L'erreur de champ est bornée uniformément : $\sup_z \|F(z) - \hat{F}(z)\| \leq \varepsilon$.
- Les conditions initiales peuvent différer : $\|\mathbf{z}_{\text{sym}}(0) - \mathbf{z}_{\text{true}}(0)\| \leq \delta_0$.

**Énoncé.**

$$
\|\mathbf{z}_{\text{sym}}(t) - \mathbf{z}_{\text{true}}(t)\| \leq \frac{\varepsilon}{L}\bigl(e^{Lt} - 1\bigr) + \delta_0\, e^{Lt}.
$$

**Démonstration.** Posons $e(t) = \mathbf{z}_{\text{sym}}(t) - \mathbf{z}_{\text{true}}(t)$. Par les équations intégrales :

$$
e(t) = e(0) + \int_0^t \bigl[\hat{F}(\mathbf{z}_{\text{sym}}(s)) - F(\mathbf{z}_{\text{true}}(s))\bigr]\,ds.
$$

En ajoutant et soustrayant $\hat{F}(\mathbf{z}_{\text{true}}(s))$ :

$$
\|e(t)\| \leq \delta_0 + \int_0^t \|\hat{F}(\mathbf{z}_{\text{sym}}) - \hat{F}(\mathbf{z}_{\text{true}})\|\,ds + \int_0^t \|\hat{F}(\mathbf{z}_{\text{true}}) - F(\mathbf{z}_{\text{true}})\|\,ds.
$$

Par Lipschitz et par la borne d'erreur :

$$
\|e(t)\| \leq \delta_0 + L \int_0^t \|e(s)\|\,ds + \varepsilon\, t.
$$

Le lemme de Grönwall avec $\alpha = \delta_0 + \varepsilon\, t$ appliqué à $u(t) = \|e(t)\|$ donne, après intégration :

$$
\|e(t)\| \leq \delta_0\, e^{Lt} + \frac{\varepsilon}{L}\bigl(e^{Lt} - 1\bigr). \quad \square
$$

**Interprétation.** La borne est composée de deux termes :
- $\frac{\varepsilon}{L}(e^{Lt}-1)$ : contribution de l'erreur de champ, amplifiée exponentiellement ;
- $\delta_0\,e^{Lt}$ : propagation des erreurs de condition initiale.

Lorsque $\delta_0 = 0$ (même CI) et $L \ll 1$, la borne croît linéairement : $\varepsilon\, t + O(\varepsilon L t^2)$.

### A.1.3 Validation empirique (métrique S4)

Le script `synthetic/src/synth_methods.py`, fonction `prop1_check`, calcule pour chaque trajectoire :

$$
\varepsilon = \max_{t} \|F(\mathbf{z}_{\text{true}}(t)) - \hat{F}(\mathbf{z}_{\text{true}}(t))\|,
\quad L = \max_{\text{pts}} \sigma_{\max}\!\left(\nabla \hat{F}\right), \quad B = \frac{\varepsilon}{L}(e^{Lt_{\max}} - 1).
$$

La constante de Lipschitz est estimée numériquement par la plus grande valeur singulière du Jacobien différences finies ($h = 10^{-4}$) sur les points des trajectoires vraies et symboliques. Sur **70 runs** (14 variantes × 5 graines), la fraction $\{E \leq B\}$ est **1.000** (seuil préenregistré : 0.95). ✓

---

## A.2 Cinétique de Hill et systèmes de régulation génique

### A.2.1 Fonction de Hill

La fonction de Hill activatrice ($+$) ou répressive ($-$) est :

$$
H^+(x;\, K, n) = \frac{x^n}{K^n + x^n}, \qquad H^-(x;\, K, n) = \frac{K^n}{K^n + x^n} = 1 - H^+(x;\, K, n).
$$

Avec la substitution $h = 1/(1 + (x/K)^n)$, l'implémentation (fichier `real_identify.py`) utilise directement :

```python
def hill(x, K, n, typ):
    h = 1.0 / (1.0 + (np.maximum(x, 0) / K) ** n)
    return h if typ == 'rep' else 1.0 - h
```

**Propriétés :** $H^\pm \in [0, 1]$ ; $H^+(K, n) = 1/2$ ; monotone croissante / décroissante ; sigmoïde pour $n > 1$.

### A.2.2 Modèle de réseau régulateur

L'équation différentielle pour le gène $i$ est :

$$
\frac{dz_i}{dt} = \sum_{k=1}^{K_i} a_{ik}\, H^{\pm}(z_{j_k};\, K_{ik}, n_{ik}) + a_{i0} - \gamma_i\, z_i,
$$

où $a_{ik} > 0$ (moindres carrés non négatifs), $\gamma_i > 0$ (taux de dégradation), et l'indice $j_k$ identifie le régulateur du terme $k$.

**Paramètres à identifier :** pour chaque terme Hill : $(a_{ik}, K_{ik}, n_{ik})$ + constante $\gamma_i$ → $3K_i + 1$ paramètres par gène.

### A.2.3 Complexité Ω

La complexité de pénalisation utilisée dans le front de Pareto est :

$$
\Omega(\mathcal{M}) = 1 + \sum_{k : \text{Hill}} 3 + \sum_{k : \text{const}} 1 + 1 \quad [\text{pour } \gamma].
$$

**Règle de sélection préenregistrée :** on choisit le modèle de plus petite complexité $\Omega$ tel que son erreur de validation soit $\leq (1 + \tau) \times$ la meilleure erreur, avec $\tau = 0.05$.

---

## A.3 Stabilité de Lyapunov et bistabilité

### A.3.1 Définitions

Un point d'équilibre $\mathbf{z}^* : F(\mathbf{z}^*) = 0$ est **asymptotiquement stable** si toutes les valeurs propres de $J = \nabla F(\mathbf{z}^*)$ ont une partie réelle strictement négative ($\lambda_{\max}(\text{Re}(J)) < 0$).

Un système est **bistable** s'il possède exactement deux attracteurs stables séparés par un point col (selle).

### A.3.2 Calcul numérique du Jacobien

```python
def jacobian(fun, x, eps=1e-5):
    d = len(x); J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps
        J[:, j] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return J
```

Les valeurs propres sont calculées par `np.linalg.eigvals(J)`.

---

## A.4 Perte d'entraînement Neural ODE

### A.4.1 Données réelles — appariement de vitesse

La perte est l'erreur quadratique pondérée entre le champ $f_\theta(\mathbf{z}_i)$ et la vélocité mesurée $\mathbf{v}_i$ :

$$
\mathcal{L}_{\text{vel}} = \frac{1}{|I_{\text{tr}}|} \sum_{i \in I_{\text{tr}}} \frac{w_i \|\mathbf{v}_i - f_\theta(\mathbf{z}_i)\|^2}{\nu}, \quad \nu = \frac{1}{|I_{\text{tr}}|} \sum_i w_i \|\mathbf{v}_i\|^2 + 10^{-12}.
$$

Les poids $w_i = $ `velocity_confidence` normalisés à moyenne 1. La normalisation $\nu$ rend la perte sans dimension.

**Régularisation :** $\mathcal{L} = \mathcal{L}_{\text{vel}} + \lambda \cdot \|f_\theta(\mathbf{z})\|^2 / \nu$ (pénalité de magnitude).

### A.4.2 Données synthétiques — trajectoires

La perte est l'erreur de reconstruction des trajectoires complètes :

$$
\mathcal{L}_{\text{obs}} = \frac{1}{|\mathcal{T}_{\text{tr}}|} \sum_{(\text{traj}, t)} \frac{\|\hat{\mathbf{z}}(t) - \mathbf{z}(t)\|^2}{\sigma^2_{\text{data}}}.
$$

Le solveur est RK4 à pas fixe ($\Delta t = 0.1$) ou la méthode adjointe (variante `adjoint`).
