"""Garde-fous d'execution (chapitre Implementation, section 10) : le regime dynamique est verifie AVANT tout entrainement ; un systeme non conforme arrete l'execution."""
import numpy as np
from toggle import fixed_points

STARTS = [(a, b) for a in np.linspace(0, 7, 15) for b in np.linspace(0, 7, 15)]


class GuardError(RuntimeError):
    pass


def assert_bistable(fun, label, min_separation=None):
    """Exige exactement 2 points fixes stables et 1 point selle ; separation selle-etat stable >= min_separation si fournie."""
    fps = fixed_points(fun, starts=STARTS); st = [f for f in fps if f['stable']]; sd = [f for f in fps if not f['stable']]
    if len(st) != 2 or len(sd) != 1:
        raise GuardError(f'[GARDE-FOU] {label} : regime non bistable (etats stables : {len(st)}, selles/instables : {len(sd)}). Arret AVANT entrainement.')
    sep = float(min(np.linalg.norm(s['x'] - sd[0]['x']) for s in st))
    if min_separation is not None and sep < min_separation:
        raise GuardError(f'[GARDE-FOU] {label} : separation selle-etat stable {sep:.2f} < {min_separation}. Arret AVANT entrainement.')
    return dict(stable=[s['x'].tolist() for s in st], saddle=sd[0]['x'].tolist(), separation=sep)


def assert_finite(X, label):
    if not np.all(np.isfinite(X)): raise GuardError(f'[GARDE-FOU] {label} : valeurs non finies dans les donnees. Arret.')
