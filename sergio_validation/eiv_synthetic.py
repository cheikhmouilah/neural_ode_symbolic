"""Modele synthetique d'erreur sur le regulateur (EIV) : n_hat en fonction du bruit relatif sur x.
Vraie loi : P = K x^n/(h^n+x^n), K=3, h=1.3, n=2, bruit gaussien sd=0.25 sur y ; x observe = x_vrai*(1+CV*eps).
x_vrai tire dans les niveaux du regulateur 9 de SERGIO. Ecrit result_eiv.json."""
import os as _os, json, numpy as np
from scipy.optimize import curve_fit
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
rng = np.random.default_rng(1)
K, h, n0, sd = 3.0, 1.3, 2.0, 0.25
f = lambda x, a, hh, n: a * x**n / (hh**n + x**n)
x_emp = np.load(WORK + '/sergio_clean.npz')['U'][9].astype(float)
out = {}
for cv in (0.0, 0.15, 0.3, 0.5):
    nh = []
    for _ in range(200):
        xt = rng.choice(x_emp, 1800); y = f(xt, K, h, n0) + rng.normal(0, sd, 1800)
        xo = np.clip(xt * (1 + cv * rng.standard_normal(1800)), 1e-3, None)
        try:
            p, _ = curve_fit(f, xo, y, p0=[y.max(), xo.mean(), 2.0], bounds=([0, 1e-6, 0.5], [np.inf, np.inf, 6.0]), maxfev=20000); nh.append(p[2])
        except Exception:
            pass
    q = np.percentile(nh, [25, 50, 75]); out[str(cv)] = q.tolist()
    print(f"bruit sur x (CV) = {cv:.2f} -> n_hat median={q[1]:.2f}  IQR=[{q[0]:.2f},{q[2]:.2f}]")
json.dump(out, open(WORK + '/result_eiv.json', 'w'))
