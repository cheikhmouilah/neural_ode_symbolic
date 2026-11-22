import os as _os
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
SERGIO_DIR = _os.path.join(WORK, 'SERGIO')
"""Test : la mauvaise identification de n pour les activations vient-elle de la plage d'expression du regulateur ?
Vraie loi : P = K x^n/(h^n+x^n), K=3, h=1.3, n=2, bruit gaussien sd=0.35 (ordre de grandeur de la dispersion SERGIO)."""
import numpy as np, json
from scipy.optimize import curve_fit
rng = np.random.default_rng(0)
K, h, n_true, sd = 3.0, 1.3, 2.0, 0.35
f = lambda x, a, hh, n: a * x**n / (hh**n + x**n)
x_emp = np.load(WORK + '/sergio_clean.npz')['U'][9].astype(float)      # regulateur reel (gene 9) : 6 niveaux
designs = {
    "observe (SERGIO, gene 9)":        lambda m: rng.choice(x_emp, m),
    "plage elargie x0.3-3h":           lambda m: rng.uniform(0.3*h, 3*h, m),
    "plage elargie x0.2-6h":           lambda m: rng.uniform(0.2*h, 6*h, m),
}
res = {}
for name, draw in designs.items():
    nh, ok = [], 0
    for _ in range(300):
        x = draw(1800); y = f(x, K, h, n_true) + rng.normal(0, sd, x.size)
        try:
            p, _ = curve_fit(f, x, y, p0=[y.max(), x.mean(), 2.0], bounds=([0,1e-6,0.5],[np.inf,np.inf,6.0]), maxfev=20000)
            nh.append(p[2]); ok += abs(p[2]-n_true) < 0.5
        except Exception: pass
    q = np.percentile(nh, [25, 50, 75])
    res[name] = dict(n_hat_q25=q[0], n_hat_median=q[1], n_hat_q75=q[2], frac_n_within_0p5=ok/len(nh))
    print(f"{name:28s} n_hat median={q[1]:.2f} IQR=[{q[0]:.2f},{q[2]:.2f}]  |n_hat-2|<0.5 : {100*ok/len(nh):.0f}%")
json.dump(res, open(WORK + '/result_identifiability.json','w'), indent=1)
