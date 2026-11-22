"""Neural ODE sur donnees reelles : champ f_theta(z) autonome, appris par appariement de vitesse (eq. 8, 10 de la partie theorique).
Pas de temps experimental ni de pseudotemps : le terme L_obs de l'eq. 9 est ABSENT (ecart documente dans docs/CONSISTENCE_THEORIE.md).
Le solveur (torchdiffeq, dopri5) sert a integrer le champ appris (trajectoires depuis les progeniteurs)."""
import os, time, numpy as np, torch, torch.nn as nn
from torchdiffeq import odeint
torch.set_num_threads(int(os.environ.get('NODE_THREADS', '1')))   # 1 processus = 1 coeur ; le lanceur parallelise sur les coeurs


class VF(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d))

    def forward(self, t, x):
        return self.net(x)


def field_np(f, X):
    with torch.no_grad():
        return f(0.0, torch.as_tensor(np.asarray(X, np.float32).reshape(-1, X.shape[-1]))).numpy().astype(float).reshape(np.shape(X))


def train_vel_node(Z, V, W, tr, va, cfg, seed, lam_reg=None, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    d = Z.shape[1]; lam = cfg['lam_reg'] if lam_reg is None else lam_reg
    tr = np.asarray(tr)
    if cfg.get('train_subsample', 1.0) < 1.0:
        tr = rng.choice(tr, size=int(cfg['train_subsample'] * len(tr)), replace=False)
    zt, vt, wt = [torch.as_tensor(a[tr], dtype=torch.float32) for a in (Z, V, W)]
    zv, vv, wv = [torch.as_tensor(a[va], dtype=torch.float32) for a in (Z, V, W)]
    norm = float((wt[:, None] * vt ** 2).sum(1).mean()) + 1e-12      # echelle des vitesses : perte sans dimension
    f = VF(d, cfg['hidden']); opt = torch.optim.Adam(f.parameters(), lr=cfg['lr'])
    best = (np.inf, None, 0); t0 = time.time(); bad = 0
    for ep in range(cfg['epochs']):
        opt.zero_grad(); pred = f(0.0, zt)
        l_vel = (wt[:, None] * (vt - pred) ** 2).sum(1).mean() / norm                 # eq. (8) : L_vel
        reg = (pred ** 2).sum(1).mean() / norm if cfg['reg_type'] == 'magnitude' else 0.0   # R : magnitude du champ
        loss = l_vel + lam * reg; loss.backward(); nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step()
        with torch.no_grad():
            lv = float((wv[:, None] * (vv - f(0.0, zv)) ** 2).sum(1).mean() / norm)
        if lv < best[0] - 1e-6: best = (lv, {k: x.clone() for k, x in f.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad > cfg['patience']: break
        if ep % 100 == 0: log(f'  ep={ep:4d} train_vel={float(l_vel):.4f} val_vel={lv:.4f}')
    f.load_state_dict(best[1])
    return f, dict(best_val=best[0], best_epoch=best[2], seconds=time.time() - t0, n_train=int(len(tr)), lam_reg=lam, norm=norm)


def integrate(f, z0, T, n=60, rtol=1e-6, atol=1e-8, method='dopri5'):
    t = torch.linspace(0, T, n)
    with torch.no_grad():
        out = odeint(f, torch.as_tensor(np.asarray(z0, np.float32)), t, rtol=rtol, atol=atol, method=method)
    return out.permute(1, 0, 2).numpy().astype(float)       # (cells, n, d)
