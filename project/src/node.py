"""Modele B - Neural ODE (torchdiffeq, dopri5), apprentissage sur trajectoires avec curriculum d'horizon."""
import time, numpy as np, torch, torch.nn as nn
from torchdiffeq import odeint
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class VF(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, t, x):
        return self.net(x)


def field_np(f, X):
    with torch.no_grad():
        return f(0.0, torch.as_tensor(np.asarray(X, float).reshape(-1, 2))).numpy().reshape(np.shape(X))


def predict(f, x0, t, cfg):
    with torch.no_grad():
        out = odeint(f, torch.as_tensor(np.asarray(x0, float)), torch.as_tensor(t), rtol=cfg['rtol'], atol=cfg['atol'], method=cfg['solver'])
    return out.permute(1, 0, 2).numpy()


def train_node(t, Xtr, Xva, cfg, seed, Vtr=None, log=print, perturb_rel=0.0):
    torch.manual_seed(seed)
    f = VF(cfg['hidden'])
    if perturb_rel:      # sonde de sensibilite : perturbation relative des poids initiaux (generateur separe : le tirage nominal n'est pas modifie)
        g_ = torch.Generator().manual_seed(seed + 99991)
        with torch.no_grad():
            for p_ in f.parameters(): p_.mul_(1.0 + perturb_rel * torch.randn(p_.shape, generator=g_, dtype=p_.dtype))
    opt = torch.optim.Adam(f.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg['lr_decay'])
    tt = torch.as_tensor(t); Xt = torch.as_tensor(Xtr); Xv = torch.as_tensor(Xva)
    Vt = None if Vtr is None else torch.as_tensor(Vtr).reshape(-1, 2)
    vw = cfg.get('velocity_weight', 0.0)
    best = (np.inf, None); it = 0; t0 = time.time(); hist = []
    stages = list(zip(cfg['curriculum_points'], cfg['iters_per_stage']))
    for si, (npts, iters) in enumerate(stages):
        for k in range(iters):
            opt.zero_grad()
            pred = odeint(f, Xt[:, 0, :], tt[:npts], rtol=cfg['rtol'], atol=cfg['atol'], method=cfg['solver']).permute(1, 0, 2)
            loss = ((pred - Xt[:, :npts, :]) ** 2).mean()
            if vw > 0 and Vt is not None:
                loss = loss + vw * ((f(0.0, Xt.reshape(-1, 2)) - Vt) ** 2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(f.parameters(), 5.0); opt.step(); sched.step(); it += 1
            if it % 50 == 0 or (si == len(stages) - 1 and k == iters - 1):
                with torch.no_grad():
                    pv = odeint(f, Xv[:, 0, :], tt, rtol=cfg['rtol'], atol=cfg['atol'], method=cfg['solver']).permute(1, 0, 2)
                    vl = float(((pv - Xv) ** 2).mean())
                hist.append((it, float(loss), vl)); log(f"  it={it:4d} stage={si} npts={npts:2d} train={float(loss):.3e} val(full)={vl:.3e} t={time.time()-t0:.0f}s")
                if si == len(stages) - 1 and vl < best[0]:
                    best = (vl, {k_: v.clone() for k_, v in f.state_dict().items()})
    f.load_state_dict(best[1]); return f, dict(best_val=best[0], history=hist, seconds=time.time() - t0)
