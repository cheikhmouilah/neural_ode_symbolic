"""
Coupled (u, s) Biophysical Dynamical System:
Models the joint transcription-splicing-degradation kinetics:
    du_i/dt = alpha_i(s) - beta_i * u_i
    ds_i/dt = beta_i * u_i - gamma_i * s_i

where:
- u_i is unspliced pre-mRNA (transcribed from DNA, spliced into mature mRNA)
- s_i is mature spliced mRNA (translated into proteins, regulates transcription factors)
- alpha_i(s) is the regulated transcription rate (modeled via Hill kinetics or Neural ODE)
- beta_i is the splicing rate constant
- gamma_i is the degradation rate constant
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares, nnls
from scipy.integrate import solve_ivp

class CoupledNeuralField(nn.Module):
    """
    Learns the regulated transcription rate alpha_theta(s) >= 0
    while enforcing the analytical linear splicing-degradation kinetics:
        du/dt = alpha_theta(s) - beta * u
        ds/dt = beta * u - gamma * s
    """
    def __init__(self, d, hidden=64):
        super().__init__()
        self.d = d
        # alpha_theta(s): Neural network mapping mature regulators s to transcription rate
        self.net_alpha = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
            nn.Softplus() # Guarantees non-negative transcription rate alpha >= 0
        )
        # Log-rates for positive splicing beta > 0 and degradation gamma > 0
        self.log_beta = nn.Parameter(torch.zeros(d))
        self.log_gamma = nn.Parameter(torch.zeros(d))

    def forward(self, u, s):
        alpha = self.net_alpha(s)
        beta = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        du_dt = alpha - beta * u
        ds_dt = beta * u - gamma * s
        return du_dt, ds_dt

def train_coupled_us_node(U_tr, S_tr, Vel_tr, cfg, seed=0, epochs=350, lr=0.003):
    """
    Trains the coupled (u, s) system on single-cell unspliced and spliced moments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = S_tr.shape[1]
    model = CoupledNeuralField(d, hidden=cfg.get('hidden', 64))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    u_t = torch.as_tensor(U_tr, dtype=torch.float32)
    s_t = torch.as_tensor(S_tr, dtype=torch.float32)
    v_target = torch.as_tensor(Vel_tr, dtype=torch.float32)
    norm = float((v_target ** 2).sum(1).mean()) + 1e-12

    best_loss = np.inf
    best_weights = None

    for ep in range(epochs):
        opt.zero_grad()
        du_pred, ds_pred = model(u_t, s_t)
        # ds/dt must match observed RNA velocity (beta*u - gamma*s = v)
        loss_vel = ((ds_pred - v_target) ** 2).sum(1).mean() / norm
        # Regularization on transcription magnitude
        loss_reg = 0.001 * (du_pred ** 2).sum(1).mean() / norm
        loss = loss_vel + loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    beta_est = torch.exp(model.log_beta).detach().numpy()
    gamma_est = torch.exp(model.log_gamma).detach().numpy()
    return model, dict(best_loss=best_loss, beta=beta_est, gamma=gamma_est)

def fit_coupled_hill_system(U_data, S_data, genes, beta, gamma):
    """
    Identifies the explicit Hill equations governing transcription:
        alpha_i(s) = sum_k a_ik * H(s_jk; K, n) + a_i0
    from the steady-state transcription requirements alpha_i = beta_i * u_i + du_i/dt.
    """
    d = len(genes)
    models = []
    # In pseudo-steady-state or dynamic flow: alpha_target = beta * u
    Alpha_target = beta[None, :] * U_data

    for i in range(d):
        y_i = Alpha_target[:, i]
        # Screen candidate regulators among mature mRNAs
        best_r2 = -np.inf
        best_eq = None
        for j in range(d):
            x_j = S_data[:, j]
            for typ in ['act', 'rep']:
                for K in [0.2, 0.5, 0.8, 1.2]:
                    for n in [1.0, 2.0, 4.0]: # Discrete biophysical stoichiometry
                        h = 1.0 / (1.0 + (x_j / K)**n) if typ == 'rep' else (x_j / K)**n / (1.0 + (x_j / K)**n)
                        # Fit amplitude a and basal a0 via NNLS
                        A = np.column_stack([h, np.ones_like(h)])
                        coef, _ = nnls(A, y_i)
                        pred = A @ coef
                        r2 = 1.0 - np.sum((y_i - pred)**2) / np.sum((y_i - y_i.mean())**2)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_eq = dict(gene=genes[i], regulator=genes[j], typ=typ,
                                           K=K, n=n, a=float(coef[0]), a0=float(coef[1]),
                                           beta=float(beta[i]), gamma=float(gamma[i]), r2=float(r2))
        models.append(best_eq)
    return models

def simulate_coupled_us(model_eqs, u0, s0, T=5.0, steps=100):
    """
    Simulates the continuous (u, s) 2d-dimensional ODE system over time.
    """
    d = len(model_eqs)
    genes = [m['gene'] for m in model_eqs]

    def ode_fun(t, x):
        u = x[:d]
        s = x[d:]
        du_dt = np.zeros(d)
        ds_dt = np.zeros(d)
        for i, m in enumerate(model_eqs):
            j = genes.index(m['regulator'])
            sj = s[j]
            K, n = m['K'], m['n']
            h = 1.0 / (1.0 + (sj / K)**n) if m['typ'] == 'rep' else (sj / K)**n / (1.0 + (sj / K)**n)
            alpha_i = m['a'] * h + m['a0']
            du_dt[i] = alpha_i - m['beta'] * u[i]
            ds_dt[i] = m['beta'] * u[i] - m['gamma'] * s[i]
        return np.concatenate([du_dt, ds_dt])

    x0 = np.concatenate([u0, s0])
    t_eval = np.linspace(0, T, steps)
    sol = solve_ivp(ode_fun, (0, T), x0, t_eval=t_eval, method='LSODA')
    return sol.t, sol.y[:d, :].T, sol.y[d:, :].T # t, U_traj, S_traj
