import os as _os
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
SERGIO_DIR = _os.path.join(WORK, 'SERGIO')
"""
Validation synthetique : le pipeline (champ neuronal -> extraction symbolique)
retrouve-t-il les equations de regulation injectees dans SERGIO ?

Verite terrain SERGIO (cible t, regulateur unique r, coop n=2) :
    dU_t/dt = |K| * Hill(U_r; h, n)  - lambda * U_t        (activation, K>0)
    dU_t/dt = |K| * (1 - Hill(U_r))  - lambda * U_t        (repression,  K<0)
avec Hill(x) = x^n / (h^n + x^n), h = moyenne de U_r sur les types cellulaires, lambda = 0.8.
Production observable (approximation quasi-stationnaire) : P = lambda * U_t.
"""
import sys, warnings, json
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPRegressor
from gplearn.genetic import SymbolicRegressor

warnings.filterwarnings("ignore")
LAM = 0.8
D = SERGIO_DIR + "/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/"


# ---------- verite terrain ----------
def load_truth():
    truth = []
    for line in open(D + "Interaction_cID_7.txt"):
        v = [float(x) for x in line.strip().split(",")]
        t, nreg = int(v[0]), int(v[1])
        if nreg == 1:
            truth.append(dict(t=t, r=int(v[2]), K=v[3], n=v[4]))
    return truth


# ---------- familles de formes candidates (biologiquement plausibles) ----------
def f_lin(x, a):            return a * x
def f_mm(x, a, h):          return a * x / (h + x)
def f_act(x, a, h, n):      return a * x**n / (h**n + x**n)
def f_rep(x, a, h, n):      return a * h**n / (h**n + x**n)


def fit_family(x, y):
    """Ajuste chaque forme, selection par BIC. Retourne dict de resultats."""
    N = len(x); out = {}
    xm, ym = max(x.mean(), 1e-6), max(np.abs(y).max(), 1e-6)
    specs = {
        "lineaire":    (f_lin, [ym / xm],            ([-np.inf], [np.inf])),
        "michaelis":   (f_mm,  [ym, xm],             ([0, 1e-6], [np.inf, np.inf])),
        "hill_act":    (f_act, [ym, xm, 2.0],        ([0, 1e-6, 0.5], [np.inf, np.inf, 6.0])),
        "hill_rep":    (f_rep, [ym, xm, 2.0],        ([0, 1e-6, 0.5], [np.inf, np.inf, 6.0])),
    }
    for name, (f, p0, bnds) in specs.items():
        try:
            p, _ = curve_fit(f, x, y, p0=p0, bounds=bnds, maxfev=20000)
            rss = np.sum((y - f(x, *p)) ** 2)
            k = len(p)
            bic = N * np.log(rss / N + 1e-12) + k * np.log(N)
            out[name] = dict(p=[float(q) for q in p], bic=float(bic), rss=float(rss))
        except Exception:
            pass
    best = min(out, key=lambda k: out[k]["bic"])
    return best, out


def nrmse(y_true, y_hat):
    rng = y_true.max() - y_true.min()
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)) / max(rng, 1e-9))


def shape(c):
    m = np.abs(c).max()
    return c / m if m > 0 else c


def true_curve(x, K, h, n):
    hill = x**n / (h**n + x**n)
    return abs(K) * (hill if K > 0 else 1 - hill)


# ---------- pipeline ----------
def run_pipeline(U, label, normalize, rng_seed=0, max_targets=14):
    truth = load_truth()
    rows = []
    for tr in truth[:max_targets]:
        xr, xt = U[tr["r"]].astype(float), U[tr["t"]].astype(float)
        h_true = xr.mean() if not normalize else 1.0     # SERGIO : h = moyenne du regulateur
        if normalize:   # donnees bruitees : echelle absolue inconnue -> on travaille en unites relatives
            sx = xr.mean(); sy = np.abs(LAM * xt).mean() if np.abs(xt).mean() > 0 else 1
            x = xr / sx; y = LAM * xt / sy
            # la vraie courbe vit dans les memes unites relatives
        else:
            x, y = xr, LAM * xt

        # 1) etape "neuronale" : champ de production appris par MLP
        order = np.argsort(x)
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="adam",
                           max_iter=3000, early_stopping=True, random_state=rng_seed,
                           n_iter_no_change=40, learning_rate_init=3e-3)
        mlp.fit(x[:, None], y)
        lo, hi = np.percentile(x, [2, 98])
        grid = np.linspace(lo, hi, 120)
        f_nn = mlp.predict(grid[:, None])

        # 2a) regression symbolique libre (GP) sur le champ appris
        sr = SymbolicRegressor(population_size=600, generations=12, tournament_size=15,
                               function_set=("add", "sub", "mul", "div"),
                               parsimony_coefficient=0.01, metric="rmse",
                               stopping_criteria=1e-4, random_state=rng_seed, n_jobs=1, verbose=0)
        sr.fit(grid[:, None], f_nn)
        f_gp = sr.predict(grid[:, None])

        # 2b) selection parmi formes plausibles (Hill/MM/lineaire) sur le champ appris
        best, fam = fit_family(grid, f_nn)
        p = fam[best]["p"]
        fam_fun = {"lineaire": f_lin, "michaelis": f_mm, "hill_act": f_act, "hill_rep": f_rep}[best]
        f_fam = fam_fun(grid, *p)

        # 3) oracle : memes familles ajustees directement sur les donnees brutes (borne de l'identifiabilite)
        best_raw, fam_raw = fit_family(x, y)

        # verite
        if normalize:
            # unites relatives : h=1 (x normalise par sa moyenne) ; amplitude comparee apres mise a l'echelle
            hill = grid**tr["n"] / (1.0 + grid**tr["n"])
            f_true = hill if tr["K"] > 0 else 1 - hill
            h_ref = 1.0
        else:
            f_true = true_curve(grid, tr["K"], h_true, tr["n"])
            h_ref = h_true
        true_type = "hill_act" if tr["K"] > 0 else "hill_rep"

        if normalize:   # echelle absolue inconnue : on compare les formes (courbes normalisees par leur max)
            f_true, f_nn, f_gp, f_fam = shape(f_true), shape(f_nn), shape(f_gp), shape(f_fam)
        row = dict(
            target=tr["t"], reg=tr["r"], true_type=true_type, K_true=abs(tr["K"]), n_true=tr["n"],
            h_true=float(h_ref), x_range=[float(lo), float(hi)],
            nn_fit_nrmse_vs_truth=nrmse(f_true, f_nn),
            gp_expr=str(sr._program), gp_nrmse_vs_truth=nrmse(f_true, f_gp),
            family_selected=best, family_params=p,
            family_nrmse_vs_truth=nrmse(f_true, f_fam),
            type_correct=(best == true_type),
            oracle_selected=best_raw, oracle_params=fam_raw[best_raw]["p"],
        )
        rows.append(row)
        print(f"[{label}] t={tr['t']:>3} r={tr['r']:>3} vrai={true_type:8s} -> choisi={best:9s} "
              f"nrmse(NN)={row['nn_fit_nrmse_vs_truth']:.3f} nrmse(Hill)={row['family_nrmse_vs_truth']:.3f} "
              f"nrmse(GP)={row['gp_nrmse_vs_truth']:.3f}  params={np.round(p,3).tolist()}", flush=True)
    return rows


def summarize(rows, label):
    if not rows:
        return {}
    tc = np.mean([r["type_correct"] for r in rows])
    good = np.mean([r["family_nrmse_vs_truth"] < 0.10 for r in rows])
    gp_good = np.mean([r["gp_nrmse_vs_truth"] < 0.10 for r in rows])
    nhat = [r["family_params"][2] for r in rows if r["family_selected"].startswith("hill")]
    s = dict(label=label, n_cibles=len(rows), type_correct=float(tc),
             courbe_retrouvee_nrmse_lt_10pct_famille=float(good),
             courbe_retrouvee_nrmse_lt_10pct_gp=float(gp_good),
             n_hat_median=float(np.median(nhat)) if nhat else None,
             n_hat_iqr=[float(np.percentile(nhat, 25)), float(np.percentile(nhat, 75))] if nhat else None,
             median_nrmse_nn=float(np.median([r["nn_fit_nrmse_vs_truth"] for r in rows])),
             median_nrmse_famille=float(np.median([r["family_nrmse_vs_truth"] for r in rows])),
             median_nrmse_gp=float(np.median([r["gp_nrmse_vs_truth"] for r in rows])))
    return s


def add_umi_noise(U, S, alpha, seed=1):
    """Bruit technique controle (auteur, distinct du bruit SERGIO par defaut qui donne 99% de zeros) :
    facteur de taille cellulaire log-normal, echantillonnage de Poisson (UMI), dropout dependant de l'expression."""
    rng = np.random.default_rng(seed)
    s_c = rng.lognormal(0, 0.3, size=U.shape[1])
    out = []
    for M in (U, S):
        mu = alpha * s_c[None, :] * M
        cnt = rng.poisson(mu).astype(float)
        p_drop = 0.5 * np.exp(-0.3 * mu)
        cnt[rng.random(cnt.shape) < p_drop] = 0
        out.append(cnt)
    return out


def negative_control(U, truth, seed=0, n=14):
    """Paires (regulateur aleatoire non lie, cible) : le meilleur ajustement de famille explique-t-il quelque chose ?"""
    rng = np.random.default_rng(seed)
    real_r2, ctrl_r2 = [], []
    regs_of = {t["t"]: t["r"] for t in truth}
    for tr in truth[:n]:
        y = LAM * U[tr["t"]]
        for tag, r in (("real", tr["r"]), ("ctrl", int(rng.choice([g for g in range(U.shape[0]) if g not in (tr["t"], tr["r"])])))):
            x = U[r].astype(float)
            best, fam = fit_family(x, y)
            r2 = 1 - fam[best]["rss"] / np.sum((y - y.mean()) ** 2)
            (real_r2 if tag == "real" else ctrl_r2).append(float(r2))
    return real_r2, ctrl_r2


if __name__ == "__main__":
    which = sys.argv[1]            # clean | noisy_mild | noisy_severe | control
    c = np.load(WORK + "/sergio_clean.npz")
    if which == "control":
        real, ctrl = negative_control(c["U"], load_truth())
        print(json.dumps(dict(R2_vrais_regulateurs_median=float(np.median(real)), R2_controle_median=float(np.median(ctrl)),
                              R2_vrais=real, R2_controle=ctrl), indent=1))
        json.dump(dict(real=real, ctrl=ctrl), open(WORK + "/result_control.json", "w"))
        sys.exit()
    if which == "clean":
        U, norm = c["U"], False
    else:
        alpha = 1.0 if which == "noisy_mild" else 0.3
        cu, cs = add_umi_noise(c["U"], c["S"], alpha)
        print(f"[{which}] zeros U = {(cu==0).mean():.2f}, moyenne U = {cu.mean():.2f}")
        lib = (cu + cs).sum(0); lib = lib / np.median(lib)         # normalisation par taille de librairie
        U, norm = cu / np.where(lib > 0, lib, 1), True
    rows = run_pipeline(U, which, normalize=norm)
    s = summarize(rows, which)
    print(json.dumps(s, indent=2))
    json.dump(dict(summary=s, rows=rows), open(f"{WORK}/result_{which}.json", "w"), indent=1)
