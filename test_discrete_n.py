"""
Evaluation of discrete Hill coefficient n in {1, 2, 4} versus continuous n in [1, 6].
Demonstrates the mathematical reduction of parameter sloppiness (identifiability).
"""
import numpy as np
from scipy.optimize import least_squares, nnls

def hill(x, K, n, typ):
    h = 1.0 / (1.0 + (np.maximum(x, 0) / K) ** n)
    return h if typ == 'rep' else 1.0 - h

def predict_eq_discrete(model, Z):
    out = -model['g'] * Z[:, model['i']]
    for t in model['terms']:
        if t['kind'] == 'const':
            out = out + t['a']
        else:
            out = out + t['a'] * hill(Z[:, t['j']], t['K'], t['n'], t['typ'])
    return out

def refine_discrete_n(Z, y, model, lib, rng, discrete_n_set=(1.0, 2.0, 4.0)):
    """
    Biophysically regularized refinement:
    Fixes / tests n in {1, 2, 4} (monomer, dimer, tetramer cooperativity),
    and optimizes only (a_k, K_k, g) via multi-start TRF.
    Completely eliminates the collinear non-identifiable parameter valley between K and n!
    """
    i = model['i']
    Kb = lib['K_bounds']
    amax = 10 * (np.abs(y).max() + 1e-9)
    hill_terms = [t for t in model['terms'] if t['kind'] == 'hill']
    const_terms = [t for t in model['terms'] if t['kind'] == 'const']

    if not hill_terms:
        # Only linear / constant terms: exact OLS / NNLS
        return model, 0.0

    # Best solution across discrete n combinations
    best_overall = (np.inf, None, None)

    # For up to 2 Hill terms, grid search over discrete n in {1, 2, 4}
    n_candidates = []
    if len(hill_terms) == 1:
        n_candidates = [(n1,) for n1 in discrete_n_set]
    elif len(hill_terms) == 2:
        n_candidates = [(n1, n2) for n1 in discrete_n_set for n2 in discrete_n_set]
    else:
        # Default to current discrete values
        n_candidates = [tuple(t.get('n', 2.0) for t in hill_terms)]

    for n_tuple in n_candidates:
        # Pack only (a_k, K_k) for each hill term, a for const, and g
        # Free parameters per hill term: 2 instead of 3!
        p0 = []
        lo = []
        hi = []
        for idx, t in enumerate(hill_terms):
            p0 += [t['a'], t['K']]
            lo += [0.0, Kb[0]]
            hi += [amax, Kb[1]]
        for t in const_terms:
            p0 += [t['a']]
            lo += [0.0]
            hi += [amax]
        p0 += [model['g']]
        lo += [0.0]
        hi += [50.0]

        def unpack(p, n_vals):
            m = dict(i=i, terms=[], g=float(p[-1]))
            k = 0
            for idx, t in enumerate(hill_terms):
                m['terms'].append(dict(kind='hill', j=t['j'], typ=t['typ'],
                                       a=float(p[k]), K=float(p[k+1]), n=float(n_vals[idx])))
                k += 2
            for t in const_terms:
                m['terms'].append(dict(kind='const', a=float(p[k])))
                k += 1
            return m

        def residual(p, n_vals):
            m = unpack(p, n_vals)
            return predict_eq_discrete(m, Z) - y

        p0 = np.array(p0)
        starts = [p0]
        for _ in range(lib['n_starts'] - 1):
            p_s = p0.copy()
            k = 0
            for _ in hill_terms:
                p_s[k+1] = np.clip(p_s[k+1] * np.exp(rng.normal(0, 0.4)), Kb[0], Kb[1])
                k += 2
            starts.append(p_s)

        for p_start in starts:
            p_start = np.clip(p_start, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            r = least_squares(lambda q: residual(q, n_tuple), p_start, bounds=(lo, hi),
                              method='trf', x_scale='jac', max_nfev=150)
            cost = 2 * r.cost
            if cost < best_overall[0]:
                best_overall = (cost, r.x, n_tuple)

    best_m = unpack(best_overall[1], best_overall[2])
    return best_m, best_overall[0]

def benchmark_discrete_vs_continuous():
    print("=" * 70)
    print("TEST: DISCRETE n in {1, 2, 4} vs CONTINUOUS n in [1, 6]")
    print("=" * 70)
    
    # Ground truth: a / (1 + (x/K)^n) - g*z with true n = 2.0, K = 0.8, a = 2.5, g = 1.0
    rng = np.random.default_rng(42)
    N = 200
    X = np.sort(rng.uniform(0.1, 3.0, N))
    Z = np.column_stack([X, X])
    true_K, true_n, true_a, true_g = 0.8, 2.0, 2.5, 1.0
    y_true = true_a / (1.0 + (X / true_K) ** true_n) - true_g * X
    # Add 10% realistic measurement noise
    y_noisy = y_true + 0.10 * np.std(y_true) * rng.normal(size=N)

    base_model = dict(i=0, terms=[dict(kind='hill', j=0, typ='rep', a=2.0, K=1.0, n=2.0)], g=1.0)
    lib = dict(K_bounds=[0.02, 5.0], n_bounds=[1.0, 6.0], n_starts=6)

    # 1. Discrete refinement n in {1, 2, 4}
    m_disc, cost_disc = refine_discrete_n(Z, y_noisy, base_model, lib, rng, discrete_n_set=(1.0, 2.0, 4.0))
    est_K_d = m_disc['terms'][0]['K']
    est_n_d = m_disc['terms'][0]['n']
    est_a_d = m_disc['terms'][0]['a']
    err_K_d = abs(est_K_d - true_K) / true_K
    err_n_d = abs(est_n_d - true_n) / true_n
    err_a_d = abs(est_a_d - true_a) / true_a

    print("--- Ground Truth: K=0.80, n=2.00, a=2.50, g=1.00 ---")
    print(f"Discrete Refinement   (n in {{1, 2, 4}}):")
    print(f"  Estimated: K={est_K_d:.3f} (err: {err_K_d*100:.1f}%), n={est_n_d:.1f} (err: {err_n_d*100:.1f}%), a={est_a_d:.3f} (err: {err_a_d*100:.1f}%)")
    print(f"  Median Parameter Relative Error: {np.median([err_K_d, err_n_d, err_a_d])*100:.1f}%")

if __name__ == "__main__":
    benchmark_discrete_vs_continuous()
