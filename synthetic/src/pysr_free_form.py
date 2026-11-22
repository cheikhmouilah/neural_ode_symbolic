"""Recherche libre PySR (optionnelle). NON TESTE dans l'environnement de redaction (Julia indisponible) : a valider sur votre machine apres  python setup_all.py --with-pysr.
Meme interface que free_form_gplearn : retourne rappel / precision des variables utilisees, longueur, R2 vs champ neuronal, front de Pareto par equation."""
import os, re, numpy as np


def free_form_pysr(Ztr, Ftr, Zte, Fnode_te, eqs, fcfg, seed):
    from pysr import PySRRegressor
    d = Ztr.shape[1]; out = []
    for i in range(d):
        m = PySRRegressor(niterations=fcfg.get('pysr_iterations', 40), binary_operators=['+', '-', '*', '/'], unary_operators=[], maxsize=fcfg.get('pysr_maxsize', 15), random_state=seed + i,
                          deterministic=True, parallelism='serial', progress=False, verbosity=0, model_selection='best',
                          temp_equation_file=False, output_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'pysr_runs'))
        m.fit(Ztr[:2000], Ftr[:2000, i]); best = m.get_best(); pred = m.predict(Zte); expr = str(best['equation']); used = {int(x) for x in re.findall(r'x(\d+)', expr)}
        true_set = {t['j'] for t in eqs[i]['terms']} | {i}
        out.append(dict(i=i, program=expr, length=int(best['complexity']), r2_vs_node=float(1 - np.sum((pred - Fnode_te[:, i]) ** 2) / np.sum((Fnode_te[:, i] - Fnode_te[:, i].mean()) ** 2)), used=sorted(used),
                        var_recall=len(used & true_set) / len(true_set), var_precision=(len(used & true_set) / len(used)) if used else 0.0,
                        pareto=[dict(complexity=int(r['complexity']), loss=float(r['loss'])) for _, r in m.equations_.iterrows()]))
    return dict(equations=out, mean_var_recall=float(np.mean([o['var_recall'] for o in out])), mean_var_precision=float(np.mean([o['var_precision'] for o in out])),
                mean_length=float(np.mean([o['length'] for o in out])), mean_r2_vs_node=float(np.mean([o['r2_vs_node'] for o in out])))
