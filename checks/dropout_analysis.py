"""Print full dropout analysis — all methods, all seeds."""
import json, glob, numpy as np

for variant in ['drop10','drop20','drop40','drop60']:
    files = sorted(glob.glob(f'synthetic/results/synth/synth_{variant}_seed*.json'))
    sym_f1s, fd_f1s, sindy_f1s, node_cos = [], [], [], []
    for f in files:
        r = json.load(open(f))
        sf = r.get('distilled',{}).get('structure',{}).get('F1')
        ff = r.get('direct_fd',{}).get('structure',{}).get('F1')
        wf = r.get('weak_sindy',{}).get('structure',{}).get('F1')
        nc = r.get('node_metrics',{}).get('cos_vs_truth_support')
        if sf is not None: sym_f1s.append(sf)
        if ff is not None: fd_f1s.append(ff)
        if wf is not None: sindy_f1s.append(wf)
        if nc is not None: node_cos.append(nc)
    n = len(sym_f1s)
    if n:
        sm = np.mean(sym_f1s); ss = np.std(sym_f1s)
        fm = np.mean(fd_f1s); fs2 = np.std(fd_f1s)
        wm = np.mean(sindy_f1s) if sindy_f1s else 0
        nm = np.mean(node_cos) if node_cos else 0
        diff = sm - fm
        verdict = "SYM WINS" if diff > 0.05 else ("TIED" if abs(diff) < 0.05 else "FD WINS")
        print(f"{variant} (n={n}): sym={sm:.3f}+/-{ss:.3f}  FD={fm:.3f}+/-{fs2:.3f}  SINDy={wm:.3f}  NODE_cos={nm:.3f}  diff={diff:+.3f} [{verdict}]")
