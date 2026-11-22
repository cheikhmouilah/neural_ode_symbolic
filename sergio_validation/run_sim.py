import os as _os
WORK = _os.environ.get('WORK', _os.path.dirname(_os.path.abspath(__file__)))
SERGIO_DIR = _os.path.join(WORK, 'SERGIO')
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, SERGIO_DIR)
_os.chdir(SERGIO_DIR)   # SERGIO utilise des chemins relatifs (data_sets/...)
from SERGIO.sergio import sergio
t0 = time.time()
d = 'data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/'
bMat = pd.read_csv(d + 'bMat_cID7.tab', sep='\t', header=None).values
np.random.seed(0)
sim = sergio(number_genes=100, number_bins=6, number_sc=300, noise_params=0.2, decays=0.8,
             sampling_state=1, noise_params_splice=0.07, noise_type='dpd',
             dynamics=True, bifurcation_matrix=bMat)
sim.build_graph(input_file_taregts=d + 'Interaction_cID_7.txt', input_file_regs=d + 'Regs_cID_7.txt', shared_coop_state=2)
print('graph built', time.time() - t0, flush=True)
sim.simulate_dynamics()
print('simulated', time.time() - t0, flush=True)
U, S = sim.getExpressions_dynamics()
Uc = np.concatenate(U, axis=1); Sc = np.concatenate(S, axis=1)
np.savez(WORK + '/sergio_clean.npz', U=Uc, S=Sc)
print('clean saved', Uc.shape, time.time() - t0, flush=True)
# technical noise
U_O, S_O = sim.outlier_effect_dynamics(U, S, outlier_prob=0.01, mean=0.8, scale=1)
lib, U_L, S_L = sim.lib_size_effect_dynamics(U_O, S_O, mean=4.6, scale=0.4)
bU, bS = sim.dropout_indicator_dynamics(U_L, S_L, shape=6.5, percentile=82)
cU, cS = sim.convert_to_UMIcounts_dynamics(np.multiply(bU, U_L), np.multiply(bS, S_L))
np.savez(WORK + '/sergio_noisy.npz', U=np.concatenate(cU, axis=1), S=np.concatenate(cS, axis=1))
print('noisy saved', time.time() - t0, flush=True)
