# Neural ODE & Symbolic Hill Kinetics for scRNA-seq Regulatory Dynamics

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Verification](https://img.shields.io/badge/JSON%20Audit-183%2F183%20OK-brightgreen.svg)](checks/verify_results.py)
[![Consistency](https://img.shields.io/badge/Physical%20Checks-15%2F15%20PASS-brightgreen.svg)](checks/consistency.py)
[![Pre-registration](https://img.shields.io/badge/Pre--registration-Git%20Sealed-success.svg)](PREREGISTRATION.json)
[![Manuscript](https://img.shields.io/badge/IEEE%20Format-8%20Pages%20Ready-blue.svg)](docs/Article_IEEE_8pages_corrige.docx)

> **Continuous-time non-parametric vector field learning, biophysical Hill kinetics distillation ($n \in \{1, 2, 4\}$), coupled $(u, s)$ transcription-splicing dynamics, and out-of-sample CRISPR Perturb-seq evaluation from single-cell transcriptomics.**

---

## 1. Project Overview

Inferring gene regulatory network (GRN) dynamics from cross-sectional single-cell RNA sequencing (scRNA-seq) snapshots is a classical underdetermined inverse problem. Because cell lysis is destructive, longitudinal trajectories cannot be observed directly. RNA velocity provides local transcriptional direction vectors, but local velocity vectors do not constitute explicit regulatory laws.

This repository provides an end-to-end, cryptographically reproducible pipeline:
1. **Continuous Neural Field Learning:** Fits autonomous Neural Ordinary Differential Equations (Neural ODEs) to RNA velocity fields under leakage-free Voronoi block partition splits (60% train / 20% validation / 20% test).
2. **Support-Constrained Hill Distillation:** Trajectories are integrated from progenitor states, constrained to the data-supported manifold, and distilled into explicit, cooperative Hill kinetic equations. The Hill coefficient is restricted to stoichiometric binding states ($n \in \{1, 2, 4\}$), eliminating parameter collinearity and reducing synthetic parameter error from 44.1% to 0.6%.
3. **Coupled $(u, s)$ Biophysical Model:** Explicitly couples unspliced ($u$) and spliced ($s$) mRNA dynamics:
   $$\frac{du_i}{dt} = \alpha_i(\mathbf{s}) - \beta_i u_i, \qquad \frac{ds_i}{dt} = \beta_i u_i - \gamma_i s_i$$
   resolving the *Ins2* transcription divergence ($R^2 = 0.993$) where uncoupled models fail.
4. **Three Real Mouse Tissues:** Evaluated across mouse Pancreas E15.5 ($N=3{,}696$), Dentate Gyrus ($N=2{,}930$), and Erythroid Gastrulation ($N=9{,}815$).
5. **Synthetic Ground Truth & Noise Robustness:** Evaluated across 15 synthetic ground-truth variants (75 runs) and a 4-level dropout benchmark (10%, 20%, 40%, 60% zero-inflation, 5 seeds each). Distillation demonstrates an advantage under moderate dropout (10%–20%), while remaining statistically tied with finite differences in the severe noise collapse regime (40%–60%).
6. **Autonomous CRISPR Perturb-seq Proof of Concept:** Benchmarks *in-silico* knock-outs against empirical single-cell CRISPR screens (Replogle et al., *Cell* 2022) on a canonical 8-TF hematopoietic circuit, reaching 71.4% out-of-sample sign concordance.
7. **Uncompromising Scientific Honesty:** All metrics are evaluated against pre-registered thresholds sealed to git commits (`PREREGISTRATION.json`). **8 of 29 thresholds are not met and reported without cherry-picking**, documenting exact performance boundaries (e.g., Neural ODE does not outperform affine linear baselines on uncentered cosine in uncorrupted data).

---

## 2. Key Empirical Findings

| Benchmark / Dataset | Key Metric | Result | Pre-registered Target | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Pancreas E15.5** | NODE test cosine (uncentered) | 0.520 ± 0.030 | $\ge 0.50$ | **Met** (Linear: 0.588; no seed beats linear) |
| **Pancreas E15.5** | Centered cosine (validation $\lambda$) | 0.607 ± 0.020 | exploratory | Linear: 0.586 (exploratory amendment) |
| **Dentate Gyrus** | NODE test cosine (5 seeds) | 0.592 ± 0.038 | $\ge 0.50$ | **Met** (Linear: 0.634) |
| **Dentate Gyrus** | Distilled fidelity ($R^2$) | 0.953 ± 0.026 | $\ge 0.90$ | **Met** (5/5 seeds $\ge 0.90$) |
| **Erythroid Gastrulation** | NODE test cosine (5 seeds) | 0.467 ± 0.026 | $\ge 0.50$ | **Not met** (Linear: 0.516; Direct FD: 0.541) |
| **Erythroid Gastrulation** | Distilled fidelity ($R^2$) | 0.939 ± 0.009 | $\ge 0.90$ | **Met** (5/5 seeds $\ge 0.90$) |
| **Perturb-seq (CRISPR)** | Sign concordance (out-of-sample) | 71.4% (40/56 pairs) | baseline > 50% | **Met** (Linear baseline: 41.1%) |
| **Synthetic Base (S1)** | Structure recovery $F_1$ | 0.778 | $\ge 0.70$ | **Met** (Direct FD: 0.887; Weak-SINDy: 0.864) |
| **Dropout 10% (S1-D10)** | Structure $F_1$ (5 seeds) | 0.483 ± 0.087 | exploratory | FD: 0.280 (+0.203 advantage, 4/5 wins) |
| **Dropout 20% (S1-D20)** | Structure $F_1$ (5 seeds) | 0.430 ± 0.130 | exploratory | FD: 0.123 (+0.307 advantage, 4/5 wins) |
| **Dropout 40% (S1-D40)** | Structure $F_1$ (5 seeds) | 0.212 ± 0.099 | exploratory | FD: 0.246 ± 0.090 (within 1 s.d., noise collapse) |
| **Dropout 60% (S1-D60)** | Structure $F_1$ (5 seeds) | 0.237 ± 0.093 | exploratory | FD: 0.161 ± 0.096 (within 1 s.d., noise collapse) |
| **Grönwall Bound (S4)** | Trajectory bound holds (Prop. 1) | 1.000 (70/70 runs) | $\ge 0.95$ | **Met** (14 unwarped variants $\times$ 5 seeds) |
| **Overall Pre-registration** | Confirmed thresholds | **21 / 29 met** | 100% target | **8 thresholds not met** (fully documented) |

---

## 3. Repository Architecture

```
neural_ode_symbolic/
├── checks/                     # Reproducibility, audit & pre-registration suites
│   ├── common.py               # Shared I/O, hash digests, leaf tree traversal
│   ├── consistency.py          # 15 automated physical consistency tests
│   ├── make_report.py          # Generates results report from raw JSON
│   ├── prereg.py               # Cryptographic SHA-256 pre-registration validator
│   └── verify_results.py       # Strict audit of all 183 JSON results (no NaN/Inf)
├── docs/                       # Official manuscript, annexes & technical guides
│   ├── Article_IEEE_8pages_corrige.docx # Master submission manuscript (IEEE 8 pages)
│   ├── ARTICLE_FINAL_IEEE_8PAGES.md     # Full-text Markdown mirror of the paper
│   ├── GUIDE_PYSR.md           # Julia & PySR installation and usage guide
│   ├── GUIDE_TEST.md           # Comprehensive test protocol
│   ├── RAPPORT_COMPLET_PROJET.md # Full technical report
│   ├── annexes/                # Mathematical proofs & algorithm pseudo-code
│   │   ├── algorithmes.md
│   │   ├── demonstrations_mathematiques.md
│   │   └── protocole_reproductibilite.md
│   ├── figures/                # Source figures for documentation
│   └── tables/                 # Source tables for documentation
├── figures_article/            # Publication-grade vector figures (PDF & high-res PNG)
├── project/                    # Phase 1: Bistable Toggle Switch benchmark
│   ├── configs/                # Experiment configurations
│   ├── results/phase1/         # 45 validated JSON runs
│   └── src/                    # Toggle simulation, Neural ODE, Hill identification
├── real_data/                  # Real scRNA-seq pipelines (Pancreas, Dentate Gyrus, Erythroid)
│   ├── configs/                # Configuration files (SHA-256 pinned)
│   ├── results/real/           # 48 validated JSON runs across tissues
│   ├── scripts/                # Execution runners (run_real_erythroid.py, etc.)
│   └── src/                    # Voronoi partitioning, Neural ODE, TRF distillation, coupled (u,s)
├── sergio_validation/          # 100-gene synthetic GRN benchmark with SERGIO
│   ├── SERGIO/                 # SERGIO clean submodule (NumPy 2.x compatible)
│   └── validate.py             # SERGIO simulation & validation runner
├── synthetic/                  # 15 synthetic multi-gene benchmark variants
│   ├── configs/                # Synthetic configurations
│   ├── results/synth/          # 85 validated JSON runs (noise, dropout, size, misspecification)
│   └── src/                    # Synthetic ODE generators, Weak-SINDy, PySR, Grönwall checks
├── datasets.json               # Raw dataset URLs, target paths, and SHA-256 hashes
├── environment.yml             # Conda / Mamba environment specification
├── PREREGISTRATION.json        # Immutable pre-registration ledger sealed to git commits
├── requirements.txt            # Python pip dependencies
├── run_advanced_benchmarks.py  # Coupled (u,s) system & Perturb-seq CRISPR benchmark
├── run_phase1_dropout.py       # 4-level dropout benchmark runner (10%, 20%, 40%, 60%)
├── run_phase2_erythroid.py     # 5-seed Erythroid gastrulation runner
├── setup_all.py                # Environment verification & dataset downloader
├── smoke_test.py               # Non-regression test suite
└── make_figures.py             # Script to regenerate all manuscript figures and tables
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python **3.10 to 3.14** (64-bit).
- Standard C/C++ compiler (for SciPy / torchdiffeq).
- Recommended: ~8 GB RAM, 10 GB free disk space.
- Optional: Julia 1.9+ (only required if running open-ended free-form PySR searches).

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/neural_ode_symbolic.git
cd neural_ode_symbolic

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify environment and download real datasets (Pancreas, Dentate Gyrus)
python setup_all.py
```

*Note on Windows:* To ensure UTF-8 console output, run:
```powershell
$env:PYTHONUTF8=1
```
or use `LANCER_WINDOWS.bat`.

---

## 5. Reproducibility & Verification Suites

This repository is built under a **zero-trust** verification philosophy. Every metric reported in the article can be verified directly against the underlying JSON results without recomputing:

### 1. Audit all 183 experimental result files
Checks for complete JSON integrity, absence of `NaN`/`Inf`, and config hash coherence:
```bash
python checks/verify_results.py --strict
# Expected: 183 fichier(s) examines : 183 OK, 0 avertissement(s), 0 echec(s)
```

### 2. Verify physical consistency invariants
Evaluates 15 physical monotonicities (noise degradation, data quantity scaling, shuffle control collapse, Grönwall error bounds):
```bash
python checks/consistency.py
# Expected: 15 controle(s) : 0 echec(s), 0 non evalue(s)
```

### 3. Verify pre-registration integrity
Verifies all configuration files against the sealed pre-registration ledger:
```bash
python checks/prereg.py verify
# Expected: gel du 2026-11-11T20:24:05Z ; amendements : 2 ; configurations conformes au gel
```

### 4. Run non-regression smoke tests
```bash
python smoke_test.py --skip-pysr
# Expected: ALL TESTS PASSED
```

---

## 6. Running the Experiments

To recompute or explore specific experimental tiers:

### Phase 1 — Dropout Robustness Benchmark
Runs 5 seeds across 4 zero-inflation levels (10%, 20%, 40%, 60%):
```bash
python run_phase1_dropout.py
```

### Phase 2 — Erythroid Gastrulation Pipeline
Runs 5 seeds at full 600 epochs with Voronoi block splits on the erythroid lineage ($N=9{,}815$ cells):
```bash
python run_phase2_erythroid.py
```

### Advanced Benchmarks (Coupled $(u, s)$ & Perturb-seq)
Fits the coupled transcriptional delay system on *Ins2* and runs the CRISPR Perturb-seq benchmark on the 8-TF hematopoietic circuit:
```bash
python run_advanced_benchmarks.py
```

### Regularization Tuning & Exponent Misspecification
```bash
# 5-seed lambda sweep on Pancreas
python experiment_lambda_sweep_5seeds.py

# Exponent misspecification analysis (n = 1.5 and 3.0 outside the grid)
python experiment_discrete_n_misspecification.py
```

### Regenerate All Article Figures & Tables
```bash
python make_figures.py
# Regenerates figures in figures_article/ (PDF & PNG) and summary tables (.csv, .tex)
```

---

## 7. Official Manuscript

The finalized, audited 8-page IEEE manuscript is available in two formats:
- **Microsoft Word (Master Submission):** [`docs/Article_IEEE_8pages_corrige.docx`](docs/Article_IEEE_8pages_corrige.docx)  
  *Formatted in native IEEE two-column layout with native OMML equations, 35 references, and exactly 8 pages.*
- **Markdown Mirror:** [`docs/ARTICLE_FINAL_IEEE_8PAGES.md`](docs/ARTICLE_FINAL_IEEE_8PAGES.md)  
  *Full text, mathematics in LaTeX, and complete empirical tables.*
- **Mathematical Annexes:** [`docs/annexes/`](docs/annexes/)  
  *Proof of Proposition 1 (Grönwall bound), algorithm pseudo-codes, and reproducibility protocol.*

---

## 8. Data Availability & Provenance

All single-cell datasets are publicly available:
1. **Mouse Pancreas E15.5** ($N=3{,}696$ cells): Bastidas-Ponce et al., *Development* 2019. SHA-256: `9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6`.
2. **Mouse Dentate Gyrus** ($N=2{,}930$ cells): Hochgerner et al., *Nature Neuroscience* 2018. SHA-256: `6fbf1ec64e0fba6d18d14bc342b3ba2aab8133972c1e50cb2cb717d4cc336871`.
3. **Mouse Erythroid Gastrulation** ($N=9{,}815$ cells): Pijuan-Sala et al., *Nature* 2019. SHA-256: `f07c3f0c79832981ee7af984cca7fc4974200d571aa6c9ecb2e147653d461283`.
4. **CRISPR Perturb-seq Screens**: Replogle et al., *Cell* 2022 / Norman et al., *Science* 2019. Curated response profiles for the 8-TF circuit are serialized in `real_data/results/real/perturb_seq_benchmark.json`.

---

## 9. Citation

If you use this codebase, methodology, or benchmark in your research, please cite:

```bibtex
@article{neural_ode_symbolic_2026,
  author    = {[Author One] and [Author Two] and [Author Three]},
  title     = {Distilling Cooperative Hill Kinetics from Neural Ordinary Differential Equations on Single-Cell Velocity Fields},
  journal   = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year      = {2026},
  note      = {Under review. Pre-registered repository: https://github.com/your-username/neural_ode_symbolic}
}
```

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
