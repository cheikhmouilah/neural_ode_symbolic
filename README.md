# Neural ODE & Symbolic Regression Coupling for Gene Regulatory Dynamics

**Status:** Python 3.10–3.12 | License: MIT | Smoke Tests: 16/16 PASS | Integrity: 166/166 OK

> **Continuous-Time Non-Parametric Vector Field Learning, Biophysical Hill Kinetics, Coupled $(u, s)$ Transcription-Splicing Modeling, and CRISPR Perturb-seq Validation from Single-Cell Transcriptomics.**

---

## 1. Project Overview

Inferring continuous-time gene regulatory network (GRN) dynamics from cross-sectional single-cell RNA sequencing (scRNA-seq) snapshots is a fundamental inverse problem. Because cell lysis is destructive, longitudinal trajectories cannot be tracked directly. RNA velocity provides local transcriptional derivatives, yet these estimates are noisy, uncoupled, and often biased by transient pre-steady states.

This project implements an end-to-end, cryptographically reproducible framework:
1. **Continuous Neural Vector Field Learning:** Parameterizes velocity fields as autonomous Neural Ordinary Differential Equations (Neural ODEs) fitted via confidence-weighted velocity matching.
2. **Support-Constrained Symbolic Distillation:** Rolls out manifold trajectories and fits explicit, cooperative Hill kinetic equations with biophysical exponent discretization ($n \in \{1, 2, 4\}$) via Non-Negative Least Squares (NNLS) and bounded Trust-Region Reflective (TRF) optimization.
3. **Coupled $(u, s)$ Biophysical Modeling:** Explicitly couples unspliced ($u$) and spliced ($s$) mRNA dynamics:
   $$\frac{du_i}{dt} = \alpha_i(\mathbf{s}) - \beta_i u_i, \qquad \frac{ds_i}{dt} = \beta_i u_i - \gamma_i s_i$$
   resolving the *Ins2* pre-steady-state degradation paradox ($R^2 = 0.993$).
4. **Real-World CRISPR Perturb-seq Benchmark:** Evaluates *in-silico* knock-out predictions against empirical single-cell CRISPR screens (Replogle et al. 2022 / Norman et al. 2019) across 8 core hematopoietic transcription factors.
5. **Mathematical Error Bounds & Cryptographic Reproducibility:** Verifies Proposition 1 Grönwall trajectory error bounds across 70 test runs ($100\%$ pass), supported by 15 automated physical consistency checks and SHA-256 pre-registration.

---

## 2. Repository Structure

```
neural_ode_symbolic/
├── checks/                     # Verification, preregistration & consistency checks
│   ├── consistency.py          # 15 automated physical consistency tests
│   ├── make_report.py          # Generates RAPPORT_RESULTATS.md from raw JSON
│   ├── prereg.py               # SHA-256 hash checks against PREREGISTRATION.json
│   └── verify_results.py       # Audits all 166 JSON result files
├── docs/                       # Comprehensive documentation & scientific papers
│   ├── IEEE_SECTION_IMPLEMENTATION_AND_CONCLUSION.md # Complete IEEE draft
│   └── perturb_seq_results.json# Detailed CRISPR Perturb-seq metrics
├── figures_article/            # Publication-ready vector figures (PNG & PDF)
├── project/                    # Phase 1 & 6 benchmarks (Toggle switch & ablation)
├── real_data/                  # Real scRNA-seq pipeline (Pancreas & Dentate Gyrus)
│   ├── configs/                # Preregistered JSON configurations
│   ├── results/real/           # 166 validated experimental JSON result files
│   └── src/                    # Neural ODE, Hill distillation, coupled (u, s) ODEs
├── synthetic/                  # 15 synthetic multi-gene benchmark variants
├── environment.yml             # Conda / Mamba environment definition
├── requirements.txt            # Python pip dependencies
├── run_advanced_benchmarks.py  # Coupled (u, s) system & Perturb-seq benchmark
├── make_figures.py             # Generates all article figures and CSV tables
├── smoke_test.py               # 16-test non-regression test suite
├── RAPPORT_RESULTATS.md        # Comprehensive 46-block automated results report
└── README.txt                  # Plain-text documentation
```

---

## 3. Installation

**Prerequisites:** Python 3.10–3.12 (CPU only; no GPU cluster required).

### Option A: Using pip & venv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/neural_ode_symbolic.git
cd neural_ode_symbolic

# 2. Create and activate a virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Using Conda / Mamba

```bash
conda env create -f environment.yml
conda activate neural_ode_symbolic
```

---

## 4. How to Run & Generate Reports

### Step 1: Run Fast Smoke Tests (16/16 non-regression tests)
```bash
python smoke_test.py
```

### Step 2: Generate Automated Report and Publication Figures
```bash
# 1. Compile the comprehensive evaluation report (RAPPORT_RESULTATS.md)
python checks/make_report.py

# 2. Render all 6 figures and 3 tables into figures_article/
python make_figures.py
```

### Step 3: Run Advanced Biophysical & CRISPR Benchmarks
```bash
python run_advanced_benchmarks.py
```
*Outputs generated:*
* `figures_article/fig_coupled_us_phase_portraits.png`
* `figures_article/fig_perturb_seq_benchmark.png`
* `docs/perturb_seq_results.json`

### Step 4: Verify Scientific Integrity & Physical Consistency
```bash
# Strict audit of all 166 result files (zero NaNs, exact schema)
python checks/verify_results.py --strict

# Run 15 automated physical guardrails
python checks/consistency.py

# Verify cryptographic SHA-256 configuration hashes
python checks/prereg.py verify
```

---

## 5. Key Results Summary

| Benchmark | Evaluated Metric | Result | Significance |
| :--- | :--- | :---: | :--- |
| **Identifiability ($n \in \{1, 2, 4\}$)** | Parameter Relative Error ($S_2$) | **$0.6\,\%$** | Solves parameter sloppiness; dropped from $44.1\,\%$. |
| **Pancreas E15.5 ($\lambda = 1.0$)** | Test Directional Cosine | **$0.640$** | Decisively beats the linear baseline ($0.588$). |
| **Latent PCA Space ($d=8$)** | Test Directional Cosine | **$0.767$** | Outperforms linear ($0.647$) by $+0.120$ via manifold denoising. |
| **Coupled $(u, s)$ System** | *Ins2* Transcription $R^2$ | **$0.993$** | Fully resolves the *Ins2* pre-steady-state velocity divergence. |
| **CRISPR Perturb-seq Screen** | Knock-out Sign Concordance | **$71.4\,\%$** | Continuous Neural ODE captures causal cascades ($\cos = +0.427$). |
| **Proposition 1 Trajectory Bound** | Grönwall Bound Satisfaction | **$100\,\%$** | Validated across 70 runs ($70/70$ hold). |

---

## 6. Citation

```bibtex
@article{neural_ode_symbolic_2026,
  title={Neural ODE--Symbolic Regression Coupling for Gene Regulatory Dynamics: Implementation, System Design, and Experimental Evaluation},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence / IEEE Access},
  year={2026}
}
```
