================================================================================
NEURAL ODE & SYMBOLIC REGRESSION COUPLING FOR GENE REGULATORY DYNAMICS
================================================================================

Continuous-Time Non-Parametric Vector Field Learning, Biophysical Hill Kinetics,
Coupled (u, s) Transcription-Splicing Modeling, and CRISPR Perturb-seq Validation
from Single-Cell Transcriptomics.

Repository: https://github.com/<your-username>/neural_ode_symbolic


--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
Inferring continuous-time gene regulatory network (GRN) dynamics from cross-
sectional single-cell RNA sequencing (scRNA-seq) snapshots is a fundamental inverse
problem. Because cell lysis is destructive, longitudinal trajectories cannot be
tracked directly. RNA velocity provides local transcriptional derivatives, yet these
estimates are noisy, uncoupled, and often biased by transient pre-steady states.

This project implements an end-to-end, cryptographically reproducible framework:
1. Continuous Neural Vector Field Learning:
   Parameterizes velocity fields as autonomous Neural Ordinary Differential
   Equations (Neural ODEs) fitted via confidence-weighted velocity matching.
2. Support-Constrained Symbolic Distillation:
   Rolls out manifold trajectories and fits explicit, cooperative Hill kinetic
   equations with biophysical exponent discretization (n in {1, 2, 4}) via Non-
   Negative Least Squares (NNLS) and bounded Trust-Region Reflective (TRF) optimization.
3. Coupled (u, s) Biophysical Transcription-Splicing Modeling:
   Explicitly couples unspliced (u) and spliced (s) mRNA dynamics:
       du_i/dt = alpha_i(s) - beta_i * u_i
       ds_i/dt = beta_i * u_i - gamma_i * s_i
   resolving the famous "Ins2 pre-steady-state degradation paradox" (R^2 = 0.993).
4. Real-World CRISPR Perturb-seq Benchmark:
   Evaluates in-silico knock-out predictions against empirical single-cell CRISPR
   screens (Replogle et al. 2022 / Norman et al. 2019) across 8 core hematopoietic TFs.
5. Mathematical Error Bounds & Cryptographic Reproducibility:
   Verifies Proposition 1 Grönwall trajectory error bounds across 70 test runs (100% pass),
   supported by 15 automated physical consistency checks and SHA-256 preregistration.


--------------------------------------------------------------------------------
2. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------
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
└── README.txt                  # This file


--------------------------------------------------------------------------------
3. PREREQUISITES & INSTALLATION
--------------------------------------------------------------------------------
Platform: Windows, Linux, or macOS
Python: 3.10, 3.11, or 3.12 (CPU only; no GPU required)

Option A: Installation with pip & venv (Recommended)
----------------------------------------------------
1. Clone the repository:
   git clone https://github.com/<your-username>/neural_ode_symbolic.git
   cd neural_ode_symbolic

2. Create and activate a virtual environment:
   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install required packages:
   pip install --upgrade pip
   pip install -r requirements.txt

Option B: Installation with Conda / Mamba
----------------------------------------
   conda env create -f environment.yml
   conda activate neural_ode_symbolic


--------------------------------------------------------------------------------
4. HOW TO RUN THE PROJECT (EXECUTION COMMANDS)
--------------------------------------------------------------------------------

Step 1: Run Fast Non-Regression Smoke Tests
------------------------------------------
Verifies all 16 core components (Neural ODEs, Hill identification, scVelo, PySR,
baselines) in under 3 minutes:
   python smoke_test.py

Step 2: Generate Automated Report and Publication Figures
---------------------------------------------------------
Instantly generates the full markdown report and all figures from verified runs:
   # 1. Compile the comprehensive evaluation report (RAPPORT_RESULTATS.md)
   python checks/make_report.py

   # 2. Render all 6 figures and 3 tables into figures_article/
   python make_figures.py

Step 3: Execute Advanced Biophysical & CRISPR Benchmarks
-------------------------------------------------------
Runs the coupled (u, s) transcription-splicing system on Pancreas E15.5 and
evaluates in-silico knock-out predictions against real CRISPR Perturb-seq screens:
   python run_advanced_benchmarks.py

Outputs generated:
- figures_article/fig_coupled_us_phase_portraits.png (Phase plane & nullclines)
- figures_article/fig_perturb_seq_benchmark.png (CRISPR KO comparison)
- docs/perturb_seq_results.json

Step 4: Verify Scientific Integrity and Physical Consistency
------------------------------------------------------------
Audits result artifacts against the preregistered protocol:
   # Strict validation of all 166 result files (zero NaNs, exact schema):
   python checks/verify_results.py --strict

   # Run 15 physical guardrail consistency checks:
   python checks/consistency.py

   # Verify cryptographic SHA-256 configuration hashes:
   python checks/prereg.py verify

Step 5: Inspect Metrics by Experimental Phase
---------------------------------------------
   # Phase 1: 2D Toggle switch benchmark
   python project/scripts/analyze_results.py phase1

   # Phase 6: Noise degradation and subsampling robustness
   python project/scripts/analyze_results.py phase6

   # Real scRNA-seq analysis (Pancreas & Dentate Gyrus):
   python real_data/scripts/analyze_real.py


--------------------------------------------------------------------------------
5. KEY SCIENTIFIC FINDINGS & PERFORMANCE
--------------------------------------------------------------------------------

1. Discrete Hill Cooperativity (n in {1, 2, 4}):
   Restricting the Hill exponent to biophysically realistic values (monomers,
   dimers, tetramers) eliminates the parameter sloppiness valley between K and n,
   reducing median parameter relative error from 44.1% down to 0.6%.

2. Beating the Linear Baseline on Real Pancreas scRNA-seq:
   Increasing magnitude regularization to lambda = 1.0 filters high-frequency
   velocity noise, elevating test directional cosine to 0.640 and decisively
   outperforming the linear baseline (0.588).

3. Multi-Space Representations (Gene vs. Module vs. Latent Space):
   Projecting dynamics into latent PCA space (d=8) removes single-cell sequencing
   dropout noise, reaching a test cosine of 0.767 (+0.120 over linear).
   In module space, symbolic distillation fidelity reaches 0.969 (>= 0.90 met).

4. Coupled (u, s) Transcription-Splicing Modeling:
   Resolves the "Ins2 paradox" where transient unspliced accumulation (Mu >> Ms)
   previously caused uncoupled velocity algorithms (such as dynamo) to project
   aberrant negative directions (-0.932). Splicing rates beta=2.37 h^-1 and
   degradation gamma=2.44 h^-1 are recovered, yielding an Ins2 transcription fit
   of R^2 = 0.993.

5. CRISPR Perturb-seq Real Validation:
   The continuous Neural ODE captures true causal knock-out cascades, achieving
   71.4% sign concordance and a directional cosine of +0.427 against empirical
   CRISPR Perturb-seq screens, whereas linear models fail (41.1%, cos = +0.035).


--------------------------------------------------------------------------------
6. CITATION & CONTACT
--------------------------------------------------------------------------------
If you use this codebase or methodology in your research, please cite:

"Neural ODE–Symbolic Regression Coupling for Gene Regulatory Dynamics:
Implementation, System Design, and Experimental Evaluation",
IEEE Transactions on Pattern Analysis and Machine Intelligence / IEEE Access (2026).

Documentation available in:
docs/IEEE_SECTION_IMPLEMENTATION_AND_CONCLUSION.md
================================================================================
