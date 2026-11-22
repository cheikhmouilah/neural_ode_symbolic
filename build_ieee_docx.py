"""
Generates a publication-grade IEEE formatted Microsoft Word document (.docx)
containing Section IV (Implementation and System Design) and Section V (Conclusion and Future Perspectives).
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOC_PATH = r"c:\neural_ode_symbolic\docs\IEEE_SECTION_IMPLEMENTATION_AND_CONCLUSION.docx"
FIG_DIR = r"c:\neural_ode_symbolic\figures_article"

def create_document():
    doc = docx.Document()
    
    # Page setup: Standard IEEE Letter margins (0.75 in = 54 pt)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    # Set default style to Times New Roman
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0x1a, 0x1a, 0x1a)
    
    # Helper functions
    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{m}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_cell_shading(cell, color_hex):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    def add_sec_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        return p

    def add_subsec_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.italic = True
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        return p

    def add_subsubsec_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        return p

    def add_body_p(text, bold_prefix="", indent=True):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.2)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = 'Times New Roman'
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        return p

    def add_code_block(code_text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_shading(cell, "F4F6F9")
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        r = p.add_run(code_text)
        r.font.name = 'Consolas'
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def add_figure(img_rel_path, caption_text, width=Inches(6.5)):
        full_path = os.path.join(FIG_DIR, img_rel_path)
        if os.path.exists(full_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(3)
            p_img.paragraph_format.keep_with_next = True
            run_img = p_img.add_run()
            run_img.add_picture(full_path, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(8)
            p_cap.paragraph_format.line_spacing = 1.1
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = 'Times New Roman'
            run_cap.font.size = Pt(8.5)
            run_cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # -------------------------------------------------------------
    # DOCUMENT HEADER
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    run_t = p_title.add_run("Neural ODE–Symbolic Regression Coupling for Gene Regulatory Dynamics:\nImplementation, System Design, and Experimental Evaluation")
    run_t.bold = True
    run_t.font.name = 'Times New Roman'
    run_t.font.size = Pt(16)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("Comprehensive IEEE Transactions Section IV & Section V Scientific Documentation\nArtifact Directory: c:\\neural_ode_symbolic\\docs\\")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(9.5)
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # -------------------------------------------------------------
    # IV. IMPLEMENTATION AND SYSTEM DESIGN
    # -------------------------------------------------------------
    add_sec_heading("IV. IMPLEMENTATION AND SYSTEM DESIGN")
    
    # A. Implementation Objectives
    add_subsec_heading("A. Implementation Objectives")
    add_body_p(
        "The software implementation addresses the challenging inverse problem of identifying continuous, biophysically interpretable "
        "gene regulatory network (GRN) dynamical laws from cross-sectional single-cell RNA sequencing (scRNA-seq) measurements. "
        "Because single-cell sequencing is an intrinsically destructive protocol, individual cells cannot be tracked longitudinally over time [1], [2]. "
        "RNA velocity circumvents this limitation by estimating instantaneous transcriptional derivatives from the ratio of unspliced to mature spliced mRNA counts, "
        "yet these velocities are noisy, biased, and discrete. The implementation was specifically engineered to satisfy five major technical and scientific goals:",
        bold_prefix=""
    )
    add_body_p("1) Non-Parametric Continuous Vector Field Modeling: Avoiding rigid initial functional assumptions by parameterizing the velocity field as an autonomous Neural Ordinary Differential Equation (Neural ODE) [8] capable of capturing complex non-linearities directly in gene expression space.", indent=False)
    add_body_p("2) Support-Constrained Symbolic Distillation: Translating the deep black-box Neural ODE into explicit, cooperative Hill kinetic equations using non-negative least squares (NNLS) and bounded trust-region reflective (TRF) optimization restricted to the observed manifold support [3].", indent=False)
    add_body_p("3) Prevention of Data Leakage via Spatial Partitioning: Developing a 60-cluster k-means Voronoi neighborhood splitting in PCA space to prevent information contamination caused by scVelo's kNN moment smoothing across training and testing partitions.", indent=False)
    add_body_p("4) Guaranteed Trajectory Bounds via Grönwall Formalization: Verifying that reconstructed symbolic trajectories remain mathematically bounded by the Neural ODE approximation error via an empirical Proposition 1 validation pipeline.", indent=False)
    add_body_p("5) End-to-End Cryptographic Reproducibility: Enforcing pre-registered thresholds (PREREGISTRATION.json), automated SHA-256 data validation, and 15 defensive consistency checks that halt the pipeline if physical laws (such as monotonic degradation under noise) are violated.", indent=False)

    # B. Overall System Architecture
    add_subsec_heading("B. Overall System Architecture")
    add_body_p(
        "The overall computational architecture of the framework is structured into four interconnected and modular tiers, "
        "illustrated in Fig. 1. The design strictly isolates data ingestion, deep learning surrogates, symbolic distillation, and formal dynamic verification."
    )
    
    add_figure("fig1_overall_architecture.png", "Fig. 1. End-to-End System Architecture: From Single-Cell Transcripts to Interpretable Regulatory Laws. Tier 1 performs cryptographic verification, scVelo kinetic preprocessing, and spatial block splitting. Tier 2 optimizes an autonomous Neural ODE velocity field. Tier 3 executes manifold trajectory sampling and structured Hill regression. Tier 4 conducts formal Grönwall bound checks, stability analysis, and reproducible reporting.")

    add_body_p(
        "The operational tiers execute sequentially through clearly defined data interfaces:",
        indent=True
    )
    add_body_p("• Tier 1 (Data Ingestion & Manifold Preprocessing): Raw single-cell count matrices (.h5ad) are loaded, cryptographically validated against pre-registered SHA-256 digests, normalized, and preprocessed with scVelo deterministic splicing kinetics. Genes are ranked by the variance of mature spliced counts (Ms), selecting the top d=8 genes without human bias. Cells are clustered into 60 Voronoi blocks in PCA space (X_pca[:, :10]), partitioning cells into 60% training, 20% validation, and 20% testing sets while preventing kNN leakage.", indent=False)
    add_body_p("• Tier 2 (Continuous Vector Field Learning): An autonomous multilayer perceptron f_theta(z) without explicit time input is trained to minimize a dimension-free, confidence-weighted velocity matching loss L_vel with magnitude regularization. Dormand-Prince (dopri5) integration provides adaptive-step rollouts.", indent=False)
    add_body_p("• Tier 3 (Symbolic Distillation & Bounded Refinement): Synthetic trajectories are rolled out from identified progenitor clusters (e.g., Ngn3 low EP in pancreas). A 5-nearest-neighbor density filter discards out-of-support states. An analytical Hill candidate library is screened via forward selection (NNLS) and refined by bounded multi-start TRF nonlinear solvers, selecting Pareto-optimal models with a 5% tolerance rule.", indent=False)
    add_body_p("• Tier 4 (Dynamical Analysis & Reproducibility Guardrails): Computes analytical Jacobians, locates fixed points via multi-start Powell hybrids, classifies Lyapunov stability, simulates in-silico gene deletions, verifies Grönwall trajectory error bounds across 70 runs, and runs 15 automated consistency checks.", indent=False)

    # C. Software Environment and Technologies
    add_subsec_heading("C. Software Environment and Technologies")
    add_body_p(
        "The computational framework was developed and executed using Python 3.14.2 on a 64-bit Windows 11 platform "
        "(8 CPU cores, 1 thread per worker process to avoid deadlocks). Tensor operations are CPU-optimized, "
        "requiring no dedicated GPU clusters. Table I details the software environment, library versions, and their technical roles."
    )

    # TABLE I
    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(2)
    p_t1.paragraph_format.keep_with_next = True
    r_t1 = p_t1.add_run("TABLE I\nSOFTWARE ENVIRONMENT AND COMPONENT TAXONOMY")
    r_t1.bold = True
    r_t1.font.name = 'Times New Roman'
    r_t1.font.size = Pt(9)

    table1 = doc.add_table(rows=1, cols=4)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table1.rows[0].cells
    hdr_titles = ["Module / Library", "Version", "Main Function", "Role and Scientific Justification"]
    for idx, title in enumerate(hdr_titles):
        hdr_cells[idx].text = title
        set_cell_shading(hdr_cells[idx], "EAECEE")
        set_cell_margins(hdr_cells[idx], top=80, bottom=80, left=100, right=100)
        p = hdr_cells[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.runs[0]
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)

    t1_data = [
        ("Python", "3.14.2", "Core runtime environment", "Base runtime supporting modern typing, process isolation, and subprocess execution."),
        ("PyTorch", ">= 2.14.0", "Automatic differentiation", "Parameterizes the Neural ODE vector field f_theta(z); computes analytical autograd gradients."),
        ("torchdiffeq", ">= 0.2.2", "Differentiable ODE solvers", "Provides Dormand-Prince (dopri5) adaptive integration and fixed-step Runge-Kutta (rk4)."),
        ("NumPy", ">= 2.4.0", "Numerical array manipulation", "Vectorized array calculations, linear algebra, random number generators, and matrix norms."),
        ("SciPy", ">= 1.17.0", "Mathematical optimization", "Implements nnls for term screening, least_squares (trf) for refinement, and solve_ivp."),
        ("Scanpy", ">= 1.11.0", "Single-cell genomics toolkit", "HVG selection via Seurat flavor on log1p counts; principal component analysis (PCA)."),
        ("scVelo", "0.3.4", "RNA velocity modeling", "Moments computation (Ms, Mu), deterministic velocity estimation, and confidence weights."),
        ("AnnData", ">= 0.12.0", "Genomic data structures", "Annotated data matrix storage holding counts, layers (Ms, velocity), and cell/gene metadata."),
        ("scikit-learn", ">= 1.8.0", "Machine learning utilities", "Neighborhood block splitting (KMeans), manifold density filtering (NearestNeighbors)."),
        ("PySR", ">= 2.7.0", "Free-form symbolic regression", "Julia-powered genetic programming exploring unconstrained algebraic operators (+, -, *, /)."),
        ("gplearn", "0.4.3", "Genetic programming baseline", "Baseline evolutionary symbolic regression tool for tree-based equation recovery."),
        ("PySINDy", "2.1.0", "Sparse system identification", "Baseline implementation of Weak-SINDy using integrated Galerkin test functions."),
        ("SymPy", ">= 1.14.0", "Computer algebra system", "Symbolic equation parsing, LaTeX rendering, and mathematical expression formatting."),
        ("Pandas", ">= 3.0.0", "Tabular data analysis", "Formatting metadata, structuring benchmarking tables, and exporting CSV/LaTeX outputs."),
        ("Matplotlib", ">= 3.10.0", "Scientific visualization", "Production of high-resolution vector figures, phase portraits, and Pareto calibration curves.")
    ]

    for row_idx, data in enumerate(t1_data):
        row_cells = table1.add_row().cells
        bg = "FFFFFF" if row_idx % 2 == 0 else "F8F9FA"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = text
            set_cell_shading(row_cells[col_idx], bg)
            set_cell_margins(row_cells[col_idx], top=60, bottom=60, left=90, right=90)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 1] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # D. Detailed Implementation of the Modules
    add_subsec_heading("D. Detailed Implementation of the Modules")
    add_body_p(
        "The software architecture is decoupled into specialized Python modules. Each module enforces strict input/output contracts, "
        "as summarized in the functional dependency map of Fig. 3."
    )

    add_figure("fig3_module_interaction.png", "Fig. 3. Functional Module Decomposition, Interfaces, and Data Contracts. Module 1 pre-processes scRNA-seq matrices; Module 2 fits the Neural ODE field; Module 3 distills Hill equations via NNLS and TRF; Module 4 analyzes fixed points and Lyapunov spectra; Module 5 verifies Grönwall bounds on synthetic benchmarks; Module 6 audits reproducibility across 166 JSON artifacts.")

    # 1) Module 1: real_data.py
    add_subsubsec_heading("1) Module 1: Data Ingestion, Splicing Kinetics, and Spatial Block Splitting (real_data.py)")
    add_body_p(
        "Module 1 ingests raw .h5ad files, executes cryptographic verification, calculates splicing moments, and partitions cells. "
        "It verifies input files against datasets.json: for mouse pancreas E15.5 (endocrinogenesis_day15.h5ad, 3696 cells), the SHA-256 digest "
        "must match 9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6; for dentate gyrus (10X43_1.h5ad, ~3400 cells), "
        "it must match 6fbf1ec64e0fba6d18d14bc342b3ba2aab8133972c1e50cb2cb717d4cc336871. "
        "Deterministic velocity is calculated via v_i = beta_i * u_i - gamma_i * s_i. The state space is determined by selecting the d=8 genes "
        "with maximal variance in mature spliced counts (Ms), yielding Ghrl, Gnas, Ins2, Nnat, Chga, Clu, Isl1, and Cpe. "
        "To avoid kNN moment smoothing leakage, Algorithm 1 clusters cells into 60 Voronoi blocks in PCA space (X_pca[:, :10]), "
        "allocating blocks in bulk to 60% training, 20% validation, and 20% testing partitions."
    )
    add_code_block(
        "Algorithm 1: Manifold-Preserving Neighborhood Block Partitioning\n"
        "Input: Cell PCA coordinates X_pca in R^{N x 10}, Block count B = 60, Split fractions [0.6, 0.2, 0.2], Seed S\n"
        "Output: Cell assignment vector split in {0: Train, 1: Val, 2: Test}^N\n"
        "1: blocks = KMeans(n_clusters=B, n_init=3, random_state=S).fit_predict(X_pca)\n"
        "2: perm = DefaultRNG(seed=S).permutation(B)\n"
        "3: N_tr = round(0.6 * B); N_va = round(0.2 * B); N_te = B - N_tr - N_va\n"
        "4: role = {perm[k]: 0 if k < N_tr else (1 if k < N_tr + N_va else 2) for k in range(B)}\n"
        "5: return np.array([role[b] for b in blocks])"
    )

    # 2) Module 2: real_node.py
    add_subsubsec_heading("2) Module 2: Continuous Neural Vector Field Learning via Velocity Matching (real_node.py & node.py)")
    add_body_p(
        "Module 2 parameterizes the autonomous vector field f_theta(z) as a multilayer perceptron: "
        "f_theta(z) = W_3 * tanh(W_2 * tanh(W_1 * z + b_1) + b_2) + b_3, where W_1 in R^{64 x d}, W_2 in R^{64 x 64}, and W_3 in R^{d x 64}, "
        "totaling 5,256 trainable weights for d=8. The objective function balances confidence-weighted velocity matching and magnitude regularization:\n"
        "L(theta) = (1 / (nu * |I_tr|)) * sum_{i in I_tr} w_i * ||v_i - f_theta(z_i)||^2 + (lambda / (nu * |I_tr|)) * sum_{i in I_tr} ||f_theta(z_i)||^2\n"
        "where nu = (1 / |I_tr|) * sum w_i * ||v_i||^2 + 1e-12 ensures a dimension-free loss. Training employs Adam (lr=0.003), "
        "gradient clipping at 5.0, 80% random subsampling per seed to prevent density spike overfitting, and early stopping with patience=80."
    )

    # 3) Module 3: real_identify.py
    add_subsubsec_heading("3) Module 3: Trajectory Distillation and Structured Hill Kinetic Identification (real_identify.py)")
    add_body_p(
        "Module 3 extracts explicit differential equations from the learned neural field. "
        "First, sample_Df integrates f_theta from progenitor states over T=5.0 to generate candidate points. "
        "A kNN filter (k=5) measures local manifold distances; points exceeding the 95th-percentile cutoff are discarded, "
        "yielding M=4000 support-constrained points. Next, a candidate library of activating and repressing Hill terms is constructed:\n"
        "H^-(x; K, n) = 1 / (1 + (x/K)^n),   H^+(x; K, n) = 1 - H^-(x; K, n)\n"
        "Greedy forward selection adds up to k_max=3 terms per gene, optimizing linear amplitudes a_k >= 0 and degradation rate gamma_i >= 0 via NNLS. "
        "Nonlinear parameters (a_k, K_k, n_k, gamma_i) are subsequently refined using bounded multi-start Trust-Region Reflective (TRF) least squares "
        "(n_starts=6, K in [0.02, 5.0], n in [1.0, 6.0]). Model selection follows a Pareto frontier balancing complexity Omega = 1 + sum(3 if Hill else 1) + 1 "
        "against validation error, selecting the minimal-complexity model within 5% of minimal validation error (Algorithm 2)."
    )
    add_code_block(
        "Algorithm 2: Structured Hill Kinetic Selection with Multi-Start TRF Refinement\n"
        "Input: Distilled data Z_tr, y_tr, Validation data Z_val, y_val, Candidate library Phi, k_max = 3, tau = 0.05\n"
        "Output: Selected model M_opt, Pareto frontier P\n"
        "1: P = [(k=0, Omega=1, MSE_val=MSE(fit_nnls([], y_tr)))]\n"
        "2: S = []\n"
        "3: for k from 1 to k_max do\n"
        "4:     best_c = argmin_{c not in S} NNLS_RSS([Phi[:, S U {c}], -Z_tr[:, i]], y_tr)\n"
        "5:     S = S U {best_c}\n"
        "6:     m_refined = MultiStartTRF(Z_tr, y_tr, S, n_starts=6, bounds=(lo, hi))\n"
        "7:     P.append((k=k, Omega=n_params(m_refined), MSE_val=MSE(m_refined, Z_val)))\n"
        "8: end for\n"
        "9: M_opt = min({p in P | p.MSE_val <= (1 + tau) * min(p.MSE_val for p in P)}, key=lambda p: (p.Omega, p.MSE_val))\n"
        "10: return M_opt, P"
    )

    # 4) Module 4: real_dyn.py
    add_subsubsec_heading("4) Module 4: Dynamical Systems Analysis, Equilibria, and Stability (real_dyn.py)")
    add_body_p(
        "Module 4 evaluates phase space topology. Fixed points satisfying F_sym(z*) = 0 are identified using Powell's hybrid method "
        "(scipy.optimize.root) initialized from 120 multi-start guesses. Analytical Jacobians J_ij = dF_i / dz_j are approximated via central finite differences (eps=1e-5). "
        "Equilibria are classified as asymptotically stable if and only if all eigenvalues satisfy max_j Re(lambda_j) < 0. "
        "Distances between stable attractors and empirical cell cluster centroids are evaluated to assess terminal fate alignment."
    )

    # 5) Module 5: synth_methods.py
    add_subsubsec_heading("5) Module 5: Synthetic Benchmark Suite, Weak-SINDy, and Grönwall Bound Verification (synth_methods.py)")
    add_body_p(
        "Module 5 benchmarks the framework against ground-truth systems across 15 experimental variants (noise, dropout, dimension, pseudotime). "
        "It implements Weak-SINDy by integrating noisy trajectories against compact test functions phi_s(t) = [(t-a)(b-t)]^2 over sliding windows (w=5, stride=1), "
        "evaluating integrals by parts to avoid direct numerical differentiation. "
        "Furthermore, prop1_check computes Proposition 1: ||z_sym(t) - z_true(t)|| <= (eps / L) * (e^{Lt} - 1) + delta_0 * e^{Lt}, "
        "measuring field error eps and Lipschitz constant L = max_t sigma_max(nabla F_hat). Across 70 benchmark runs, the bound is verified with 100% satisfaction."
    )

    # 6) Module 6: checks/ & Verification
    add_subsubsec_heading("6) Module 6: Multi-Layer Verification and Cryptographic Preregistration (checks/)")
    add_body_p(
        "Module 6 safeguards scientific integrity. prereg.py verifies SHA-256 hashes of all configuration files against PREREGISTRATION.json, "
        "aborting if thresholds are modified without recorded amendments. consistency.py evaluates 15 automated physical guardrails "
        "(e.g., verifying that noise degradation from sigma=0.02 to 0.20 does not artificially inflate field cosine by > 0.03). "
        "verify_results.py inspects all 166 JSON result files for schema validity and unauthorized NaNs, and make_report.py compiles RAPPORT_RESULTATS.md."
    )

    # E. Overall Workflow and Execution Process
    add_subsec_heading("E. Overall Workflow and Execution Process")
    add_body_p(
        "The end-to-end execution workflow is orchestrated via run_everything.py. "
        "Fig. 2 presents the detailed flowchart illustrating conditional branching, defensive integrity checkpoints, and output artifacts."
    )

    add_figure("fig2_system_workflow.png", "Fig. 2. Execution Flowchart and Guarded Verification Protocol. From configuration ingestion and SHA-256 validation to Neural ODE fitting, manifold distillation, Pareto selection, 15 automated consistency checks, and publication-ready reporting.")

    add_body_p(
        "Execution proceeds through seven deterministic stages:",
        indent=True
    )
    add_body_p("1. Configuration and Data Integrity Gate: Evaluates pre-registered SHA-256 configuration hashes and datasets.json. Any mismatch aborts execution.", indent=False)
    add_body_p("2. Transcript Kinetic Preprocessing: Computes spliced/unspliced moments (Ms, Mu) and establishes 60-cluster k-means Voronoi neighborhood splitting.", indent=False)
    add_body_p("3. Autonomous Neural Vector Field Optimization: Trains f_theta(z) under Adam with early stopping monitored on the validation blocks.", indent=False)
    add_body_p("4. Manifold Trajectory Distillation: Generates candidate states D_f along simulated rollouts, filtering out-of-support points via 95th-percentile kNN pruning.", indent=False)
    add_body_p("5. Bounded Symbolic Model Selection: Fits activating/repressing Hill kinetics through NNLS and 6-start TRF refinement, selecting Pareto-optimal models with 5% tolerance.", indent=False)
    add_body_p("6. Automated Consistency and Grönwall Auditing: Evaluates 15 consistency checks in consistency.py and verifies Proposition 1 trajectory bounds across 70 test runs.", indent=False)
    add_body_p("7. Publication Figure and Report Compilation: Parses 166 JSON artifacts to produce RAPPORT_RESULTATS.md and vector plots in figures_article/.", indent=False)

    # F. Experimental Results and Performance Evaluation
    add_subsec_heading("F. Experimental Results and Performance Evaluation")
    add_body_p(
        "Experimental evaluations were conducted across real single-cell transcriptomes (mouse pancreas E15.5 and dentate gyrus), "
        "a 2D bistable toggle switch, and 15 multi-gene synthetic benchmark variants (75 runs total). "
        "All quantitative results are compiled directly from verified JSON artifacts in Table III, with hyperparameter settings in Table II."
    )

    # TABLE II
    p_t2 = doc.add_paragraph()
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.space_before = Pt(8)
    p_t2.paragraph_format.space_after = Pt(2)
    p_t2.paragraph_format.keep_with_next = True
    r_t2 = p_t2.add_run("TABLE II\nEXPERIMENTAL PARAMETERS AND HYPERPARAMETER CONFIGURATION")
    r_t2.bold = True
    r_t2.font.name = 'Times New Roman'
    r_t2.font.size = Pt(9)

    table2 = doc.add_table(rows=1, cols=4)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr2_cells = table2.rows[0].cells
    hdr2_titles = ["Parameter / Setting", "Real scRNA-seq (Pancreas)", "Synthetic Benchmarks", "Scientific Description"]
    for idx, title in enumerate(hdr2_titles):
        hdr2_cells[idx].text = title
        set_cell_shading(hdr2_cells[idx], "EAECEE")
        set_cell_margins(hdr2_cells[idx], top=80, bottom=80, left=100, right=100)
        p = hdr2_cells[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.runs[0]
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)

    t2_data = [
        ("State Dimension (d)", "8 genes (Ghrl, Gnas, Ins2, Nnat, etc.)", "3, 5, or 8 genes (Base: d=5)", "Number of coupled dynamic state variables."),
        ("Neural ODE Architecture", "d -> 64 -> 64 -> d (Tanh)", "d -> 64 -> 64 -> d (Tanh)", "Autonomous MLP vector field parameterization."),
        ("Optimizer & Learning Rate", "Adam (eta = 0.003)", "Adam (eta = 0.003)", "First-order adaptive gradient descent optimizer."),
        ("Training Epochs & Patience", "600 epochs (Patience = 80)", "400 epochs (Patience = 25)", "Maximum epochs and early stopping convergence threshold."),
        ("Loss Formulation", "Weighted Velocity Matching (L_vel)", "Trajectory Loss (L_obs) / NB Loss", "Empirical training objective function."),
        ("Magnitude Regularization (lambda)", "lambda = 0.001", "lambda = 0.0001", "Penalty suppressing spurious high-frequency field gradients."),
        ("Training Subsampling", "80% subsampled per seed", "100% of trajectory points", "Prevents memorization of local cell density spikes."),
        ("ODE Integration Solver", "Dopri5 (adaptive step size)", "Fixed-step RK4 (dt = 0.1)", "Numerical integration scheme for differential rollouts."),
        ("Distillation Sample Size (M)", "M = 4,000 points", "M = 3,000 points", "Density-filtered trajectory points for symbolic fitting."),
        ("kNN Density Cutoff", "k=5, 95th-percentile cutoff", "k=5, 95th-percentile cutoff", "Pruning threshold for out-of-support manifold states."),
        ("Hill Discrete Grids", "K: 8 levels (0.05..1.5), n: 5 levels (1..4)", "K: 8 levels (0.1..2.0), n: 5 levels (1..4)", "Candidate grid for forward term screening."),
        ("TRF Parameter Bounds", "K in [0.02, 5.0], n in [1.0, 6.0]", "K in [0.02, 5.0], n in [1.0, 6.0]", "Feasible parameter box for nonlinear refinement."),
        ("Multi-Start Restarts", "n_starts = 6", "n_starts = 4", "Stochastic restarts escaping sub-optimal local minima."),
        ("Pareto Model Tolerance (tau)", "tau = 0.05 (5% error relaxation)", "tau = 0.05 (5% error relaxation)", "Threshold selecting lower-complexity models within 5% error."),
        ("Spatial Block Partitioning", "60-cluster k-means in PCA space", "Trajectory-level split (60 tr / 20 te)", "Neighborhood grouping preventing kNN data leakage."),
        ("Replicate Random Seeds", "5 seeds (0, 1, 2, 3, 4)", "5 seeds per variant (75 runs)", "Multi-seed evaluation verifying statistical stability.")
    ]

    for row_idx, data in enumerate(t2_data):
        row_cells = table2.add_row().cells
        bg = "FFFFFF" if row_idx % 2 == 0 else "F8F9FA"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = text
            set_cell_shading(row_cells[col_idx], bg)
            set_cell_margins(row_cells[col_idx], top=60, bottom=60, left=90, right=90)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [0, 1] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # TABLE III
    p_t3 = doc.add_paragraph()
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.space_before = Pt(8)
    p_t3.paragraph_format.space_after = Pt(2)
    p_t3.paragraph_format.keep_with_next = True
    r_t3 = p_t3.add_run("TABLE III\nQUANTITATIVE PERFORMANCE EVALUATION AND COMPARATIVE BENCHMARKS")
    r_t3.bold = True
    r_t3.font.name = 'Times New Roman'
    r_t3.font.size = Pt(9)

    table3 = doc.add_table(rows=1, cols=7)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr3_cells = table3.rows[0].cells
    hdr3_titles = ["Category", "Code", "Metric Description", "Observed", "Threshold", "Verdict", "Baseline Comparison"]
    for idx, title in enumerate(hdr3_titles):
        hdr3_cells[idx].text = title
        set_cell_shading(hdr3_cells[idx], "EAECEE")
        set_cell_margins(hdr3_cells[idx], top=80, bottom=80, left=60, right=60)
        p = hdr3_cells[idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.runs[0]
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.0)

    t3_data = [
        ("Real Data\n(Pancreas E15.5)", "R1a", "Neural ODE Test Field Cosine", "0.520", ">= 0.50", "Satisfied", "Outperformed by Linear Baseline (0.588)."),
        ("Real Data", "R1b", "Margin over Shuffled Control", "0.155", ">= 0.30", "Not Met", "Shuffled control cosine = 0.365 (margin 0.155)."),
        ("Real Data", "R2", "Distilled Symbolic / NODE Cosine", "0.880", ">= 0.90", "Not Met", "Direct FD reaches 0.575 vs Distilled 0.519 on v."),
        ("Real Data", "R7", "Cosine Ratio: cos(sym, v) / cos(NODE, v)", "0.998", ">= 0.80", "Satisfied", "High relative preservation of directional field."),
        ("Real Baselines", "CR", "CellRank Directional Cosine", "0.475", "Baseline", "—", "Identifies Alpha, Beta, Delta terminal states."),
        ("Real Baselines", "DYN", "dynamo Directional Cosine", "-0.932", "Baseline", "—", "Divergence on Ins2 (Mu >> Ms, degradation phase)."),
        ("Real Baselines", "BO", "BoolODE Directional Cosine", "0.020", "Baseline", "—", "Low alignment due to discrete Boolean rules."),
        ("Synthetic\n(15 variants)", "S1", "Structure Recovery F1 (base)", "0.778", ">= 0.70", "Satisfied", "Direct FD reaches 0.887; Weak-SINDy 0.864."),
        ("Synthetic", "S2", "Parameter Relative Error (median)", "0.441", "<= 0.25", "Not Met", "Non-convexity and manifold parameter collinearity."),
        ("Synthetic", "S3", "Neural ODE vs True Field Cosine", "0.939", ">= 0.90", "Satisfied", "High-fidelity vector field recovery on clean data."),
        ("Synthetic", "S4", "Grönwall Bound Holds (Prop. 1)", "1.000", ">= 0.95", "Satisfied", "Verified on 70/70 runs (100% satisfaction)."),
        ("Synthetic", "S5", "Pseudotime Warped Cosine (gamma=2)", "0.712", ">= 0.90", "Not Met", "Sensitive to nonlinear pseudotime warping."),
        ("Synthetic", "S6", "Manifold Out-of-Support Drop", "0.014", ">= 0.10", "Not Met", "Extrapolation field drop is modest (< 0.10)."),
        ("Synthetic", "KO", "In-Silico KO Shift Cosine (Sym)", "0.628", "Baseline", "—", "Outperforms unconstrained Neural ODE (0.605)."),
        ("Free-Form PySR", "PT1", "Toggle R^2 vs Neural ODE Field", "0.936", ">= 0.95", "Not Met", "PySR yields polynomial rational approximations."),
        ("Free-Form PySR", "PT2", "Toggle Field NRMSE vs Ground Truth", "0.0214", "<= 0.05", "Satisfied", "High numerical accuracy in 2D phase plane."),
        ("Free-Form PySR", "PT3", "Toggle 2 Stable Attractors Retained", "1 / 3 seeds", "3 / 3 seeds", "Not Met", "Unconstrained search fails global bistability."),
        ("Free-Form PySR", "PT4", "Toggle True Regulator Recovery", "1.000", ">= 1.00", "Satisfied", "Correct cross-regulatory variable in 100% runs."),
        ("Free-Form PySR", "RP1", "Real Data PySR vs NODE Cosine", "0.764", ">= 0.90", "Not Met", "Unconstrained operator search yields lower fit."),
        ("Free-Form PySR", "RP2", "Real Data PySR Cosine Ratio", "0.852", ">= 0.80", "Satisfied", "Exceeds 0.80 relative fidelity cutoff."),
        ("Free-Form PySR", "S7", "Synthetic Ground-Truth Recall", "0.920", ">= 0.80", "Satisfied", "92% of true regulatory variables identified.")
    ]

    for row_idx, data in enumerate(t3_data):
        row_cells = table3.add_row().cells
        bg = "FFFFFF" if row_idx % 2 == 0 else "F8F9FA"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].text = text
            set_cell_shading(row_cells[col_idx], bg)
            set_cell_margins(row_cells[col_idx], top=50, bottom=50, left=50, right=50)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx in [1, 3, 4, 5] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = 'Times New Roman'
            r.font.size = Pt(7.5)
            if col_idx == 5:
                if text == "Satisfied":
                    r.font.color.rgb = RGBColor(0x00, 0x66, 0x00)
                    r.bold = True
                elif text == "Not Met":
                    r.font.color.rgb = RGBColor(0x99, 0x00, 0x00)
                    r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Detailed Figure Analyses
    add_subsubsec_heading("Quantitative Analysis of Generated Experimental Figures")
    add_body_p(
        "• Real scRNA-seq Method Benchmark (Fig. 4, fig_real_cosines.png): The linear regression baseline achieves the highest test cosine (0.588), "
        "outperforming the Neural ODE (0.520) and distilled symbolic models (0.519). CellRank achieves 0.475, successfully locating terminal Alpha, Beta, and Delta fates. "
        "dynamo displays a sharp negative divergence (-0.932) on Ins2, where unspliced counts heavily exceed mature spliced counts (Mu >> Ms), "
        "signaling pre-steady-state degradation at stage E15.5. BoolODE yields 0.020, demonstrating the limitation of discrete Boolean abstractions on continuous velocities."
    )
    add_figure("fig_real_cosines.png", "Fig. 4. Real Data Method Benchmark on Mouse Pancreas E15.5: Directional cosine similarity evaluated across test partitions for Linear baseline (0.588), Direct Finite Differences (0.575), Neural ODE (0.520), Distilled Hill Symbolic Model (0.519), CellRank (0.475), BoolODE (0.020), and dynamo (-0.932).")

    add_body_p(
        "• Attractor Endpoints and Phase Space Trajectories (Fig. 5, fig_real_endpoints.png): Long-term numerical integration of the distilled symbolic field "
        "from Ngn3 low EP progenitors accurately converges toward stable fixed points aligned with mature endocrine clusters (Alpha, Beta, Delta), "
        "corroborating analytical Jacobian eigenvalue stability classifications."
    )
    add_figure("fig_real_endpoints.png", "Fig. 5. Dynamical Attractor Endpoints: Simulated continuous trajectory rollouts originating from progenitor states (Ngn3 low EP) converging toward verified stable fixed points corresponding to terminal endocrine cell types.")

    add_body_p(
        "• Regularization and State Space Sweeps (fig_real_lambda_sweep.png & fig_real_spaces.png): Sweeping magnitude regularization lambda in [0, 1.0] "
        "increases test cosine from 0.481 (lambda=0) to 0.640 (lambda=1.0) on seed 0, suppressing unconstrained high-frequency gradients. "
        "Comparing state space representations confirms that gene space (d=8) provides direct biophysical interpretability, while latent PCA space achieves smoother rollouts."
    )
    add_figure("fig_real_lambda_sweep.png", "Fig. 6. Regularization Sensitivity Analysis: Test cosine similarity response as magnitude regularization lambda ranges from 0.0 to 1.0, demonstrating performance stabilization up to 0.640.")

    add_body_p(
        "• Synthetic Network Recovery and Noise Robustness (Fig. 7, fig_synth_structure_F1.png & fig_synth_robustness.png): "
        "On nominal synthetic trajectories, direct finite differences (F1=0.887) and Weak-SINDy (F1=0.864) exceed neural distillation (F1=0.778). "
        "However, as measurement noise escalates to sigma=20%, finite difference F1 scores degrade precipitously by 0.21, whereas Neural ODE distillation "
        "acts as a stabilizing manifold filter, maintaining F1=0.700."
    )
    add_figure("fig_synth_structure_F1.png", "Fig. 7. Synthetic Network Structure Recovery (F1 Score): Benchmarking Distilled Hill selection against Direct Finite Differences, Weak-SINDy, and free-form gplearn across 15 synthetic benchmark variants.")

    # G. Discussion
    add_subsec_heading("G. Discussion")
    add_body_p(
        "The experimental audit reveals critical foundational findings regarding the applicability and limitations of neural-symbolic modeling in single-cell genomics:",
        bold_prefix=""
    )
    add_body_p("1) The Distillation Smoothing Trade-off: On clean synthetic data, direct differential estimation (Weak-SINDy and direct finite differences) outperforms neural distillation. Neural ODE training acts as an implicit regularizer that rounds sharp non-linear Hill saturation transitions, shifting optimal parameter values and elevating median parameter relative error (S2 = 0.441). Distillation is justified primarily in high-noise regimes (sigma >= 10%), where unregularized derivatives amplify observational variance.", indent=False)
    add_body_p("2) Linear Baseline Dominance on Real Data: On mouse pancreas E15.5, a simple linear model outperforms the Neural ODE (0.588 vs. 0.520). This indicates that within noisy, 8-dimensional single-cell snapshots, coarse transcriptional dynamics are dominated by monotonic drift. Non-linear Hill kinetics provide localized mechanistic interpretability but do not universally improve generalizable test cosine similarity across unobserved Voronoi blocks.", indent=False)
    add_body_p("3) Biological Authenticity of Negative Baselines: The sharp negative cosine observed for dynamo (-0.932 on Ins2) is not an algorithmic bug, but an authentic biological manifestation of pre-steady-state degradation (Mu >> Ms). At stage E15.5, Ins2 transcription has peaked while spliced mRNA degradation remains active, producing negative velocity vectors opposite to the static expression gradient.", indent=False)
    add_body_p("4) Structured Hill Kinetics vs. Free-Form Search: While PySR achieved excellent numerical field fit (NRMSE = 0.0214) and 100% regulatory variable recall (PT4), it preserved stable fixed points in only 1 of 3 seeds (PT3). Unconstrained polynomial discovery generates unbounded rational functions that fail global phase space stability. In contrast, the structured Hill library strictly guarantees bounded trajectories and superior in-silico knock-out fidelity (0.628 vs. 0.605).", indent=False)

    # -------------------------------------------------------------
    # V. CONCLUSION AND FUTURE PERSPECTIVES
    # -------------------------------------------------------------
    add_sec_heading("V. CONCLUSION AND FUTURE PERSPECTIVES")
    
    # A. Conclusion
    add_subsec_heading("A. Conclusion")
    add_body_p(
        "This work established a cryptographically reproducible, theoretically grounded framework coupling autonomous Neural Ordinary Differential Equations "
        "with structured symbolic distillation for inferring gene regulatory dynamics from single-cell RNA sequencing data. "
        "By training a continuous vector field via confidence-weighted velocity matching and distilling it into biophysically constrained Hill kinetics, "
        "the architecture bridges black-box deep learning surrogates and interpretable biological systems. "
        "The system was comprehensively evaluated across two biological scRNA-seq datasets (mouse pancreas E15.5 and dentate gyrus), "
        "a 2D bistable toggle switch, and a 15-variant synthetic benchmark suite comprising 75 experimental runs. "
        "Theoretical Grönwall trajectory error bounds (Proposition 1) were validated empirically with 100% satisfaction across 70 runs. "
        "Crucially, rigorous benchmarking against classical baselines revealed transparent scientific boundaries: linear models match or exceed "
        "Neural ODE performance on observational snapshots, and direct differential formulations outperform distillation on low-noise synthetic benchmarks. "
        "These findings demonstrate that neural distillation should be adopted as an interpretable hypothesis-generation mechanism rather than an assumed replacement for simpler models."
    )

    # B. Future Perspectives
    add_subsec_heading("B. Future Perspectives")
    add_body_p(
        "To directly overcome the empirical limitations identified in this study, future research will pursue five key directions:",
        bold_prefix=""
    )
    add_body_p("1) Wet-Lab CRISPR Perturbation Validation: Calibrating inferred regulatory amplitudes against pooled single-cell CRISPRi/a screens (e.g., Perturb-seq) to directly validate whether inferred Hill thresholds match physical binding affinities.", indent=False)
    add_body_p("2) Multi-Omic Prior Constraining via scATAC-seq: Incorporating single-cell chromatin accessibility maps to prune inactive transcription factor motifs a priori, resolving parameter collinearity on low-dimensional manifolds.", indent=False)
    add_body_p("3) Globally Coupled Multi-Gene Optimization: Transitioning from decoupled gene-by-gene identification to joint multi-objective optimization enforcing global Lyapunov stability and exact bifurcation topologies across the entire network.", indent=False)
    add_body_p("4) Dynamic Pseudotime Unwarping: Developing an auxiliary reparameterization network t = g(tau) to correct for nonlinear developmental warping, addressing the performance decline observed in the pseudo2 benchmark.", indent=False)
    add_body_p("5) Decentralized Public Preregistration: Anchoring configuration trees into immutable public registries (such as the Open Science Framework) prior to execution, elevating scientific accountability beyond local SHA-256 verification.", indent=False)

    # REFERENCES
    add_sec_heading("REFERENCES")
    refs = [
        "[1] G. La Manno et al., \"RNA velocity of single cells,\" Nature, vol. 560, no. 7719, pp. 494–498, 2018.",
        "[2] V. Bergen, M. Lange, S. Peidli, F. A. Wolf, and F. J. Theis, \"Generalizing RNA velocity to transient cell states through dynamical modeling,\" Nature Biotechnology, vol. 38, no. 12, pp. 1408–1414, 2020.",
        "[3] H. de Jong, \"Modeling and simulation of genetic regulatory systems: A literature review,\" Journal of Computational Biology, vol. 9, no. 1, pp. 67–103, 2002.",
        "[4] N. Meinshausen and P. Bühlmann, \"High-dimensional graphs and variable selection with the Lasso,\" Annals of Statistics, vol. 34, no. 3, pp. 1436–1462, 2006.",
        "[5] P. Dibaeinia and S. Sinha, \"SERGIO: A single-cell expression simulator guided by gene regulatory networks,\" Cell Systems, vol. 11, no. 3, pp. 252–271, 2020.",
        "[6] X. Qiu et al., \"Mapping transcriptomic vector fields of single cells,\" Cell, vol. 185, no. 4, pp. 690–711, 2022.",
        "[7] M. Lange et al., \"CellRank for directed single-cell fate mapping,\" Nature Methods, vol. 19, no. 2, pp. 159–170, 2022.",
        "[8] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, \"Neural ordinary differential equations,\" in Advances in Neural Information Processing Systems (NeurIPS), 2018, pp. 6571–6583.",
        "[9] M. Cranmer et al., \"Discovering symbolic models from deep learning with inductive biases,\" in Advances in Neural Information Processing Systems (NeurIPS), 2020.",
        "[10] T. Stephens, gplearn: Genetic Programming in Python with a scikit-learn inspired API, 2015. [Online]. Available: https://gplearn.readthedocs.io",
        "[11] S. L. Brunton, J. L. Proctor, and J. N. Kutz, \"Discovering governing equations from data by sparse identification of nonlinear dynamical systems,\" Proc. Natl. Acad. Sci. U.S.A., vol. 113, no. 15, pp. 3932–3937, 2016.",
        "[12] D. A. Messenger and D. M. Bortz, \"Weak SINDy for partial differential equations,\" Journal of Computational Physics, vol. 443, p. 110525, 2021.",
        "[13] A. Bastidas-Ponce et al., \"Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis,\" Development, vol. 146, no. 12, p. dev173849, 2019.",
        "[14] D. Hochgerner et al., \"Conserved properties of dentate gyrus neurogenesis across postnatal development revealed by single-cell RNA sequencing,\" Nature Neuroscience, vol. 21, no. 2, pp. 290–299, 2018.",
        "[15] X. Qiu et al., \"Reversed graph embedding resolves complex single-cell trajectories,\" Nature Methods, vol. 14, no. 10, pp. 979–982, 2017.",
        "[16] A. Pratapa, A. P. Jalihal, J. N. Law, A. Bharadwaj, and T. M. Murali, \"Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data,\" Nature Methods, vol. 17, no. 2, pp. 147–154, 2020.",
        "[17] V. A. Huynh-Thu, A. Irrthum, L. Wehenkel, and P. Geurts, \"Inferring regulatory networks from expression data using tree-based methods,\" PLoS ONE, vol. 5, no. 9, p. e12776, 2010.",
        "[18] D. A. Messenger and D. M. Bortz, \"Weak SINDy: Galerkin-based data-driven model selection,\" Multiscale Modeling & Simulation, vol. 19, no. 3, pp. 1474–1497, 2021."
    ]
    for r_text in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p_ref.add_run(r_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8.5)

    doc.save(DOC_PATH)
    print(f"Document successfully created at: {DOC_PATH}")

if __name__ == "__main__":
    create_document()
