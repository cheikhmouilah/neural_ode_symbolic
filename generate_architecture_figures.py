"""
Script to generate IEEE publication-grade architectural diagrams:
- Figure 1: Overall System Architecture (Input data, preprocessing, Neural ODE, Symbolic distillation, Evaluation)
- Figure 2: Proposed System Workflow (Flowchart with conditions, decision points, validation)
- Figure 3: Detailed Module Interaction (Inter-module data contracts, APIs, dependencies)
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set IEEE style font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUT_DIR = r"c:\neural_ode_symbolic\figures_article"
DOCS_FIG_DIR = r"c:\neural_ode_symbolic\docs\figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_FIG_DIR, exist_ok=True)

# Color Palette (IEEE Transactions muted professional palette)
C_PRIMARY = '#1f4e79'    # Deep Blue / Navy
C_SECONDARY = '#2e75b6'  # Medium Blue
C_ACCENT = '#008080'     # Deep Teal
C_ORANGE = '#c55a11'     # Rust / Ochre
C_PURPLE = '#7030a0'     # Violet
C_BG_LIGHT = '#f2f4f7'   # Soft grey-blue container bg
C_BOX_WHITE = '#ffffff'
C_BORDER = '#2c3e50'
C_TEXT = '#1a1a1a'
C_MUTED = '#555555'

# ==============================================================================
# FIGURE 1: Overall System Architecture
# ==============================================================================
def draw_figure_1():
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 97.5, "Figure 1. End-to-End System Architecture: From Single-Cell Transcripts to Interpretable Regulatory Laws",
            ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)

    # 4 Major Stages Columns / Blocks
    stages = [
        {"title": "1. Data Ingestion & Preprocessing", "x": 2, "w": 22, "color": C_SECONDARY},
        {"title": "2. Continuous Vector Field Learning", "x": 26.5, "w": 23, "color": C_PRIMARY},
        {"title": "3. Symbolic Distillation & Refinement", "x": 51.5, "w": 23.5, "color": C_ACCENT},
        {"title": "4. Dynamics & Formal Verification", "x": 76.5, "w": 21.5, "color": C_ORANGE}
    ]

    for st in stages:
        # Background container
        rect = patches.FancyBboxPatch((st['x'], 6), st['w'], 87, boxstyle="round,pad=0.8,rounding_size=1.5",
                                     linewidth=1.2, edgecolor=st['color'], facecolor=C_BG_LIGHT, alpha=0.85)
        ax.add_patch(rect)
        # Header banner
        header = patches.FancyBboxPatch((st['x'], 87), st['w'], 6, boxstyle="round,pad=0.2,rounding_size=1.0",
                                       linewidth=0, facecolor=st['color'])
        ax.add_patch(header)
        ax.text(st['x'] + st['w']/2, 90, st['title'], ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white')

    # --- Column 1 Boxes ---
    c1_boxes = [
        ("Input Datasets", ["• Real scRNA-seq (.h5ad): Pancreas E15.5, Dentate Gyrus", "• Synthetic Benchmarks: 15 variants, 75 runs", "• SHA-256 pre-execution hash verification"], 72, 13),
        ("Transcript Kinetics Preprocessing", ["• Spliced / Unspliced separation (u, s)", "• scVelo deterministic velocity estimation", "• High-variance HVG selection (d=8 genes)", "• 99th-percentile expression scaling"], 50, 18),
        ("Manifold-Preserving Splitting", ["• k-means on top 10 PCA dimensions", "• 60 neighborhood-aware cell clusters", "• 60% Train / 20% Val / 20% Test split", "• Eliminates kNN spatial leakage"], 26, 20),
        ("Outputs: Z (state), V (velocity), w (conf)", [], 10, 11)
    ]
    for title, items, y, h in c1_boxes:
        box = patches.FancyBboxPatch((3.5, y), 19, h, boxstyle="round,pad=0.4",
                                     edgecolor='#999999', facecolor=C_BOX_WHITE, linewidth=0.8)
        ax.add_patch(box)
        ax.text(4.5, y + h - 2.2, title, fontsize=9, fontweight='bold', color=C_PRIMARY)
        for idx, it in enumerate(items):
            ax.text(4.8, y + h - 5.0 - idx * 3.3, it, fontsize=7.8, color=C_TEXT)

    # --- Column 2 Boxes ---
    c2_boxes = [
        ("Neural ODE Architecture (VF)", ["• Autonomous MLP parameterization", "• Architecture: d -> 64 -> 64 -> d", "• Non-saturating Tanh activations", "• Parameter count: 5,256 weights (d=8)"], 68, 17),
        ("Objective Function & Optimization", ["• Velocity Matching Loss: L_vel", "• Confidence-weighted: w_i * ||v_i - f_θ(z_i)||^2", "• Kinetic scale normalization nu", "• Magnitude regularization: λ * ||f_θ||^2", "• Adam optimizer, lr=0.003, clip=5.0"], 40, 24),
        ("ODE Solvers & Integration", ["• Real: Dopri5 adaptive Runge-Kutta", "• Synthetic: Fixed-step RK4 (Δt=0.1)", "• Early stopping (patience=80)", "• Subsampling 80% per seed for stability"], 15, 21)
    ]
    for title, items, y, h in c2_boxes:
        box = patches.FancyBboxPatch((28, y), 20, h, boxstyle="round,pad=0.4",
                                     edgecolor='#999999', facecolor=C_BOX_WHITE, linewidth=0.8)
        ax.add_patch(box)
        ax.text(29, y + h - 2.2, title, fontsize=9, fontweight='bold', color=C_PRIMARY)
        for idx, it in enumerate(items):
            ax.text(29.3, y + h - 5.0 - idx * 3.3, it, fontsize=7.8, color=C_TEXT)

    # --- Column 3 Boxes ---
    c3_boxes = [
        ("Trajectory Distillation", ["• Fine trajectory rollouts from progenitors", "• kNN density estimation (k=5 neighbors)", "• Manifold jittering & out-of-support rejection", "• Generates M=4,000 dense state targets"], 68, 17),
        ("Structured Hill Library Screening", ["• Activating: H+(z; K, n) = z^n / (K^n + z^n)", "• Repressing: H-(z; K, n) = K^n / (K^n + z^n)", "• Discrete grids: K (8 levels), n (5 levels)", "• Greedy forward term selection (k_max=3)"], 44, 20),
        ("Bounded Multi-Start Refinement", ["• Non-Negative Least Squares (NNLS)", "• Trust-Region Reflective (TRF) nonlinear solver", "• Multi-start optimization (n_starts=6)", "• Parameter boundaries: K in [0.02, 5], n in [1, 6]"], 22, 18),
        ("Pareto Model Selection", ["• Complexity penalty: Ω(M) = 1 + 3*k_hill + 1", "• Strict 5% tolerance Pareto frontier rule"], 9, 9)
    ]
    for title, items, y, h in c3_boxes:
        box = patches.FancyBboxPatch((53, y), 20.5, h, boxstyle="round,pad=0.4",
                                     edgecolor='#999999', facecolor=C_BOX_WHITE, linewidth=0.8)
        ax.add_patch(box)
        ax.text(54, y + h - 2.2, title, fontsize=9, fontweight='bold', color=C_ACCENT)
        for idx, it in enumerate(items):
            ax.text(54.3, y + h - 4.8 - idx * 3.3, it, fontsize=7.8, color=C_TEXT)

    # --- Column 4 Boxes ---
    c4_boxes = [
        ("Formal Grönwall Bound (Prop. 1)", ["• Trajectory error bound verification", "• Bound: ||e(t)|| <= (ε/L)(e^{Lt}-1) + δ_0 e^{Lt}", "• Verified across 70 runs (100% holds)"], 68, 17),
        ("Dynamical & Stability Analysis", ["• Jacobian computation J = ∇F_sym(z*)", "• Root-finding multi-starts (n=120)", "• Eigenvalue real parts (Re(λ) < 0)", "• Attractor-to-cluster assignment"], 44, 20),
        ("In-Silico Perturbation (KO)", ["• Virtual gene deletion (z_k = 0, dz_k = 0)", "• Trajectory shift cosine evaluation", "• Concordance with true phenotypic shifts"], 24, 16),
        ("Reproducibility & Verification", ["• SHA-256 config freeze (PREREGISTRATION)", "• 15 automated consistency checks passed", "• Non-regression suite (16/16 pass)"], 8, 13)
    ]
    for title, items, y, h in c4_boxes:
        box = patches.FancyBboxPatch((78, y), 18.5, h, boxstyle="round,pad=0.4",
                                     edgecolor='#999999', facecolor=C_BOX_WHITE, linewidth=0.8)
        ax.add_patch(box)
        ax.text(79, y + h - 2.2, title, fontsize=8.8, fontweight='bold', color=C_ORANGE)
        for idx, it in enumerate(items):
            ax.text(79.3, y + h - 4.8 - idx * 3.3, it, fontsize=7.6, color=C_TEXT)

    # Inter-column arrows
    arrow_props = dict(boxstyle="rarrow,pad=0.3", fc="#333333", ec="none", lw=0)
    for x_arr in [23.5, 48.5, 74.0]:
        ax.text(x_arr + 1.2, 50, "   ", bbox=arrow_props, fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_overall_architecture.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig1_overall_architecture.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(DOCS_FIG_DIR, "fig1_overall_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 generated successfully.")

# ==============================================================================
# FIGURE 2: Proposed System Workflow (Flowchart)
# ==============================================================================
def draw_figure_2():
    fig, ax = plt.subplots(figsize=(10.5, 7.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 98, "Figure 2. Execution Flowchart and Guarded Verification Protocol",
            ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)

    # Flowchart Nodes Definition (text, x, y, w, h, color, shape)
    nodes = [
        ("Start: Load Experiment Configuration & Check Prerequisites", 50, 93, 70, 4.5, C_PRIMARY, "round"),
        ("Verify Datasets SHA-256 Hashes\n(pancreas: 9e3e459e..., dentategyrus: 6fbf1ec6...)", 50, 84.5, 62, 6.0, C_SECONDARY, "box"),
        ("Hash Valid?", 50, 75.0, 24, 6.0, C_ORANGE, "diamond"),
        ("Compute RNA Velocity (scVelo) & Extract Moments (Ms, Mu)\nNeighborhood 60-Cluster k-Means Spatial Splitting", 50, 64.5, 66, 6.5, C_SECONDARY, "box"),
        ("Train Neural ODE Field f_θ(z) via Weighted Velocity Loss L_vel\n(Adam, lr=0.003, epochs=600, patience=80, clip=5.0)", 50, 54.0, 66, 6.5, C_PRIMARY, "box"),
        ("Integrate Trajectories & Sample D_f on Data Support\nkNN Density Rejection (95th-pct, k=5, M=4000 points)", 50, 43.5, 64, 6.5, C_ACCENT, "box"),
        ("Greedy Hill Term Screening (NNLS) &\nBounded Multi-Start Optimization (TRF Solver)", 50, 33.0, 62, 6.5, C_ACCENT, "box"),
        ("Select Minimal Complexity Model Ω on Pareto Frontier\nSubject to: Validation MSE <= (1 + 0.05) * MSE_min", 50, 22.5, 62, 6.0, C_ACCENT, "box"),
        ("Run 15 Automated Consistency Checks & Theoretical Guardrails\n(Monotonicity, Shuffled Control Margin, Grönwall Bound S4=1.000)", 50, 12.5, 66, 6.5, C_ORANGE, "box"),
        ("Export Evaluated Metrics, LaTeX Tables & High-Resolution Figures\n(RAPPORT_RESULTATS.md, CSV, PDF, figures_article/)", 50, 3.5, 70, 4.8, C_PRIMARY, "round")
    ]

    for label, cx, cy, w, h, col, shape in nodes:
        if shape == "round":
            b = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.4,rounding_size=2.0",
                                       fc=col, ec=col, lw=1.2)
            ax.add_patch(b)
            ax.text(cx, cy, label, ha='center', va='center', fontsize=8.8, fontweight='bold', color='white')
        elif shape == "diamond":
            diamond = patches.Polygon([[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]],
                                      fc='#fff2cc', ec=C_ORANGE, lw=1.5)
            ax.add_patch(diamond)
            ax.text(cx, cy, label, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#804000')
        else:
            b = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                       fc=C_BOX_WHITE, ec=col, lw=1.2)
            ax.add_patch(b)
            ax.text(cx, cy, label, ha='center', va='center', fontsize=8.4, color=C_TEXT)

    # Connections between sequential nodes
    arrow_args = dict(arrowstyle="->", lw=1.4, color='#333333')
    
    # 1 -> 2
    ax.annotate("", xy=(50, 87.5), xytext=(50, 90.7), arrowprops=arrow_args)
    # 2 -> 3
    ax.annotate("", xy=(50, 78.0), xytext=(50, 81.5), arrowprops=arrow_args)
    # 3 -> 4 (Yes)
    ax.annotate("", xy=(50, 67.8), xytext=(50, 72.0), arrowprops=arrow_args)
    ax.text(52, 70.0, "Yes (Valid)", fontsize=8, color='green', fontweight='bold')
    
    # 3 -> Halt (No)
    ax.annotate("", xy=(85, 75.0), xytext=(62, 75.0), arrowprops=arrow_args)
    ax.text(68, 76.2, "No (Mismatch)", fontsize=8, color='red', fontweight='bold')
    halt_box = patches.FancyBboxPatch((85, 72.5), 12, 5, boxstyle="round,pad=0.2",
                                      fc='#fadbd8', ec='red', lw=1.2)
    ax.add_patch(halt_box)
    ax.text(91, 75, "Abort Execution\n(Integrity Error)", ha='center', va='center', fontsize=7.5, color='darkred', fontweight='bold')

    # 4 -> 5
    ax.annotate("", xy=(50, 57.3), xytext=(50, 61.2), arrowprops=arrow_args)
    # 5 -> 6
    ax.annotate("", xy=(50, 46.8), xytext=(50, 50.7), arrowprops=arrow_args)
    # 6 -> 7
    ax.annotate("", xy=(50, 36.3), xytext=(50, 40.2), arrowprops=arrow_args)
    # 7 -> 8
    ax.annotate("", xy=(50, 25.5), xytext=(50, 29.7), arrowprops=arrow_args)
    # 8 -> 9
    ax.annotate("", xy=(50, 15.8), xytext=(50, 19.5), arrowprops=arrow_args)
    # 9 -> 10
    ax.annotate("", xy=(50, 5.9), xytext=(50, 9.2), arrowprops=arrow_args)

    # Lateral annotation for Consistency Failures
    ax.annotate("", xy=(85, 12.5), xytext=(83, 12.5), arrowprops=arrow_args)
    ax.plot([83, 83], [12.5, 12.5], color='#333333')
    halt_box2 = patches.FancyBboxPatch((85, 10.0), 12, 5, boxstyle="round,pad=0.2",
                                       fc='#fadbd8', ec='red', lw=1.2)
    ax.add_patch(halt_box2)
    ax.text(91, 12.5, "Halt on Guard\nInconsistency", ha='center', va='center', fontsize=7.5, color='darkred', fontweight='bold')
    ax.annotate("", xy=(85, 12.5), xytext=(83, 12.5))
    ax.plot([83, 85], [12.5, 12.5], color='red', lw=1.2)
    ax.text(80, 14, "If 15/15 Fail", fontsize=7.5, color='red', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_system_workflow.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig2_system_workflow.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(DOCS_FIG_DIR, "fig2_system_workflow.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 generated successfully.")

# ==============================================================================
# FIGURE 3: Detailed Module Interaction
# ==============================================================================
def draw_figure_3():
    fig, ax = plt.subplots(figsize=(11.0, 7.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 97.5, "Figure 3. Functional Module Decomposition, Interfaces, and Data Contracts",
            ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)

    # 6 Core Functional Modules (Nodes)
    mod_specs = [
        {"id": "M1", "title": "Module 1: real_data.py\n(Data Ingestion & Split)", "x": 6, "y": 66, "w": 26, "h": 24, "col": C_SECONDARY,
         "in": "Input: Raw .h5ad files, config.json", "out": "Output: Z (scaled Ms), V (velocity),\nw (confidence), split (60 blocks)"},
        {"id": "M2", "title": "Module 2: real_node.py\n(Neural ODE Learning)", "x": 38, "y": 66, "w": 26, "h": 24, "col": C_PRIMARY,
         "in": "Input: Z, V, w, tr/va indices, cfg", "out": "Output: PyTorch f_θ(z) weights,\nloss curves, node_cos_test"},
        {"id": "M3", "title": "Module 3: real_identify.py\n(Symbolic Distillation)", "x": 70, "y": 66, "w": 26, "h": 24, "col": C_ACCENT,
         "in": "Input: Distilled D_f, f_θ(z), Hill grid", "out": "Output: Symbolic models {eq_i},\nPareto frontiers, estimated K, n, γ"},
        {"id": "M4", "title": "Module 4: real_dyn.py\n(Stability & Dynamics)", "x": 70, "y": 14, "w": 26, "h": 24, "col": C_ORANGE,
         "in": "Input: Symbolic models {eq_i}, clusters", "out": "Output: Fixed points z*, Jacobian J,\neigenvalues, cluster radii"},
        {"id": "M5", "title": "Module 5: synth_methods.py\n(Benchmarks & Grönwall)", "x": 38, "y": 14, "w": 26, "h": 24, "col": C_PURPLE,
         "in": "Input: Synthetic trajectories, truth EDO", "out": "Output: S1-S7 metrics, Prop. 1 bound\n(E <= B holds fraction), F1 direct/weak"},
        {"id": "M6", "title": "Module 6: checks/ & Verification\n(prereg, consistency, verify)", "x": 6, "y": 14, "w": 26, "h": 24, "col": '#333333',
         "in": "Input: All 166 result JSON files", "out": "Output: RAPPORT_RESULTATS.md,\n15/15 consistency pass, figures"}
    ]

    for m in mod_specs:
        b = patches.FancyBboxPatch((m['x'], m['y']), m['w'], m['h'], boxstyle="round,pad=0.4,rounding_size=1.2",
                                   fc=C_BOX_WHITE, ec=m['col'], lw=1.5)
        ax.add_patch(b)
        
        # Header strip inside box
        header = patches.FancyBboxPatch((m['x'], m['y'] + m['h'] - 6.5), m['w'], 6.5, boxstyle="round,pad=0.1,rounding_size=1.0",
                                       fc=m['col'], ec=m['col'])
        ax.add_patch(header)
        ax.text(m['x'] + m['w']/2, m['y'] + m['h'] - 3.2, m['title'], ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white')

        # Content
        ax.text(m['x'] + 1.2, m['y'] + 11.5, m['in'], fontsize=7.8, fontweight='bold', color=C_MUTED)
        ax.text(m['x'] + 1.2, m['y'] + 4.5, m['out'], fontsize=7.8, color=C_TEXT)

    # Connections and Data Flow Arrows
    arrow_style = dict(arrowstyle="->", lw=1.5, color='#2c3e50')
    
    # M1 -> M2 (Data matrices)
    ax.annotate("", xy=(38, 78), xytext=(32, 78), arrowprops=arrow_style)
    ax.text(35, 80, "Z, V, w", ha='center', fontsize=7.8, fontweight='bold', color=C_PRIMARY)

    # M2 -> M3 (Vector field predictions & distillation cloud)
    ax.annotate("", xy=(70, 78), xytext=(64, 78), arrowprops=arrow_style)
    ax.text(67, 80, "D_f, f_θ(z)", ha='center', fontsize=7.8, fontweight='bold', color=C_ACCENT)

    # M3 -> M4 (Identified symbolic ODE system)
    ax.annotate("", xy=(83, 38), xytext=(83, 66), arrowprops=arrow_style)
    ax.text(85, 52, "{dz_i/dt}", ha='left', fontsize=8, fontweight='bold', color=C_ORANGE)

    # M2 -> M5 (Neural ODE integration vs Truth for Prop 1)
    ax.annotate("", xy=(51, 38), xytext=(51, 66), arrowprops=arrow_style)
    ax.text(53, 52, "Trajectory rollouts", ha='left', fontsize=7.8, color=C_PURPLE)

    # M3 -> M5 (Symbolic field vs True field)
    ax.annotate("", xy=(64, 26), xytext=(70, 26), arrowprops=arrow_style)
    ax.text(67, 28, "F_sym", ha='center', fontsize=7.8, color=C_ACCENT)

    # M5 -> M6 (Validation results & Grönwall checks)
    ax.annotate("", xy=(32, 26), xytext=(38, 26), arrowprops=arrow_style)
    ax.text(35, 28, "Metrics JSON", ha='center', fontsize=7.8, color='#333333')

    # M1 -> M6 (Data hashes & split verification)
    ax.annotate("", xy=(19, 38), xytext=(19, 66), arrowprops=arrow_style)
    ax.text(21, 52, "SHA-256 checks", ha='left', fontsize=7.8, color='#333333')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_module_interaction.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "fig3_module_interaction.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(DOCS_FIG_DIR, "fig3_module_interaction.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 generated successfully.")

if __name__ == "__main__":
    draw_figure_1()
    draw_figure_2()
    draw_figure_3()
