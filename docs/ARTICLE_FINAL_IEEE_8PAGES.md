# Symbolic Discovery of Governing Equations for Cell Fate Decisions: Coupling Neural ODEs and Biophysical Distillation from Single-Cell Transcriptomics

**[Author One], [Author Two], and [Author Three]**  
*[Affiliation, City, Country] — [corresponding author e-mail]*

---

### Abstract
Single-cell RNA sequencing measures each cell once, so cell-fate dynamics must be inferred from asynchronous snapshots. RNA velocity gives a local direction of change, but a vector field is not a set of regulatory laws. We couple a Neural ODE, fitted to velocity vectors on eight genes, with a step that distills Hill functions from the learned field. Unlike earlier Hill-based neural models, the exponent is restricted to $n \in \{1, 2, 4\}$, and a coupled unspliced–spliced $(u, s)$ system handles the transcriptional lag that breaks uncoupled velocity. A leakage-free, pre-registered protocol (Voronoi block splits, SHA-256 checks, git-sealed amendments) evaluates the pipeline on three mouse tissues (pancreas, dentate gyrus, erythroid), on 15 synthetic variants with known equations, and against CRISPR Perturb-seq knock-outs. In none does the Neural ODE beat a linear baseline in uncentered cosine (pancreas 0.520 vs. 0.588; dentate gyrus 0.592 vs. 0.634; erythroid 0.467 vs. 0.516). In pancreas, with the penalty chosen on validation, the centered cosine reaches 0.607 vs. 0.586 and the amplitude error falls modestly; a latent PCA space reaches 0.767 (linear: 0.647). Discretizing $n$ cuts the median Hill-parameter error on synthetic data from 44.1% to 0.6%, and structure recovery holds when the true exponent lies outside the grid ($F_1 = 0.748$ at $n=3.0$). Distillation fidelity is 0.953 on dentate gyrus and 0.939 on erythroid. It beats finite differences under 10% and 20% dropout ($F_1$ 0.483 vs. 0.280; 0.430 vs. 0.123) but not clearly under 40% or 60%. The coupled model fits *Ins2* transcription with $R^2=0.993$, consistent with a pre-steady-state account of the dynamo failure. On Perturb-seq profiles the Neural ODE reaches 71.4% sign concordance (linear: 41.1%); the distilled Hill model does not. Eight of 29 thresholds are not met, and we report them.

**Index Terms—** Gene regulatory networks, neural ordinary differential equations, symbolic regression, Hill kinetics, RNA velocity, single-cell transcriptomics, CRISPR Perturb-seq.

---

## I. INTRODUCTION

In the language of dynamical systems, a cell choosing a fate is a point rolling toward one of several valleys. Waddington drew that picture [1], and the genetic toggle switch later showed that cooperative mutual repression is enough to carve two stable states [2]. Single-cell RNA sequencing (scRNA-seq) lets us see such states across thousands of cells, at a price: the measurement destroys the cell, and a snapshot distribution is compatible with several different dynamics [3]. RNA velocity sidesteps part of the problem by reading unspliced and spliced transcripts as a local direction of change [4]. scVelo and dynamo extend it to dynamical models and vector fields [5], [6], and CellRank turns it into fate probabilities [7]. The estimates stay fragile, though, and depend on modeling choices [8], [9]. Even an accurate field is not a set of governing equations. It does not say which genes regulate which, through what functional form, or whether the interaction can be identified at all.

This article is about that last step. Neural ordinary differential equations learn a continuous field [10], but their weights are neither rate constants nor regulatory logic. SINDy [11] and PySR [12] compress derivative information into short expressions, and fitting symbolic models to a trained network is a known route to interpretable laws [13]. For gene regulation the obvious vocabulary is Hill kinetics, which captures cooperative, saturating activation and repression [14]. We chain the two stages: a Neural ODE trained on velocity vectors acts as a denoiser, and a structured Hill library is distilled from the frozen field. We work in gene space on a few genes and keep module and latent spaces for comparison only, since equations in latent coordinates describe the representation and not interactions between genes.

We are not the first to mix mechanism with neural dynamics. PHOENIX embeds Hill–Langmuir kinetics inside a Neural ODE [15], and DynNet places Hill functions in the drift of a neural stochastic model constrained by a prior network [16]; as we read them, the Hill form is built into the model, its thresholds are estimated, and DynNet fixes the Hill coefficient at 4. We go the other way. The neural field is learned with no prescribed form, the Hill form is extracted from it afterwards, the exponent is discretized, and a coupled $(u, s)$ layer handles cells in a pre-steady-state phase. CLERA and NeuroVelo extract interpretable dynamics in latent spaces [17], [18], LogicSR pairs Boolean logic with symbolic regression [19], and TSvelo, SymVelo and Cell-MNN add regulatory structure to velocity models [20], [21], [22]. DeepVelo [23], [24], TIGON [25] and scKINETICS [26] return fields, trajectories or networks rather than equations, and a neural surrogate followed by sparse regression has been tried on generic biological ODEs [27]. Within our targeted literature search, completed in October 2026, we found no study that quantifies, against known generating equations, how reliably symbolic search on a learned gene-level field recovers regulatory structure under noise and dropout.

Three obstacles stand in the way. Hill laws are parametrically non-identifiable: the threshold $K$ and exponent $n$ trade off along a long, flat valley of the loss. A field learned from snapshots is unconstrained away from the data, and correlated genes make several symbolic forms equivalent on the data manifold. And cells in a pre-steady-state phase break the assumptions of standard velocity estimators. In the E15.5 pancreas, unspliced counts exceed spliced counts at loci such as *Ins2*, and the dynamo field points against the differentiation flow there. A good fit therefore establishes neither causality nor mechanism.

We test four falsifiable hypotheses against thresholds fixed in a pre-registration file. (H1) A Neural ODE trained on block-held-out cells predicts velocity direction better than a linear field. (H2) Restricting $n$ to stoichiometric values brings Hill-parameter error on synthetic data below 25%. (H3) A coupled $(u, s)$ model accounts for loci on which uncoupled velocity fails. (H4) In-silico knock-outs agree in sign with Perturb-seq shifts better than a linear model does. H1 fails in its pre-registered form (uncentered cosine, frozen penalty). We then revisit it with the penalty chosen on the validation partition and with a centered cosine and an amplitude error; that change of metric came after the failure, so we flag it as exploratory.

We claim no new biological mechanism. The contributions are methodological:
1. A two-stage pipeline that distills Hill kinetics from a learned continuous field, with support-constrained sampling and a discretized exponent that reduces parametric sloppiness (44.1% to 0.6% on synthetic data, and unchanged structure recovery when the true exponent lies outside $\{1, 2, 4\}$);
2. A leakage-controlled, pre-registered protocol (Voronoi block splitting, SHA-256 verification, 15 consistency checks) with linear, CellRank, dynamo, BoolODE, Weak-SINDy and PySR baselines;
3. A coupled $(u, s)$ formulation that explains the *Ins2* velocity divergence and yields distilled transcription laws for four more loci; and
4. A validation on synthetic data, on two further tissues (dentate gyrus, erythroid gastrulation) and against Perturb-seq screens, reported together with the thresholds that are not met.

Section II sets up the problem, Sections III and IV describe the models and the system, Section V reports results, Section VI discusses limits, and Section VII concludes.

---

## II. PROBLEM FORMULATION AND MATHEMATICAL FOUNDATIONS

### A. State, Time and Velocity
That cells occupy distinct heritable states, and that cooperative mutual repression can generate bistability, is established; that cell types are attractors of a gene-level ODE is a working hypothesis; that expression equals the regulatory state is a modeling assumption. The state $\mathbf{z} \in \mathbb{R}^d$ is obtained from expression $\mathbf{x} \in \mathbb{R}^G$ by a map $\pi$. We compare three spaces of dimension $d=8$: gene space ($\pi$ selects genes), module space (genes aggregated into meta-genes) and latent PCA space. Hill terms have a regulatory reading only in the first two, and latent equations describe the representation.

Each cell is measured once, so temporal structure is inferred. The experimental time $t$ is a coarse sampling label, the pseudotime $\tau$ a monotone ordering and not a clock, and the velocity $\mathbf{v}$ an estimate of local change and not an exact derivative. Our data come from a single sampling day (E15.5), so every rate below is in the inferred kinetic time of the estimator. RNA velocity derives from the kinetic model, with transcription $\alpha$, splicing $\beta$ and degradation $\gamma$:
$$\frac{du}{dt} = \alpha - \beta u, \quad \frac{ds}{dt} = \beta u - \gamma s, \quad v = \frac{ds}{dt} \tag{1}$$
whose rates are estimated under debated assumptions [8], [9]; we use $\mathbf{v}$ as a noisy, confidence-weighted supervision signal. Two identifiability facts matter. Under a time reparametrization $\tau = \phi(t)$ the field becomes $\mathbf{f}/\phi'$, so its direction is constrained but its speed only up to an unknown, possibly state-dependent factor; rates are then relative, while $K$ and $n$ survive a constant rescaling. And if all data lie in the zero set of a function $q$, then $\mathbf{f}$ and $\mathbf{f} + q\mathbf{g}$ coincide on the data: an equation is defined only modulo functions vanishing on the support, which is the mechanism of library collinearity.

### B. Autonomous Dynamics and Stability
We assume the autonomous system
$$\dot{\mathbf{z}} = \mathbf{f}(\mathbf{z}), \quad \mathbf{z}(0) = \mathbf{z}_0 \tag{2}$$
chosen deliberately, since time-dependent terms must be justified by the design. For locally Lipschitz $\mathbf{f}$ solutions are unique and trajectories cannot cross. Equilibria satisfy $\mathbf{f}(\mathbf{z}^*) = \mathbf{0}$ and are asymptotically stable if $\max_j \operatorname{Re} \lambda_j(\mathbf{J}) < 0$ for the Jacobian $\mathbf{J} = \partial \mathbf{f}/\partial \mathbf{z}$ at $\mathbf{z}^*$. Stable equilibria are candidate cell types and saddles candidate decision regions, but these are properties of the fitted model and not evidence about cells.

### C. Hill Kinetics
For a regulator of abundance $z \ge 0$, threshold $K > 0$ and coefficient $n \ge 1$ [14], [28],
$$H^-(z; K, n) = \frac{1}{1 + (z/K)^n}, \quad H^+ = 1 - H^- \tag{3}$$
and a regulated gene obeys the additive law, with independent inputs summed,
$$\frac{dz_i}{dt} = a_{i0} + \sum_k a_{ik} H^{\sigma_k}(z_{j_k}; K_k, n_k) - \gamma_i z_i \tag{4}$$
with $\sigma_k \in \{+, -\}$, non-negative amplitudes and $\gamma_i \ge 0$. A Hill term is an effective relation between transcript levels, not a model of promoter binding, and its parameters can be practically non-identifiable: for $z \ll K$ only $a K^{-n}$ and $n$ are constrained, and for $z \gg K$ the term saturates and neither $K$ nor $n$ is. Recovering a Hill-type expression implies neither the biological parameters nor the interaction itself.

---

## III. NEURAL VECTOR FIELD LEARNING AND COUPLED DYNAMICS

### A. Autonomous Neural Vector Field
The field is a multilayer perceptron without time input,
$$\mathbf{f}_\theta(\mathbf{z}) = \mathbf{W}_3 \tanh(\mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3 \tag{5}$$
with 64 hidden units per layer, i.e. 5,256 weights for $d=8$. On real data it is fitted by velocity matching,
$$\mathcal{L}(\theta) = \frac{1}{\nu |\mathcal{I}|} \sum_{i \in \mathcal{I}} w_i \|\mathbf{v}_i - \mathbf{f}_\theta(\mathbf{z}_i)\|^2 + \lambda \|\mathbf{f}_\theta(\mathbf{z}_i)\|^2 \tag{6}$$
where $w_i$ is the confidence weight of cell $i$, $\nu = |\mathcal{I}|^{-1} \sum_i w_i \|\mathbf{v}_i\|^2 + 10^{-12}$ is the empirical velocity variance that makes the loss dimension-free, and $\lambda$ weighs a magnitude penalty. Integration (adaptive Dormand–Prince, dopri5) enters afterwards, to sample trajectories from the frozen field; on synthetic data with time labels the field is trained through an RK4 trajectory loss [10]. We call both a Neural ODE because the learned object is an integrated autonomous field, but the real-data objective is regression on velocity and inherits every bias of the estimator. Training uses Adam (rate 0.003), gradient clipping at 5.0, an 80% training subsample per seed and early stopping (patience 80, at most 600 epochs). The penalty damps high-frequency components at the price of oversmoothing, and cosine similarity ignores field scale, so a magnitude penalty can raise directional agreement while shrinking the speeds on which symbolic rates depend. We therefore treat $\lambda$ as a sensitivity parameter, sweep it over $\{0, 0.001, 0.01, 0.1, 1\}$, choose it per seed on the validation partition and freeze it before the test partition is touched. Because the raw cosine rewards a direction shared by all cells, we also report a centered cosine, computed after removing the mean velocity, and the normalized root-mean-square error (NRMSE) of the velocity amplitude.

### B. Coupled Transcription–Splicing Dynamics
Uncoupled velocity treats transcription as constant and reads $v_i = \beta_i u_i - \gamma_i s_i$. This fails for cells still accumulating unspliced RNA: at E15.5 endocrine commitment is fast and $M_u \gg M_s$ at loci such as *Ins2*. We model both species,
$$\frac{du_i}{dt} = \alpha_i(\mathbf{s}) - \beta_i u_i, \quad \frac{ds_i}{dt} = \beta_i u_i - \gamma_i s_i \tag{7}$$
with transcription a regulatory function of the spliced abundances, $\alpha_i(\mathbf{s}) = \operatorname{MLP}_\psi(\mathbf{s})$. The explicit $u$ compartment carries the transcriptional delay. The rates $\beta_i, \gamma_i$ are identified jointly with $\psi$ by nonlinear least squares, and $\alpha_i$ is then distilled into Hill form (Section IV-C).

---

## IV. SYSTEM DESIGN, ALGORITHMS, AND SYMBOLIC DISTILLATION

### A. Architecture and Guardrails
Four sequential tiers (Fig. 1) ingest and verify data and build leakage-free splits, learn the continuous field, sample the supported manifold and identify Hill kinetics, and finally analyze stability, fit the coupled model and run perturbation tests. Eight modules implement them, executed by a workflow that halts when a guard fails: the SHA-256 digest of an input must match `datasets.json`, configuration files must match `PREREGISTRATION.json` unless an amendment is documented, 15 consistency checks must pass (one rejects runs in which more noise improves the Neural ODE cosine by over 0.03), and a final audit reads all 183 result files for NaN values. The pre-registration, frozen on 11 November 2026, is sealed to a git commit, and two later amendments (erythroid dataset, 40% and 60% dropout) are timestamped and verified by script; this is a local record, not a public deposit. These guards prevent silent errors; they do not make the science correct.

### B. Data Preparation and Leakage Control
Moments of spliced and unspliced counts are pooled over a $k\text{NN}$ graph (30 neighbors, 30 PCs) and deterministic scVelo velocities are computed. The state is the $d=8$ genes with the largest variance of the spliced moment among those passing the velocity filters, chosen without a regulatory prior: *Ghrl, Gnas, Ins2, Nnat, Chga, Clu, Isl1, Cpe*. This avoids confirmation bias but favors markers of terminal endocrine states (hormones, secretory proteins) and contains one transcription factor, *Isl1*; *Ngn3*, *Pax4* and *Arx* are not in the state. States and velocities are divided by the gene-specific 99th percentile of the spliced moment. Because $k\text{NN}$ pooling lets neighboring cells share information, a cell-level split leaks. We therefore cluster cells into 60 Voronoi blocks ($k$-means on 10 PCs) and assign whole blocks to training, validation and test (60/20/20) by a seeded permutation.

### C. Support-Constrained Hill Distillation
The distillation set $\mathcal{D}_f$ must stay where data constrain the field. Trajectories are integrated from *Ngn3*-low progenitors over $T=5.0$, candidates are perturbed by Gaussian noise scaled by the local fifth-nearest-neighbor distance, those beyond the 95th percentile of $k\text{NN}$ distance are discarded, and $M=4000$ points are kept. For each target gene the library holds activating and repressing Hill columns for every state gene, with $K$ on a grid and $n \in \{1, 2, 4\}$, plus a constant and a degradation column. Terms enter by greedy forward selection (up to three), each step minimizing the residual of
$$\min_{a_k, \gamma_i \ge 0} \left\| \sum_k a_k H_k + a_0 - \gamma_i z_i - y_i \right\|_2^2 \tag{8}$$
where $H_k$ abbreviates the $k$-th Hill column and $y_i$ is the neural field component. The structure is refined by bounded multi-start trust-region-reflective (TRF) least squares over amplitudes, $K \in [0.02, 5]$ and $\gamma_i$ (six starts). Candidates are scored by complexity $\Omega = 2 + \sum_k c_k$, with $c_k = 3$ for a Hill term and 1 for a linear one, and Algorithm 1 keeps the simplest model within 5% of the minimal validation error; this rule is fixed before test results are seen.

**Discretization of $n$:** With continuous $n \in [1, 6]$, $K$ and $n$ move together along a valley that preserves sigmoid curvature, and the median parameter error reached 44.1%. Restricting $n$ to $\{1, 2, 4\}$, read as monomeric, dimeric and tetrameric binding [28], removes the collinearity. This is a structural prior, valid only if the true cooperativity lies in the set. Distillation targets are model-generated, so low distillation error never certifies recovery of the true field; we therefore compare with direct finite differences, weak-form SINDy [29] and free-form PySR [12].

### D. Dynamical Analysis and Knock-Outs
Equilibria are found by Powell’s hybrid method from 120 starts and classified by the Jacobian spectrum. Every Hill term lies in $[0, 1]$, so $\dot{z}_i \le A_i - \gamma_i z_i$ with $A_i = a_{i0} + \sum_k a_{ik}$ and $z_i(t) \le \max\{z_i(0), A_i/\gamma_i\}$ when $\gamma_i > 0$: distilled trajectories cannot diverge, a guarantee free-form expressions lack. In-silico knock-out of gene $k$ clamps $z_k = \dot{z}_k = 0$ and integrates to the new steady state; the predicted shift is compared with the reference by cosine, sign concordance (over the $d-1$ other genes) and mean absolute error (MAE).

---

## V. EXPERIMENTAL RESULTS AND BENCHMARK ANALYSIS

### A. Environment, Data and Protocol
Code runs in Python 3.14.2 on 64-bit Windows 11 (8 CPU cores, no GPU) with PyTorch, torchdiffeq, SciPy, Scanpy, scVelo 0.3.4, scikit-learn, PySR and PySINDy. Three real datasets are used. The mouse pancreas E15.5 file [30] must match the digest `9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6`. The mouse dentate gyrus data [31] ($N=2{,}930$ cells) go through the same pipeline, with eight genes chosen by the same variance rule and radial glia-like cells as progenitors. The erythroid lineage of mouse gastrulation [32] ($N=9{,}815$ cells) is processed alike. Hyperparameters were frozen before the benchmark: a $d \to 64 \to 64 \to d$ tanh network, Adam at 0.003 with at most 600 epochs (synthetic: 400, RK4 with $\Delta t = 0.1$), $M=4000$ distillation points (synthetic: 3000), eight threshold values $K$ and $n \in \{1, 2, 4\}$, six TRF starts (synthetic: four), $\tau=0.05$ and five seeds. The validation-based choice of $\lambda$, the centered metrics, the dentate gyrus and erythroid runs, the exponent-misspecification tests and the 40% and 60% dropout regimes were added afterwards. The synthetic benchmark has 15 variants of multi-gene networks with known equations, varying noise ($\sigma \in [0.02, 0.20]$), dropout ($p \in [0.10, 0.20]$), trajectory number ($N \in [20, 120]$), dimension ($d=3, 5, 8$) and count model, with five seeds each (75 runs). A dropout benchmark covers 10%, 20%, 40% and 60% zero-inflation (five seeds each). The perturbation benchmark uses curated response profiles of eight transcription factors (GATA1, SPI1, KLF1, RUNX1, TAL1, CEBPA, FLI1, MYC) derived from single-cell CRISPR screens [33], [34]. Baselines are a linear field, finite differences, CellRank [7], dynamo [6], BoolODE [35], Weak-SINDy and PySR. Of 29 thresholds, 21 are met and 8 are not; Table I lists the principal ones. Thresholds of components added after the first run are exploratory unless a documented amendment covers them.

#### TABLE I: PRINCIPAL METRICS, PRE-REGISTERED THRESHOLDS AND BASELINES

| Group | Code | Metric | Obs. | Threshold | Verdict | Baseline comparison / note |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Pancreas E15.5** | R1a frozen | Neural ODE test cosine ($\lambda=0.001$, uncentered) | 0.520 | $\ge 0.50$ | **Met** | Linear 0.588; five-seed means; no $\lambda$ beats linear |
| | R1-CENT | Centered test cosine, $\lambda$ chosen on validation | 0.607 ± 0.022 | none | — | Linear 0.586; shuffled control 0.091 ± 0.121 |
| | R1-NRMSE | Velocity amplitude error (NRMSE) | 0.036 ± 0.001 | none | — | Linear 0.040; shuffled 0.049 |
| | R1b | Margin over shuffled control (centered) | 0.516 | $\ge 0.30$ | **Met** | 0.607 − 0.091 |
| | R2 gene | Distilled vs. Neural ODE cosine, gene space | 0.890 ± 0.024 | $\ge 0.90$ | **Not met** | Distilled vs. velocity 0.573 ± 0.053 |
| | R2 module | Distilled vs. Neural ODE cosine, module space | 0.969 | $\ge 0.90$ | **Met** | Exceeds threshold in meta-gene space |
| | SPACE-LAT | Neural ODE cosine, latent PCA ($d=8$) | 0.767 | none | — | Linear 0.647 (+0.120); mean of 3 seeds |
| | BASE-CR | CellRank cosine | 0.475 | none | — | Recovers Alpha, Beta, Delta fates |
| | BASE-DYN | dynamo cosine | −0.932 | none | — | Diverges on *Ins2* ($M_u \gg M_s$) |
| | BASE-BO | BoolODE cosine | 0.020 | none | — | Boolean switching vs. continuous kinetics |
| **Dentate gyrus** | DG-NODE | Neural ODE test cosine, 5 seeds | 0.592 ± 0.038 | $\ge 0.50$ | **Met** | Linear 0.634; seeds 0.639, 0.630, 0.591, 0.550, 0.551 |
| | DG-R2 | Distilled vs. Neural ODE cosine, 5 seeds | 0.953 ± 0.026 | $\ge 0.90$ | **Met** | 5/5 seeds; range 0.914–0.983 |
| | DG-VEL | Distilled vs. measured velocity | 0.600 ± 0.023 | none | — | Direct FD 0.632 ± 0.020 |
| **Erythroid** | ERY-NODE | Neural ODE test cosine, 5 seeds | 0.467 ± 0.026 | $\ge 0.50$ | **Not met** | Linear 0.516; direct FD 0.541 ± 0.005 |
| | ERY-R2 | Distilled vs. Neural ODE cosine, 5 seeds | 0.939 ± 0.009 | $\ge 0.90$ | **Met** | 5/5 seeds; range 0.927–0.953 |
| **Coupled $(u, s)$** | COUP-INS2 | Transcription fit $R^2$, *Ins2* | 0.993 | $\ge 0.90$ | **Met** | $\beta=2.37, \gamma=2.44$ |
| | COUP-GHRL | Transcription fit $R^2$, *Ghrl* | 0.977 | $\ge 0.90$ | **Met** | Hill $n=2, K=0.8$ |
| | COUP-NNAT | Transcription fit $R^2$, *Nnat* | 0.971 | $\ge 0.90$ | **Met** | Hill $n=1, K=0.8$ |
| **Perturb-seq** | CRISP-NODE | Neural ODE shift cosine | +0.427 | none | — | Sign concordance 71.4%; MAE 3.216 |
| | CRISP-LIN | Linear shift cosine | +0.035 | none | — | Sign concordance 41.1%; MAE 0.719 |
| | CRISP-HILL | Distilled Hill shift cosine | −0.042 | none | — | Lowest MAE, 0.557 |
| **Synthetic** | S1 | Structure recovery $F_1$ (base) | 0.778 | $\ge 0.70$ | **Met** | Direct FD 0.887; Weak-SINDy 0.864 |
| | S1-D10 | $F_1$, dropout 10% | 0.483 ± 0.087 | none | — | FD 0.280; +0.203; 5/5 seeds |
| | S1-D20 | $F_1$, dropout 20% | 0.430 ± 0.130 | none | — | FD 0.123; +0.307; 5/5 seeds |
| | S1-D40 | $F_1$, dropout 40% | 0.212 ± 0.099 | none | — | FD 0.246 ± 0.090; tied / collapse |
| | S1-D60 | $F_1$, dropout 60% | 0.237 ± 0.093 | none | — | FD 0.161 ± 0.096; both at noise baseline |
| | S2 cont. | Parameter error, $n \in [1, 6]$ | 0.441 | $\le 0.25$ | **Not met** | $K$–$n$ collinearity |
| | S2 disc. | Parameter error, $n \in \{1, 2, 4\}$ | 0.006 | $\le 0.25$ | **Met** | From 44.1% to 0.6% |
| | S2-MIS-3 | $F_1$, true $n=3.0$ (outside grid) | 0.748 ± 0.068 | $\ge 0.70$ | **Met** | Continuous exponent 0.695 ± 0.069 |
| | S2-MIS-1.5 | $F_1$, true $n=1.5$ (outside grid) | 0.667 ± 0.074 | none | — | Continuous exponent 0.660 ± 0.111 |
| | S2-CTRL-2 | $F_1$, true $n=2$ (in grid) | 0.723 ± 0.060 | $\ge 0.70$ | **Met** | Continuous exponent 0.718 ± 0.113 |
| | S4 | Grönwall bound satisfied (Prop. 1) | 1.000 | $\ge 0.95$ | **Met** | 70/70 runs (14 variants $\times$ 5 seeds) |
| | S5 | Cosine under pseudotime warping ($\gamma=2$) | 0.712 | $\ge 0.90$ | **Not met** | Sensitive to nonlinear warping |
| | S6 | Out-of-support cosine drop | 0.014 | $\ge 0.10$ | **Not met** | Drop smaller than expected |
| | S7 | True-regulator (variable) recall | 0.920 | $\ge 0.80$ | **Met** | 5 base runs |
| **Free-form PySR** | PT1 | Toggle $R^2$ to Neural ODE field | 0.936 | $\ge 0.95$ | **Not met** | Rational/polynomial approximations |
| | PT3 | Two stable attractors retained | 1/3 seeds | 3/3 | **Not met** | Global bistability not preserved |
| | RP1 | Real data, PySR vs. Neural ODE cosine | 0.764 | $\ge 0.90$ | **Not met** | Lower fidelity than Hill library |

---

### B. Pancreas: Prediction, Regularization and State Spaces
At the frozen configuration the linear baseline (0.588) beats the Neural ODE (0.520) and the distilled model (0.519); these are five-seed means of the uncentered cosine. Direct finite differences reach 0.575. A shuffled-velocity control still scores 0.365 here (0.290 for the validation-selected models), which shows that much of the raw cosine reflects a direction shared by all cells and not cell-specific information. Rollouts from *Ngn3*-low progenitors end mostly in progenitor-like clusters for the Neural ODE within $T=5.0$, while the distilled system sends most cells to Beta, some to Delta and almost none to Alpha; we do not claim to reproduce the full branching.

The sweep over $\lambda$ (Fig. 2) does not change the uncentered picture: over five seeds the Neural ODE stays between about 0.52 and 0.56 at every $\lambda$, always below the linear baseline, with a wide spread at $\lambda=1$. The centered cosine hovers around the centered linear baseline (0.586). With $\lambda$ chosen per seed on validation and frozen before testing, it reaches $0.607 \pm 0.022$, and the amplitude error is lower than the linear model’s (NRMSE $0.036 \pm 0.001$ against 0.040; shuffled control 0.049). The shuffled control collapses to $0.091 \pm 0.121$ under centering, so the margin of 0.516 over it clears the 0.30 threshold. These gains are small: about one standard deviation in centered cosine and a 10% relative drop in amplitude error. We introduced the centered metric after the frozen run failed, so we treat this as exploratory evidence that the field carries cell-specific direction, not as confirmation of H1. Across spaces (means over 5, 3 and 3 seeds), the Neural ODE scores 0.520 in gene space, about 0.66 in module space and 0.767 in latent PCA space against 0.647 for the linear model (+0.120). In module space the distilled model matches the neural field at 0.969, above the 0.90 threshold; in gene space it reaches $0.890 \pm 0.024$ and narrowly misses it.

### C. Other Tissues: Dentate Gyrus and Erythroid Lineage
On the mouse dentate gyrus the pipeline runs unchanged. Over five seeds the Neural ODE reaches a test cosine of $0.592 \pm 0.038$ (threshold 0.50) but stays below the linear baseline (0.634) and direct finite differences ($0.632 \pm 0.020$). Distillation reaches a fidelity of $0.953 \pm 0.026$ against the neural field, with all five seeds above 0.90 (range 0.914 to 0.983), and correlates with the measured velocity at $0.600 \pm 0.023$. The erythroid lineage (9,815 cells) shows the same pattern: Neural ODE $0.467 \pm 0.026$, below the threshold, the linear baseline (0.516) and finite differences ($0.541 \pm 0.005$); fidelity $0.939 \pm 0.009$ (5 of 5 seeds, 0.927 to 0.953); correlation with the measured velocity $0.476 \pm 0.022$. The variance rule picked mostly generic genes there (e.g., *Actb, Calm2, Tpm4*), so the laws describe housekeeping and cell-cycle programs more than erythroid regulation. We report no shuffled control or perturbation test for either tissue. In all three tissues, then, the Neural ODE does not beat a linear field in uncentered cosine on clean data, while distillation fidelity exceeds 0.90 on the two added tissues and in pancreas module space, but not in pancreas gene space (0.890).

### D. Coupled $(u, s)$ Kinetics and the *Ins2* Case
The coupled model gives $\beta_{\text{Ins2}} = 2.372$ and $\gamma_{\text{Ins2}} = 2.444\text{ h}^{-1}$; since E15.5 is a single time, these are relative rates in inferred kinetic time and not half-lives. The distilled transcription law of *Ins2* ($R^2=0.993$) is a saturating, self-activating Hill term with $n=1$:
$$\frac{du_{\text{Ins2}}}{dt} = \frac{5.02 \, s_{\text{Ins2}}}{1.2 + s_{\text{Ins2}}} - 2.37 \, u_{\text{Ins2}} \tag{9}$$
$$\frac{ds_{\text{Ins2}}}{dt} = 2.37 \, u_{\text{Ins2}} - 2.44 \, s_{\text{Ins2}} \tag{10}$$
Similar fits hold for *Ghrl* ($R^2=0.977$; $n=2, K=0.8$), *Nnat* (0.971; $n=1, K=0.8$), *Cpe* (0.886) and *Isl1* (0.872; $n=2, K=0.5$). The rollouts stay close to the nullcline $u = (\gamma/\beta) s$ and, for *Ghrl*, barely move, so the portraits illustrate the fitted steady-state relation more than a transient accumulation of unspliced RNA. The $R^2$ values are goodness of fit for the transcription term, not held-out prediction, and Eq. (9) shows association, not proof that *Ins2* regulates itself.

### E. Perturbation Benchmark Against CRISPR Screens
The Neural ODE reaches a shift cosine of +0.427 and a sign concordance of 71.4%, the linear model +0.035 and 41.1% (Fig. 4). With seven downstream genes per target these correspond to 40 and 23 of 56 target–gene pairs against 50% by chance; the pairs are not independent, so we give no p-value. The distilled Hill model has a mean cosine of −0.042 and the smallest amplitude error (MAE 0.557, against 0.719 for linear and 3.216 for the Neural ODE). The mean hides heterogeneity: Hill is high on GATA1, SPI1 and CEBPA (0.73–0.91) and strongly negative on RUNX1, FLI1 and MYC, while the Neural ODE is positive on six of eight targets. Two limits apply: the eight regulators are hematopoietic whereas the learned field is defined on eight pancreatic genes, so this tests the knock-out procedure in another regulatory context and not the pancreatic equations; and eight targets are few.

### F. Synthetic Ground Truth and Free-Form Search
Under nominal conditions direct finite differences ($F_1 = 0.887$) and Weak-SINDy (0.864) recover structure better than distillation (0.778) (Fig. 5), and the ordering holds in most variants: at $\sigma=20\%$ finite differences stay near 0.87 against about 0.70. Distillation is therefore not a noise-robustness device. Its advantage appears under moderate dropout: at 10% it reaches $0.483 \pm 0.087$ against 0.280 for finite differences (+0.203, 5 of 5 seeds) and at 20% $0.430 \pm 0.130$ against 0.123 (+0.307, 5 of 5), while the weak form scores about 0.15 and 0.11. It does not survive heavier corruption: at 40% finite differences are slightly ahead ($0.246 \pm 0.090$ against $0.212 \pm 0.099$) and at 60% distillation reaches $0.237 \pm 0.093$ against $0.161 \pm 0.096$ for finite differences. The difference at 60% (0.076) is within one standard deviation, so the pipeline does not claim a clear structural advantage beyond 20% dropout. Both methods degrade to the baseline of sparse network guessing ($\sim 0.20$).

The Neural ODE approximates the true field with cosine 0.939 and 92% of true regulatory inputs are recovered (S7). Fig. 6 shows that distilled $F_1$ falls only from about 0.80 to 0.70 as $\sigma$ rises to 0.20 while the field cosine stays near 0.94, but dropout lowers $F_1$ to about 0.48 and 0.43 and the field cosine to about 0.68 and 0.57: dropout is the dominant failure mode of the gene-space pipeline. Free-form PySR fits the toggle field well (NRMSE 0.0214) and includes the true regulator in every run, yet keeps two stable attractors in one of three seeds and reaches $R^2 = 0.936$ against the neural field (below 0.95); on real data its cosine with the neural field is 0.764.

A direct test of the exponent prior places the true exponent outside the grid. With $n_{\text{true}} = 3.0$ the discrete grid reaches $F_1 = 0.748 \pm 0.068$, against $0.695 \pm 0.069$ for a continuous exponent; at $n_{\text{true}} = 1.5$ the values are $0.667 \pm 0.074$ and $0.660 \pm 0.111$, and a well-specified control ($n=2$) gives $0.723 \pm 0.060$ and $0.718 \pm 0.113$. The differences lie within one standard deviation, so the grid does not hurt structure recovery when the exponent is wrong, and it does not demonstrably help beyond a lower spread at $n=1.5$. The test concerns structure ($F_1$); we did not measure parameter error under misspecification.

### G. Trajectory Error Bound
**Proposition 1.** *Let $\dot{\mathbf{z}} = \mathbf{f}(\mathbf{z})$ and $\dot{\mathbf{w}} = \mathbf{g}(\mathbf{w})$ with $\|\mathbf{z}_0 - \mathbf{w}_0\| \le \delta_0$. If $\|\mathbf{f}(\mathbf{z}(t)) - \mathbf{g}(\mathbf{z}(t))\| \le \varepsilon$ on $[0, T]$ and $\mathbf{g}$ is $L$-Lipschitz, then*
$$\|\mathbf{z}(t) - \mathbf{w}(t)\| \le \delta_0 e^{Lt} + \frac{\varepsilon}{L} (e^{Lt} - 1) \tag{11}$$
*Proof.* With $\mathbf{e} = \mathbf{z} - \mathbf{w}$, $\frac{d}{dt} \|\mathbf{e}\| \le \|\mathbf{f}(\mathbf{z}) - \mathbf{g}(\mathbf{z})\| + \|\mathbf{g}(\mathbf{z}) - \mathbf{g}(\mathbf{w})\| \le \varepsilon + L \|\mathbf{e}\|$, and Grönwall’s inequality gives Eq. (11). $\square$

On test trajectories we estimate $\varepsilon$ and $L = \max_t \sigma_{\max}(\nabla \mathbf{F})$ along the trajectory and check $E(t) \le B(t) + 10^{-12}$, with $B$ the right side of Eq. (11). It holds on 70 of 70 trajectories (14 variants $\times$ 5 seeds). Since the inequality is a theorem, this validates our numerical estimates and the code and says nothing about biology. The bound grows exponentially in $Lt$, and on real data the neural-to-true field error is not measurable, so low distillation error never certifies recovery of the true dynamics.

---

## VI. DISCUSSION

**Smoothing versus identifiability:** Regularization does not rescue the uncentered cosine: at every $\lambda$ the five-seed mean stays between about 0.52 and 0.56, below the linear baseline. The centered cosine and the amplitude error look slightly better (0.607 against 0.586; 0.036 against 0.040), but the first margin is about one standard deviation and the second a relative drop of 10%. We read the pancreas result as a Neural ODE comparable to a linear field in direction and mildly better in amplitude, not as a decisive win. Weak regularization leaves high-frequency velocity noise in the field, which the symbolic stage copies faithfully; strong regularization shrinks speeds, which the cosine cannot see. The shuffled control at 0.091 under centering, against 0.290 to 0.365 without, shows why the raw cosine was a weak yardstick here.

**Latent PCA and dynamo:** Projecting on the leading components discards gene-specific variance, which is where dropout concentrates, and the Neural ODE moves from 0.068 below the linear model in gene space (frozen setting) to 0.120 above it. The linear model also gains (0.588 to 0.647), so part of the benefit is plain denoising. We did not ablate dropout on real data, and latent coordinates carry no Hill reading. For dynamo (−0.932) we favor a real pre-steady-state phase: at *Ins2* unspliced RNA far exceeds spliced RNA, a regime where the steady-state assumption of uncoupled estimation fails, and the coupled model fits that locus with $R^2=0.993$. This is a consistent explanation and not a demonstration, because we never relaxed dynamo’s assumptions one at a time. The phase portraits of Fig. 3 show rollouts hugging the $s=0$ nullcline, so they support the fitted rate relation more than the transient itself.

**Exponent prior and other tissues:** The misspecification test lowers the main worry about the grid $\{1, 2, 4\}$: with the true exponent at 3.0 or 1.5, structure recovery ($F_1$ 0.748 and 0.667) stays at the level of a continuous-exponent fit. What it leaves open is parameter recovery, since the 0.6% error was measured with a correctly specified set. On the dentate gyrus and erythroid lineage, distillation fidelity is high and stable (0.953, 0.939), yet the distilled system does not beat direct finite differences against measured velocity (0.600 against 0.632; 0.476 against 0.541), and the Neural ODE is below the linear baseline (0.592 against 0.634; 0.467 against 0.516). We read these as evidence of robustness, not of superiority.

**Complementarity:** The Neural ODE gets knock-out directions right more often than not, but its amplitudes are the worst of the three models. The Hill model has the best amplitude, bounded dynamics and a compact reading, yet its mean cosine hides large differences between targets. The library is a likely culprit. Eq. (4) sums independent inputs and has no multiplicative gates $H(z_a) H(z_b)$, whereas real knock-outs involve heteromeric complexes (AND) or redundant paralogs (OR). We therefore use the Neural ODE as an in-silico screening engine and the Hill system as a source of stable, inspectable hypotheses. Free-form search does not replace the structured library on real data, and it often loses bistability. Table II maps the results to levels of evidence.

#### TABLE II: LEVELS OF CLAIM AND STATUS IN THIS STUDY

| Level | Claim | Status |
| :---: | :--- | :--- |
| **1** | Held-out prediction | Partly: centered cosine 0.607 vs. 0.586, NRMSE 0.036 vs. 0.040; uncentered cosine below linear in all three tissues |
| **2** | Effective field learned | Five seeds run; cross-seed field consistency not quantified |
| **3** | Fidelity to neural field | Module 0.969, dentate gyrus 0.953, erythroid 0.939 (met); pancreas gene 0.890 (not met) |
| **4–5**| Functional form, interaction class | Synthetic only ($F_1 = 0.778$, recall 0.920); holds outside the exponent grid |
| **6** | Hill parameters | Synthetic: 0.6% (discrete $n$, correct set), 44.1% (continuous); not tested under misspecification |
| **7** | Causal mechanism | Not established; Perturb-seq context differs |

**Limitations:** The pancreas result rests on one sampling day, so every rate is relative; the added tissues have a linear baseline but no shuffled control, and three tissues do not establish generality. The eight state genes were picked by variance, include a single transcription factor and leave out *Ngn3*, *Pax4* and *Arx*; the equations describe hormone and secretory programs more than upstream regulation. The centered metric, the validation-based penalty, the dentate gyrus and erythroid runs, the misspecification tests and the dropout regimes were added after the first frozen run, and we treat them as exploratory. The Perturb-seq profiles come from another tissue, and eight targets are few. Eight of 29 thresholds are not met. Among them, pseudotime warping (S5) fits the reparametrization ambiguity of Section II, and the missing out-of-support degradation (S6) means either that the field extrapolated more smoothly than expected or that our test was too mild.

---

## VII. CONCLUSION AND PERSPECTIVES

We built and tested a pipeline that goes from a learned continuous field to explicit Hill-type equations in gene space. Four results stand out:
1. Discretizing the Hill exponent to $\{1, 2, 4\}$ cut the median synthetic parameter error from 44.1% to 0.6%, and structure recovery did not degrade when the true exponent lay outside the grid.
2. Distillation was stable across tissues, with fidelity 0.969 in pancreas module space, 0.953 on the dentate gyrus and 0.939 on the erythroid lineage.
3. A coupled $(u, s)$ system fitted the *Ins2* transcription law with $R^2=0.993$ where uncoupled velocity diverges.
4. The Neural ODE predicted Perturb-seq shift signs with 71.4% concordance, in a different tissue.

The negative results count just as much. In none of the three tissues did the Neural ODE beat a linear model in uncentered cosine; in pancreas, with the penalty chosen on validation, it reached 0.607 against 0.586 in centered cosine and a modestly lower amplitude error, and only a latent space gave a clear gain (0.767 against 0.647). Direct methods recovered structure better on clean data, and free-form search lost global bistability. Distillation earns its place through robustness to moderate dropout (10% to 20%, but not clearly under 40% or 60%), bounded dynamics and a compact reading; it does not identify regulatory mechanisms from observational snapshots. Next we will add multiplicative AND/OR gates for multi-target knock-outs, use scATAC-seq priors to restrict which regulators may act on which genes, measure parameter error under exponent misspecification, test matched perturbation data in the added tissues, and deposit the pre-registration in a public archive such as Zenodo.

---

## REFERENCES

[1] C. H. Waddington, *The Strategy of the Genes*. London, U.K.: George Allen & Unwin, 1957.  
[2] T. S. Gardner, C. R. Cantor, and J. J. Collins, "Construction of a genetic toggle switch in Escherichia coli," *Nature*, vol. 403, no. 6767, pp. 339–342, Jan. 2000, doi: 10.1038/35002131.  
[3] C. Weinreb, S. Wolock, B. K. Tusi, M. Socolovsky, and A. M. Klein, "Fundamental limits on dynamic inference from single-cell snapshots," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 115, no. 10, pp. E2467–E2476, Mar. 2018, doi: 10.1073/pnas.1714723115.  
[4] G. La Manno et al., "RNA velocity of single cells," *Nature*, vol. 560, no. 7719, pp. 494–498, Aug. 2018, doi: 10.1038/s41586-018-0414-6.  
[5] V. Bergen, M. Lange, S. Peidli, F. A. Wolf, and F. J. Theis, "Generalizing RNA velocity to transient cell states through dynamical modeling," *Nat. Biotechnol.*, vol. 38, no. 12, pp. 1408–1414, Dec. 2020, doi: 10.1038/s41587-020-0591-3.  
[6] X. Qiu et al., "Mapping transcriptomic vector fields of single cells," *Cell*, vol. 185, no. 4, pp. 690–711.e45, Feb. 2022, doi: 10.1016/j.cell.2021.12.045.  
[7] M. Lange et al., "CellRank for directed single-cell fate mapping," *Nat. Methods*, vol. 19, no. 2, pp. 159–170, Feb. 2022, doi: 10.1038/s41592-021-01346-6.  
[8] V. Bergen, R. A. Soldatov, P. V. Kharchenko, and F. J. Theis, "RNA velocity—current challenges and future perspectives," *Mol. Syst. Biol.*, vol. 17, no. 8, Art. no. e10282, Aug. 2021, doi: 10.15252/msb.202110282.  
[9] G. Gorin, M. Fang, T. Chari, and L. Pachter, "RNA velocity unraveled," *PLoS Comput. Biol.*, vol. 18, no. 9, Art. no. e1010492, Sep. 2022, doi: 10.1371/journal.pcbi.1010492.  
[10] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, "Neural ordinary differential equations," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, Dec. 2018, pp. 6572–6583.  
[11] S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 113, no. 15, pp. 3932–3937, Apr. 2016.  
[12] M. Cranmer, "Interpretable machine learning for science with PySR and SymbolicRegression.jl," May 2023, arXiv:2305.01582.  
[13] M. Cranmer et al., "Discovering symbolic models from deep learning with inductive biases," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, Dec. 2020, pp. 17429–17442.  
[14] U. Alon, *An Introduction to Systems Biology: Design Principles of Biological Circuits*. Boca Raton, FL, USA: Chapman & Hall/CRC, 2006.  
[15] I. Hossain, V. Fanfani, J. Fischer, J. Quackenbush, and R. Burkholz, "Biologically informed NeuralODEs for genome-wide regulatory dynamics," *Genome Biol.*, vol. 25, Art. no. 127, May 2024, doi: 10.1186/s13059-024-03264-0.  
[16] J. Dou et al., "Inferring stochastic dynamics by biophysical Neural ODE using single-cell transcriptomics," *Nat. Commun.*, vol. 17, Art. no. 6901, May 2026, doi: 10.1038/s41467-026-73257-z.  
[17] M. Sadria and V. Swaroop, "Discovering governing equations of biological systems through representation learning and sparse model discovery," *NAR Genom. Bioinform.*, vol. 7, no. 2, Art. no. lqaf048, Apr. 2025, doi: 10.1093/nargab/lqaf048.  
[18] I. Kouadri Boudjelthia, S. Milite, N. El Kazwini, Y. Huang, A. Sottoriva, and G. Sanguinetti, "Interpretable learning of temporal cellular dynamics from single-cell data," *Cell Rep. Methods*, vol. 6, no. 3, Art. no. 101342, Mar. 2026, doi: 10.1016/j.crmeth.2026.101342.  
[19] D. Zhang et al., "LogicSR: prior-guided symbolic regression for gene regulatory network inference from single-cell transcriptomics data," *Brief. Bioinform.*, vol. 26, no. 6, Art. no. bbaf621, Nov. 2025, doi: 10.1093/bib/bbaf621.  
[20] J. Li, Z. Wang, H.-B. Shen, and Y. Yuan, "Comprehensive RNA velocity by modeling the cascade of gene regulation, transcription and splicing from single-cell RNA sequencing data with TSvelo," *eLife*, vol. 14, Art. no. RP108950, Sep. 2026, doi: 10.7554/eLife.108950.  
[21] C. Xie et al., "RNA velocity prediction via neural ordinary differential equation," *iScience*, vol. 27, no. 4, Art. no. 109635, Apr. 2024, doi: 10.1016/j.isci.2024.109635.  
[22] J.-P. von Bassewitz et al., "Learning explicit single-cell dynamics using ODE representations," in *Proc. 14th Int. Conf. Learn. Represent. (ICLR)*, Apr. 2026. [Online]. Available: https://openreview.net/forum?id=DzSNH5APPl  
[23] Z. Chen, W. C. King, A. Hwang, M. Gerstein, and J. Zhang, "DeepVelo: Single-cell transcriptomic deep velocity field learning with neural ordinary differential equations," *Sci. Adv.*, vol. 8, no. 48, Art. no. eabq3745, Dec. 2022, doi: 10.1126/sciadv.abq3745.  
[24] H. Cui, H. Maan, M. C. Vladoiu, J. Zhang, M. D. Taylor, and B. Wang, "DeepVelo: Deep learning extends RNA velocity to multi-lineage systems with cell-specific kinetics," *Genome Biol.*, vol. 25, Art. no. 27, Jan. 2024, doi: 10.1186/s13059-023-03148-9.  
[25] Y. Sha, Y. Qiu, P. Zhou, and Q. Nie, "Reconstructing growth and dynamic trajectories from single-cell transcriptomics data," *Nat. Mach. Intell.*, vol. 6, no. 1, pp. 25–39, Jan. 2024, doi: 10.1038/s42256-023-00763-w.  
[26] C. Burdziak, C. J. Zhao, D. Haviv, D. Alonso-Curbelo, S. W. Lowe, and D. Pe’er, "scKINETICS: inference of regulatory velocity with single-cell transcriptomics data," *Bioinformatics*, vol. 39, no. Suppl. 1, pp. i394–i403, Jun. 2023, doi: 10.1093/bioinformatics/btad267.  
[27] X. Wu, M. McDermott, and A. L. MacLean, "Data-driven model discovery and model selection for noisy biological systems," *PLoS Comput. Biol.*, vol. 21, no. 1, Art. no. e1012762, Jan. 2025, doi: 10.1371/journal.pcbi.1012762.  
[28] H. de Jong, "Modeling and simulation of genetic regulatory systems: A literature review," *J. Comput. Biol.*, vol. 9, no. 1, pp. 67–103, Jan. 2002, doi: 10.1089/10665270252833208.  
[29] D. A. Messenger and D. M. Bortz, "Weak SINDy: Galerkin-based data-driven model selection," *Multiscale Model. Simul.*, vol. 19, no. 3, pp. 1474–1497, 2021, doi: 10.1137/20M1343166.  
[30] A. Bastidas-Ponce et al., "Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis," *Development*, vol. 146, no. 12, Art. no. dev173849, Jun. 2019, doi: 10.1242/dev.173849.  
[31] H. Hochgerner, A. Zeisel, P. Lönnerberg, and S. Linnarsson, "Conserved properties of dentate gyrus neurogenesis across postnatal development revealed by single-cell RNA sequencing," *Nat. Neurosci.*, vol. 21, no. 2, pp. 290–299, Feb. 2018, doi: 10.1038/s41593-017-0056-2.  
[32] B. Pijuan-Sala et al., "A single-cell molecular map of mouse gastrulation and early organogenesis," *Nature*, vol. 566, no. 7745, pp. 490–495, Feb. 2019, doi: 10.1038/s41586-019-0933-9.  
[33] J. M. Replogle et al., "Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq," *Cell*, vol. 185, no. 14, pp. 2559–2575.e28, Jul. 2022, doi: 10.1016/j.cell.2022.05.013.  
[34] T. M. Norman et al., "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes," *Science*, vol. 365, no. 6455, pp. 786–793, Aug. 2019, doi: 10.1126/science.aax4438.  
[35] A. Pratapa, A. P. Jalihal, J. N. Law, A. Bharadwaj, and T. M. Murali, "Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data," *Nat. Methods*, vol. 17, no. 2, pp. 147–154, Feb. 2020, doi: 10.1038/s41592-019-0690-6.
