# RQTAD: Residual Quantization for Time Series Anomaly Detection

## 1. Overview

RQTAD is a zero-shot time series anomaly detection model based on **Residual Quantization K-Means (RQKMeans)**. It achieves SOTA performance on the TSB-AD-U benchmark without any training labels or gradient-based optimization.

This document provides a **Gaussian Process (GP) perspective** on why RQKMeans works, what it can and cannot detect, and the precise mathematical boundaries of its detection capability.

---

## 2. Gaussian Process Formulation

### 2.1 Signal Model

We model the time series as a sample from a **Gaussian Process** with a composite kernel:

$$x(t) \sim \mathcal{GP}\big(0,\; K(t, t')\big)$$

$$K(t, t') = \underbrace{\sum_{i=1}^{M} \sigma_i^2 \, \mathcal{K}_{\text{Mat\acute{e}rn}}^{\nu_i}\!\left(\frac{|t - t'|}{\ell_i}\right)}_{K_{\text{smooth}}(t,t')} \;+\; \underbrace{\sum_{j=1}^{P} \eta_j^2 \, \exp\!\left(-\frac{2\sin^2\!\big(\pi|t-t'|/p_j\big)}{w_j^2}\right)}_{K_{\text{per}}(t,t')} \;+\; \underbrace{\sigma_\epsilon^2 \, \delta_{t,t'}}_{K_{\text{noise}}(t,t')}$$

| Component | Role | Parameter |
|-----------|------|-----------|
| **Matérn** $\mathcal{K}_\nu$ | Smooth trends, slow drifts | Smoothness $\nu$, lengthscale $\ell$, variance $\sigma^2$ |
| **Periodic** $\mathcal{K}_{\text{per}}$ | Repeating patterns (daily, weekly, ...) | Period $p$, width $w$, variance $\eta^2$ |
| **Noise** | Observation noise, micro-fluctuations | Variance $\sigma_\epsilon^2$ |

The Matérn kernel with smoothness parameter $\nu$ has the spectral density:

$$S_{\text{Matérn}}(\omega) \propto \left(1 + \left(\frac{\ell \omega}{2\nu}\right)^2\right)^{-(\nu + 1/2)}$$

which implies **polynomial eigenvalue decay** in the finite-window covariance matrix.

### 2.2 Sliding Window Covariance

A sliding window of size $W$ starting at time $s$ produces the random vector:

$$\mathbf{w}_s = \big[x(s),\; x(s+1),\; \ldots,\; x(s+W-1)\big]^\top \in \mathbb{R}^W$$

Under the GP prior, $\mathbf{w}_s \sim \mathcal{N}(\mathbf{0},\; \boldsymbol{\Sigma})$ where the covariance matrix has the **Toeplitz** structure:

$$\boldsymbol{\Sigma}_{ij} = K(i, j), \quad i, j = 0, 1, \ldots, W-1$$

Since $K$ only depends on the lag $|i - j|$, we write $\boldsymbol{\Sigma} = \text{Toeplitz}\big(K(0),\; K(1),\; \ldots,\; K(W-1)\big)$.

### 2.3 Eigendecomposition and Effective Dimensionality

Let $\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$ with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_W > 0$.

**Definition (Effective Dimensionality).** The *effective dimensionality* of the sliding window distribution is:

$$d_{\text{eff}} = \frac{\left(\sum_{i=1}^{W} \lambda_i\right)^2}{\sum_{i=1}^{W} \lambda_i^2} = \frac{\big(\text{tr}\,\boldsymbol{\Sigma}\big)^2}{\|\boldsymbol{\Sigma}\|_F^2}$$

This is the reciprocal of the participation ratio: $d_{\text{eff}} \in [1, W]$, measuring the number of "active" eigenmodes.

For the composite kernel, $d_{\text{eff}}$ decomposes as:

$$d_{\text{eff}} \approx d_{\text{Matérn}} + d_{\text{periodic}}$$

**Matérn contribution.** For the Matérn kernel with smoothness $\nu$, eigenvalues decay as:

$$\lambda_k^{(\text{M})} \sim C \cdot k^{-(2\nu + 1)}, \quad k \to \infty$$

The number of eigenvalues above a threshold $\epsilon \cdot \lambda_1$ is:

$$d_{\text{Matérn}} \sim \left(\frac{C}{\epsilon \cdot \lambda_1}\right)^{\frac{1}{2\nu + 1}}$$

**Rough signals** ($\nu = 1/2$, exponential kernel): $d_{\text{Matérn}} \sim O(W)$ — nearly full rank.
**Smooth signals** ($\nu \to \infty$, SE kernel): $d_{\text{Matérn}} \sim O(1)$ — very low rank.

**Periodic contribution.** Each periodic component with period $p_j$ contributes $\lfloor W/p_j \rfloor$ pairs of dominant eigenvalues (corresponding to $\sin$ and $\cos$ Fourier modes):

$$d_{\text{periodic}} \approx 2\sum_{j=1}^{P} \left\lfloor \frac{W}{p_j} \right\rfloor$$

The remaining eigenvalues are at the noise floor $\sigma_\epsilon^2$.

---

## 3. KMeans Quantization on the GP Manifold

### 3.1 Quantization as Lossy Compression

KMeans with $k$ codewords $\{\mathbf{c}_1, \ldots, \mathbf{c}_k\} \subset \mathbb{R}^W$ partitions $\mathbb{R}^W$ into Voronoi cells and approximates each window by its nearest codeword. The **mean squared quantization error** is:

$$Q(k) = \mathbb{E}\!\left[\min_{1 \leq j \leq k} \|\mathbf{w} - \mathbf{c}_j\|^2\right]$$

where $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$.

### 3.2 Quantization Error Bound via Eigenvalue Decay

**Theorem 1 (GP Quantization Bound).**
*For $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_W$, the optimal $k$-means quantization error satisfies:*

$$Q(k) \leq \sum_{i > d_{\text{eff}}(k)} \lambda_i$$

*where $d_{\text{eff}}(k)$ is the number of eigenvalues effectively captured by $k$ codewords.*

**Intuition:** $k$ codewords in $\mathbb{R}^W$ can span at most a $(k-1)$-dimensional affine subspace. The quantization error equals the energy in the eigenmodes orthogonal to this subspace.

**Corollary 1 (Matérn-dominated signals).** For a kernel with eigenvalue decay $\lambda_k \sim k^{-\alpha}$ where $\alpha = 2\nu + 1$:

$$Q(k) \sim O\!\left(k^{-(\alpha - 1)}\right) = O\!\left(k^{-2\nu}\right)$$

| Smoothness $\nu$ | Kernel type | Decay rate $\alpha$ | $Q(k)$ rate |
|-------------------|-------------|---------------------|-------------|
| $1/2$ | Exponential | 2 | $O(k^{-1})$ |
| $3/2$ | Matérn-3/2 | 4 | $O(k^{-3})$ |
| $5/2$ | Matérn-5/2 | 6 | $O(k^{-5})$ |
| $\infty$ | SE (RBF) | $\infty$ | $O(e^{-ck})$ |

**Key insight:** The smoother the signal, the faster KMeans converges — fewer codewords are needed.

**Corollary 2 (Periodic-dominated signals).** For a kernel with $d_{\text{per}}$ dominant eigenvalues at $\eta^2$ and the rest at $\sigma_\epsilon^2$:

$$Q(k) \approx \begin{cases} (d_{\text{per}} - k) \cdot \eta^2 + (W - d_{\text{per}}) \cdot \sigma_\epsilon^2 & k < d_{\text{per}} \\ (W - k) \cdot \sigma_\epsilon^2 & k \geq d_{\text{per}} \end{cases}$$

Once $k$ exceeds the number of periodic eigenmodes, the error saturates at the noise floor.

---

## 4. Residual Quantization: Hierarchical Decomposition

### 4.1 Cascade Structure

RQKMeans uses $L$ codebooks $\mathcal{C}_0, \mathcal{C}_1, \ldots, \mathcal{C}_{L-1}$ with $k_0, k_1, \ldots, k_{L-1}$ codewords each. Each level quantizes the residual from the previous level:

$$\hat{\mathbf{w}} = \sum_{l=0}^{L-1} \mathbf{c}_l^*, \qquad \mathbf{c}_l^* = \arg\min_{\mathbf{c} \in \mathcal{C}_l} \|\mathbf{r}_{l-1} - \mathbf{c}\|^2$$

where $\mathbf{r}_{-1} = \mathbf{w}$ and $\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{c}_l^*$.

### 4.2 Why Cascade? A GP Eigenmode Argument

**Theorem 2 (Hierarchical Eigenmode Capture).**
*Under the GP model, the RQ cascade achieves hierarchical eigenmode decomposition: Level $l$ captures the dominant eigenmodes of the residual covariance $\text{Cov}(\mathbf{r}_{l-1})$. The total quantization error after $L$ levels satisfies:*

$$Q_{\text{cascade}} \leq \sum_{l=0}^{L-1} \sum_{i > k_l} \lambda_i^{(l)}$$

*where $\lambda_i^{(l)}$ are the eigenvalues of $\text{Cov}(\mathbf{r}_{l-1})$.*

**Proof sketch.** At each level, the KMeans codewords span a $k_l$-dimensional subspace that captures the top $k_l$ eigenmodes of the current residual. The remaining residual $\mathbf{r}_l$ has covariance whose eigenvalues are the tail eigenvalues $\{\lambda_i^{(l)}\}_{i > k_l}$. By induction, the total error is the sum of per-level tail energies. $\square$

**Decomposition for the composite kernel:**

$$Q_{\text{cascade}} \leq \underbrace{\sum_{i > k_0} \lambda_i^{(0)}}_{\text{Level 0: periodic modes}} + \underbrace{\sum_{i > k_1} \lambda_i^{(1)}}_{\text{Level 1: smooth modes}} + \underbrace{\sum_{i > k_2} \lambda_i^{(2)}}_{\text{Level 2: noise floor}} + \cdots$$

**Design principle:** The descending $k$-list `[40, 20, 10]` in RQTAD reflects this hierarchy:

| Level | Codewords | Captures | GP component |
|-------|-----------|----------|--------------|
| 0 | $k_0 = 40$ | Dominant periodic eigenmodes | $K_{\text{per}}$ |
| 1 | $k_1 = 20$ | Secondary smooth structure | $K_{\text{Matérn}}$ |
| 2 | $k_2 = 10$ | Residual fine structure | Remaining $K_{\text{Matérn}}$ tail + noise |

### 4.3 Product Codebook vs. Greedy Cascade

The **product codebook** $\mathcal{C}_{\text{prod}} = \{\mathbf{c}_0 + \mathbf{c}_1 + \cdots + \mathbf{c}_{L-1}\}$ has $K = \prod_l k_l$ codewords and achieves the optimal rate:

$$Q_{\text{prod}}(K) \leq \sum_{i > \log_2 K} \lambda_i$$

The **greedy cascade** (used in RQTAD) is suboptimal because each level is trained independently. However, it achieves:

$$Q_{\text{cascade}} \leq \sum_{l=0}^{L-1} Q_{\text{single}}(k_l)$$

which is a **polynomial** improvement ($L$-fold reduction) rather than exponential. The practical advantage is computational: training $L$ codebooks of size $k$ costs $O(L \cdot k \cdot N \cdot W)$ vs. $O(k^L \cdot N \cdot W)$ for the product codebook.

---

## 5. Anomaly Score Analysis

### 5.1 Score Definition

The anomaly score after $L$ levels of quantization is:

$$A(\mathbf{w}) = \max_{1 \leq j \leq W} \left| r_j^{(L)} - \text{median}\big(\mathbf{r}^{(L)}\big) \right|$$

where $\mathbf{r}^{(L)} = \mathbf{w} - \hat{\mathbf{w}}$ is the final residual vector.

### 5.2 Score Distribution Under the GP Null Hypothesis

**Theorem 3 (Normal Score Distribution).**
*For a normal window $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$, the residual after $L$-level RQ is approximately:*

$$\mathbf{r}^{(L)} \approx \mathcal{N}\!\left(\mathbf{0},\; \boldsymbol{\Sigma}_{\text{res}}^{(L)}\right)$$

*where $\boldsymbol{\Sigma}_{\text{res}}^{(L)} = \text{Cov}(\mathbf{r}^{(L)})$ has eigenvalues $\{\lambda_i^{(L)}\}_{i=1}^W$ consisting of the tail eigenvalues not captured by any codebook.*

The anomaly score $A(\mathbf{w})$ is the **range** of a Gaussian random vector projected onto the coordinate axes, shifted by the median. For approximately i.i.d. residual components (valid when the codebooks capture the dominant correlated structure):

$$\mathbb{E}[A(\mathbf{w})] \approx \sigma_{\text{res}} \cdot \mathbb{E}\!\left[\max_j |Z_j - \text{median}(\mathbf{Z})|\right]$$

where $Z_j \sim \mathcal{N}(0, 1)$ i.i.d. and $\sigma_{\text{res}}^2 = \frac{1}{W}\text{tr}(\boldsymbol{\Sigma}_{\text{res}}^{(L)})$.

For $W$ i.i.d. standard Gaussians, the expected max absolute deviation from the median scales as:

$$\mathbb{E}\!\left[\max_j |Z_j - \text{median}(\mathbf{Z})|\right] \approx \sqrt{2\ln W}$$

Therefore:

$$\mathbb{E}[A(\mathbf{w})] \approx \sigma_{\text{res}} \cdot \sqrt{2\ln W}$$

### 5.3 Detection Threshold

Setting a detection threshold at $\tau = \mu_A + \gamma \cdot \sigma_A$ where $\mu_A = \mathbb{E}[A(\mathbf{w})]$ and $\sigma_A = \text{std}(A(\mathbf{w}))$, a window is flagged as anomalous if $A(\mathbf{w}) > \tau$.

In practice, RQTAD uses **median absolute deviation (MAD)** over all window scores as a robust threshold:

$$\text{threshold} = \text{median}(\{A_i\}) + \gamma \cdot \text{MAD}(\{A_i\})$$

---

## 6. Capability Boundaries

### 6.1 Detectability Condition

**Theorem 4 (Minimum Detectable Anomaly).**
*An anomalous window $\mathbf{w}_{\text{anom}} = \mathbf{w}_{\text{normal}} + \boldsymbol{\delta}$ is detectable by RQKMeans if and only if:*

$$\|\boldsymbol{\delta}_{\perp}\| > \tau_{\text{noise}} \triangleq \sqrt{\sum_{i > d_{\text{eff}}} \lambda_i} \cdot (1 + \gamma)$$

*where $\boldsymbol{\delta}_{\perp}$ is the component of the anomaly vector orthogonal to the subspace spanned by the codebook codewords, and $\tau_{\text{noise}}$ is the noise floor determined by the unexplained eigenmodes.*

**Interpretation:** The model can only detect anomalies in directions **not already covered** by the codebooks. If an anomaly lies entirely within the span of the codewords, it will be "absorbed" by the quantization and produce a near-zero residual.

### 6.2 Undetectable Anomaly Types

| Anomaly type | $\|\boldsymbol{\delta}_\perp\|$ | Detectable? |
|-------------|------|-------------|
| **Phase shift** $\mathbf{w}_{\text{anom}} = [x(s+\Delta), \ldots]$ | $\approx 0$ (same GP realization) | **No** — lies on the same manifold |
| **Amplitude scaling** $\alpha \cdot \mathbf{w}$ | $(\alpha - 1)\|\mathbf{w}\|$ | Yes if $|\alpha - 1| \cdot \|\mathbf{w}\| > \tau_{\text{noise}}$ |
| **Gaussian noise** $\boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \sigma_a^2 \mathbf{I})$ | $\sigma_a \sqrt{W}$ | Yes if $\sigma_a > \tau_{\text{noise}} / \sqrt{W}$ |
| **Point anomaly** (single spike) | $|\Delta|$ | Yes if $|\Delta| > \tau_{\text{noise}}$ |
| **Pattern substitution** (replace with another normal pattern) | $\approx 0$ | **No** — maps to another codeword |
| **Frequency shift** (change period) | Depends on spectral overlap | Partially — if new frequency not in codebook |

### 6.3 Phase and Stride Relationship

For a GP with dominant periodic kernel at period $p$, sliding windows starting at times $s$ and $s + \delta$ have:

$$\text{corr}(\mathbf{w}_s, \mathbf{w}_{s+\delta}) = \frac{K_{\text{per}}(\delta)}{K_{\text{per}}(0)} = \exp\!\left(-\frac{2\sin^2(\pi\delta/p)}{w^2}\right)$$

**Stride = 1:** Consecutive windows are highly correlated:

$$\text{corr}(\mathbf{w}_s, \mathbf{w}_{s+1}) \approx 1 - \frac{2\pi^2}{w^2 p^2} \quad \text{(for } p \gg 1\text{)}$$

This creates massive redundancy. The **effective independent sample count** is:

$$N_{\text{eff}} \approx \frac{N}{p} \cdot \frac{p}{\gcd(\delta, p)} = \frac{N}{\gcd(\delta, p)}$$

**Stride $\delta$ coprime to period $p$:** $N_{\text{eff}} = N/p$ — all phases are covered with minimal redundancy.
**Stride $\delta = p$:** $N_{\text{eff}} = N/p$ but only one phase per period — no phase diversity.

**Optimal stride:** $\delta = 1$ maximizes information (full phase coverage) at the cost of computation. The redundant windows are harmless because KMeans naturally groups them into the same cluster.

### 6.4 Period Detection and the ACF Connection

The autocorrelation function of the GP is:

$$\text{ACF}(\tau) = \frac{K(\tau)}{K(0)} = \frac{\sum_i \sigma_i^2 \mathcal{K}_{\text{M}}(\tau/\ell_i) + \sum_j \eta_j^2 \exp(-2\sin^2(\pi\tau/p_j)/w_j^2)}{\sum_i \sigma_i^2 + \sum_j \eta_j^2 + \sigma_\epsilon^2}$$

The periodic components create **peaks** in the ACF at lags $\tau = p_j, 2p_j, 3p_j, \ldots$. The `_detect_period` method in RQTAD finds these peaks to set the window size $W = p_{\text{dominant}}$.

**Why this works:** Setting $W = p$ ensures that the Toeplitz covariance matrix $\boldsymbol{\Sigma}$ has exactly $d_{\text{periodic}}$ dominant eigenvalues (corresponding to the Fourier modes of the periodic kernel), which KMeans can efficiently capture.

---

## 7. The Complete Picture

### Pipeline

```
Raw time series x(t)
    │
    ▼
┌─────────────────────────────────────────────┐
│  GP Kernel: K = K_Matérn + K_periodic + K_ε │
│  → Toeplitz covariance Σ (W × W)            │
│  → Eigenvalues {λ₁ ≥ λ₂ ≥ ... ≥ λ_W}       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Sliding Window (stride=1, W=p_detected)     │
│  w_s ∈ ℝ^W, w_s ~ N(0, Σ)                  │
│  → N_windows ≈ N overlapping samples         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Level 0: KMeans (k₀ codewords)              │
│  → Captures top k₀ eigenmodes of Σ           │
│  → Residual r₀ = w - c₀*                    │
│  → Cov(r₀) has tail eigenvalues {λᵢ}_{i>k₀} │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Level 1: KMeans (k₁ codewords)              │
│  → Captures top k₁ eigenmodes of Cov(r₀)     │
│  → Residual r₁ = r₀ - c₁*                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Level 2: KMeans (k₂ codewords)              │
│  → Captures remaining structure              │
│  → Final residual r₂                         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Anomaly Score: A(w) = max|r₂ⱼ - median(r₂)| │
│  → Normal windows: A(w) ~ σ_res · √(2ln W)  │
│  → Anomalous windows: A(w) >> threshold      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Window→Point Score Mapping (difference array)│
│  score(t) = mean of A(w_s) for all s∋t∈w_s  │
└─────────────────────────────────────────────┘
```

### Summary of Bounds

| Quantity | Bound | Depends on |
|----------|-------|------------|
| Effective dimensionality | $d_{\text{eff}} \approx d_{\text{Matérn}} + d_{\text{per}}$ | $\nu, \ell, p, W$ |
| Single-level quantization error | $Q(k) \leq \sum_{i>k} \lambda_i$ | Eigenvalue decay |
| Cascade quantization error | $Q_{\text{cascade}} \leq \sum_l \sum_{i>k_l} \lambda_i^{(l)}$ | Per-level eigenvalues |
| Normal score expectation | $\mathbb{E}[A] \approx \sigma_{\text{res}} \sqrt{2\ln W}$ | Residual variance, window size |
| Minimum detectable anomaly | $\delta_{\min} \propto \sigma_{\text{res}}$ | Unexplained eigenmode energy |
| Effective independent samples | $N_{\text{eff}} = N / \gcd(\delta, p)$ | Stride, period |

### Capability Boundary (One Sentence)

> **RQKMeans can detect any anomaly whose projection onto the unexplained eigenmodes of the GP covariance exceeds the quantization noise floor $\sigma_{\text{res}} \sqrt{2\ln W}$; it cannot detect anomalies that are phase-shifts or substitutions of normal patterns, as these lie within the span of the codebook codewords.**

---

## 8. RQKMeans Variants

### 8.1 PyramidRQKMeans

Uses SVD-based dimensionality reduction at each level. Instead of clustering in $\mathbb{R}^W$, it projects residuals onto the top-$d_l$ singular vector subspace before clustering.

**GP interpretation:** This explicitly restricts each level to capture only the top eigenmodes, acting as a **spectral filter**. The encoder $\mathbf{V}_{d_l}^\top$ projects onto the dominant subspace, and the decoder $\mathbf{V}_{d_l}$ reconstructs in the original space.

### 8.2 PruneRQKMeans

Adds an outlier pruning mechanism during training. After fitting each codebook, codewords with anomalous relevance ratios $\rho_k = \delta_k / \|\mathbf{c}_k\|^2$ are masked out.

**GP interpretation:** This removes codewords that capture **out-of-distribution** patterns (anomalies in the training data), preventing the codebook from "explaining away" anomalies. Only codewords representing the **mainstream GP behavior** are retained.

### 8.3 AlignMAD

Adds phase-alignment before quantization via cyclic rotation: $\mathbf{w}_s \to \mathcal{A}(\mathbf{w}_s)$.

**GP interpretation:** Phase-alignment transforms the Toeplitz covariance into a **circulant** matrix. A circulant matrix is diagonalized by the DFT matrix, meaning the eigenmodes are exactly the Fourier modes. This makes the periodic kernel's contribution **rank-1** (one eigenvalue at $\eta^2$, rest at $\sigma_\epsilon^2$), dramatically reducing the effective dimensionality and improving quantization efficiency.

---

## 9. References

- [TSB-AD-U Benchmark](https://github.com/decisionintelligence/TSB-AD)
- Paparrizos et al., "Volume Under the Surface: A Three-Dimensional Accuracy Metric for Time-Series Anomaly Detection" (VLDB 2022)
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (MIT Press, 2006)
- Gray & Neuhoff, "Quantization" (IEEE Trans. IT, 1998)
