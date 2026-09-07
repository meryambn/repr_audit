# repr-audit: Auditing Transformer Representation Geometry & Linguistic Usability

[![CI](https://github.com/meryambn/repr-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/meryambn/repr-audit/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A research-grade library for analyzing the layer-wise representation geometry, directional collapse, and contextualization dynamics of transformer models.

---

## Core Research Question & Hypothesis

### Research Question
> **Does the layer at which representation geometry stabilizes correspond to the layer at which linguistic information becomes stable and usable?**

### Hypotheses Tested
1. **Hypothesis 1 (Geometric Anisotropy vs Directional Dominance)**: High pairwise cosine similarity (anisotropy) corresponds directly to uncentered first singular-value concentration ($\sigma_1^2 / \sum \sigma_i^2$), whereas centered PCA Maximum Explainable Variance (MEV) underestimates anisotropy because it subtracts the shared directional cone centroid.
2. **Hypothesis 2 (Contextualization Drift)**: Token-level contextualization (lower intra-token self-similarity across diverse sentence contexts) develops monotonically across intermediate layers before plateauing or deteriorating in final output layers.
3. **Hypothesis 3 (Semantic Onset Depth Validation)**: The hypothesized **Semantic Onset Depth (SOD)**—the layer index where layer-to-layer anisotropy change plateaus—does **not** reliably predict the layer where linguistic probing tasks (syntactic, entity, topic, and semantic similarity) achieve functional plateaus.

---

##  Mathematical Formulation of Metrics

| Metric | Formulation | Research Significance |
|---|---|---|
| **Global Anisotropy** | $\frac{2}{N(N-1)} \sum_{i < j} \frac{x_i \cdot x_j}{\|x_i\|_2 \|x_j\|_2}$ | Measures the "cone effect" (Ethayarajh 2019). Exact for $N \le 2000$; uniform sampling with 95% bootstrap CI for large $N$. |
| **Uncentered SVD Ratio** | $\frac{\sigma_1^2}{\sum_{k=1}^d \sigma_k^2} \quad \text{of } X \in \mathbb{R}^{n \times d}$ | Directly quantifies alignment with the dominant cone direction without centering. |
| **Centered MEV** | $\frac{\lambda_1}{\sum \lambda_k} \quad \text{where } \lambda = \text{eig}(\text{Cov}(X))$ | Maximum Explainable Variance of centered variations (Mu & Viswanath 2018). |
| **Token Self-Similarity** | $\frac{1}{|V|} \sum_{w \in V} \frac{2}{K_w(K_w-1)} \sum_{i < j} \cos(h_{w,i}, h_{w,j})$ | Subword-pooled representation of target token $w$ across distinct sentence contexts. |
| **Linear CKA** | $\frac{\|Y_c^T X_c\|_F^2}{\|X_c^T X_c\|_F \|Y_c^T Y_c\|_F}$ | Invariant layer-to-layer geometric alignment between consecutive layers $l$ and $l+1$ (Kornblith et al. 2019). |
| **Cosine Layer Drift** | $1 - \frac{1}{N} \sum_{i=1}^N \cos(x_i^{(l)}, x_i^{(l+1)})$ | Magnitude of directional shift between consecutive layers. |
| **Semantic Onset Depth (SOD)** | $\min \{ l : |\Delta \text{Aniso}_{l+k}| < \alpha \cdot \Delta_{\max}, \forall k \in [0, w-1] \}$ | Hypothesized layer of geometric stabilization ($\alpha=0.50, w=2$). |

---

##  Empirical Benchmark Results

Evaluated across 4 representative model architectures on standardized linguistic corpora:

| Model Architecture | Model Family | Layers | Avg Anisotropy ↓ | SVD Cone Ratio (Peak) | CKA Elbow | Max Drift Layer | SOD (Hypothesis) | Avg Probe Plateau | MAE (SOD vs Probes) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`bert-base-uncased`** | Encoder | 12 | **0.6471** | 0.7531 (L6) | Layer 1 | Layer 1 | Layer 5 | Layer 3.00 | 3.00 layers |
| **`roberta-base`** | Encoder | 12 | **0.9365** | 0.9972 (L12) | Layer 1 | Layer 1 | Layer 2 | Layer 0.25 | 1.75 layers |
| **`gpt2`** | Decoder | 12 | **0.7796** | 0.8149 (L0) | Layer 2 | Layer 1 | Layer 1 | Layer 0.75 | 0.25 layers |
| **`distilgpt2`** | Decoder | 6 | **0.8007** | 0.8351 (L0) | Layer 2 | Layer 1 | Layer 1 | Layer 0.75 | 0.25 layers |

> **Key Takeaway**: All numbers in this table are programmatically produced by `scripts/run_experiments.py` and stored in `benchmark_results/benchmark_data.json`.

---

## Systematic 4-Task Probing Suite

To test whether geometric stabilization corresponds to linguistic utility, we train linear diagnostic probes at every layer $0 \dots L$:

```
Layer 0  ──► Linear Probe Accuracy
Layer 1  ──► Linear Probe Accuracy
...
Layer L  ──► Linear Probe Accuracy
```

Compared across 4 linguistic dimensions:
1. **Syntactic**: Part-of-speech (POS) tagging across open-class words.
2. **Entity**: Named Entity Recognition (NER) across Person, Location, Organization categories.
3. **Semantic (Topic)**: Multi-class sentence topic classification.
4. **Sentence Semantics**: STS cosine similarity correlation against human ratings (Spearman $\rho$).

### Empirical Findings:
* **Linguistic plateaus occur earlier than late geometric stabilization**: In `bert-base-uncased`, topic and syntactic probes plateau by Layer 2–3, while geometric anisotropy does not stabilize until Layer 5 (MAE = 3.0 layers).
* **Extreme Anisotropy in RoBERTa**: `roberta-base` exhibits severe cone collapse (Avg Anisotropy = 0.9365), yet achieves near-perfect probe separability by Layer 0–1.
* **Autoregressive Decoders**: In `gpt2` and `distilgpt2`, representations are already strongly directional in the embedding layer ($\text{SVD}_1 \approx 0.81$), with linear probes plateauing immediately at Layer 0–1.

---

## Publication Figures

All figures are generated in 300 DPI vector-ready format:

* **Figure 1**: Cross-Architecture Representation Geometry (Anisotropy & Uncentered SVD Spectrum)
* **Figure 2**: Linguistic Probing Trajectories vs Geometric Markers (`bert-base-uncased` & `gpt2`)
* **Figure 3**: Semantic Onset Depth (SOD) Parameter Sensitivity Heatmap
* **Figure 4**: True Token-Level Contextualization Dynamics across Layers

---

## 📦 Library Architecture & Design

`repr-audit` is designed with a clean, two-tier architecture:

1. **Framework-Agnostic NumPy Metric Primitives (`repr_audit.metrics`)**:
   - Pure NumPy functions with no side effects and zero PyTorch dependency for mathematical calculations.
   - Computes exact/sampled anisotropy, uncentered SVD ratios, linear CKA, layer drift, and silhouette scores on arbitrary representations (`np.ndarray`) from **PyTorch, JAX, TensorFlow, or static embeddings**.
2. **High-Level Transformer Auditor (`repr_audit.RepresentationAuditor`)**:
   - Integrated with HuggingFace `transformers` and `torch`.
   - Automatically handles encoder vs. decoder architectures (padding direction, pooling strategies, and leak-free subword character-span mapping) in a single batched pass.

---

## 🚀 Quickstart & Usage

### 1. Installation

Install the library in editable mode:
```bash
git clone https://github.com/meryambn/repr-audit.git
cd repr-audit
pip install -e .
```

Or install with development dependencies (pytest, test coverage, ruff, black):
```bash
pip install -e ".[dev]"
```

### 2. Standalone NumPy Metrics (Framework-Agnostic)
If you already have representation matrices from custom models or non-HuggingFace pipelines:
```python
import numpy as np
from repr_audit.metrics import (
    compute_anisotropy,
    compute_uncentered_svd_ratio,
    compute_linear_cka,
    compute_semantic_onset_depth,
)

# Any matrix of shape (n_samples, hidden_dim)
X = np.random.randn(100, 768)
Y = np.random.randn(100, 768)

aniso = compute_anisotropy(X)                # Pairwise cosine similarity
svd_ratio = compute_uncentered_svd_ratio(X)  # Directional cone concentration
cka = compute_linear_cka(X, Y)               # Linear Centered Kernel Alignment
```

### 3. Full Transformer Model Audit
```python
from repr_audit import RepresentationAuditor

auditor = RepresentationAuditor("bert-base-uncased")

results = auditor.audit(
    sentences=[
        "The central bank lowered benchmark interest rates.",
        "A fisherman cast his line from the muddy river bank.",
        "Deep neural networks learn hierarchical representations."
    ],
    token_sentences={
        "bank": [
            "The commercial bank approved the loan application.",
            "Wildflowers bloomed along the steep river bank."
        ]
    }
)

print(results.summary())
results.plot()
```

### 4. Cross-Model Comparison
```python
from repr_audit import RepresentationAuditor
from repr_audit.plotting import plot_model_comparison

models = ["bert-base-uncased", "roberta-base", "gpt2"]
sentences = [...]  # your evaluation sentences

all_results = [RepresentationAuditor(m).audit(sentences) for m in models]
plot_model_comparison(all_results, metric="anisotropy")
```

---

## Reproducibility

To re-run the entire benchmark and re-generate all figures from scratch:
```bash
# 1. Run unit & integration test suite (29 tests)
pytest tests/ -v --cov=repr_audit

# 2. Run full multi-model benchmark pipeline
python scripts/run_experiments.py

# 3. Generate 300 DPI publication figures
python scripts/generate_paper_figures.py
```

---

## Limitations & Threats to Validity

1. **Probe Complexity vs Representation Form**: Linear probing measures linear separability, but nonlinear representations may be usable by subsequent self-attention layers without being linearly separable.
2. **Dataset Scale**: The benchmark uses a controlled set of 40 standard sentences across 4 balanced topics with target polysemous words. Scaling to entire corpora (e.g. 100k sentences) requires GPU resources.
3. **Anisotropy Confounding**: Residual stream accumulation naturally increases cosine similarity between subsequent layers, which can inflate anisotropy measurements independently of semantic changes.

---

## References

1. **Ethayarajh, K.** (EMNLP 2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Representations.*
2. **Mu, J., & Viswanath, P.** (ICLR 2018). *All-but-the-Top: Simple and Effective Postprocessing for Word Representations.*
3. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G.** (ICML 2019). *Similarity of Neural Network Representations Revisited.*
4. **Hewitt, J., & Manning, C. D.** (NAACL 2019). *A Structural Probe for Finding Syntax in Word Representations.*
5. **Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T. Y.** (ICLR 2019). *Representation Degeneration Problem in Language Modeling.*

---

##  License

MIT
