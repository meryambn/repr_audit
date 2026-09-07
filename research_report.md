# Does Representation Geometry Stabilization Predict Linguistic Usability in Transformers?
**An Empirical Investigation using `repr-audit`**

---

## Abstract
Contextualized word representations in transformer architectures are known to suffer from representation degeneration, often collapsing into an anisotropic cone with high pairwise cosine similarity. A frequent intuition in mechanistic interpretability is that the layer where representation geometry stabilizes—termed here the **Semantic Onset Depth (SOD)**—marks the point where the network "commits" to its contextual encoding. In this paper, we systematically interrogate this hypothesis across multiple transformer architectures (`bert-base-uncased`, `roberta-base`, `gpt2`, and `distilgpt2`). Using exact pairwise geometry metrics, uncentered singular value spectra, and a 4-task diagnostic probing suite (syntactic POS tagging, entity recognition, sentence topic classification, and semantic textual similarity), we trace layer-wise geometric stabilization against functional probe performance. Our empirical findings demonstrate that functional linguistic usability plateaus significantly earlier than geometric stabilization in bidirectional encoders (MAE = 3.0 layers in BERT), while autoregressive decoders exhibit high anisotropy from their embedding layers. These findings demonstrate that geometric stabilization does not reliably predict semantic commitment, indicating that late-layer geometric changes reflect residual accumulation rather than semantic refinement.

---

## 1. Introduction & Research Question

Modern language models map discrete tokens into continuous vector spaces $\mathbb{R}^d$. While static word embeddings (Word2Vec, GloVe) occupied relatively isotropic spaces, deep transformer models exhibit **anisotropy** (Ethayarajh, 2019; Gao et al., 2019), where vectors cluster in a narrow cone. 

A central question in transformer interpretability is:
> **Does the layer at which representation geometry stabilizes correspond to the layer at which linguistic information becomes stable and usable?**

Previous informal metrics have proposed computing the inflection point of anisotropy changes—a hypothesized **Semantic Onset Depth (SOD)**—to detect where contextualization "stabilizes." However, whether this geometric plateau aligns with empirical linguistic usability has remained untested.

In this work, we:
1. Formulate mathematical metrics to rigorously separate directional cone collapse (uncentered SVD energy ratio) from centered subspace variance (PCA Maximum Explainable Variance).
2. Implement true token-level subword contextualization, avoiding sentence-level pooling leakage in self-similarity evaluation.
3. Establish a 4-task diagnostic probing matrix (Syntactic, Entity, Topic, and STS) evaluated across all layers $0 \dots L$.
4. Evaluate four open transformer models across encoder and decoder families.
5. Provide open-source, fully reproducible code and data.

---

## 2. Methodology & Formal Metric Taxonomy

### 2.1 Global Anisotropy vs. Directional Dominance
Ethayarajh (2019) defined anisotropy as the expected cosine similarity between independent random token representations:
$$\text{Anisotropy} = \mathbb{E}_{x_i, x_j} [\cos(x_i, x_j)]$$
For small sample sizes ($N \le 2000$), we compute the **exact** pairwise mean over all $N(N-1)/2$ unique pairs:
$$\text{Anisotropy}_{\text{exact}} = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} \frac{x_i \cdot x_j}{\|x_i\|_2 \|x_j\|_2}$$

#### Centered MEV vs. Uncentered SVD
Maximum Explainable Variance (MEV; Mu & Viswanath, 2018) fits PCA on centered representations:
$$\tilde{X} = X - \frac{1}{N}\mathbf{1}\mathbf{1}^T X$$
However, because anisotropy is characterized by vectors pointing toward a shared non-zero directional mean vector $\mu$, subtracting $\mu$ removes the cone center. We introduce the **Uncentered SVD Ratio**:
$$\text{SVD}_1 = \frac{\sigma_1^2}{\sum_{k=1}^d \sigma_k^2}$$
where $\sigma_k$ are singular values of the uncentered matrix $X$. When vectors cluster tightly in a cone around $\mu$, $\text{SVD}_1 \to 1$, directly quantifying cone concentration without centering distortions.

### 2.2 Token-Level Self-Similarity
To measure polysemy and contextual sensitivity, we isolate the representations of specific target words across distinct linguistic contexts. For a target word $w$ occurring in sentences $s_1, \dots, s_K$:
1. Subword tokens forming $w$ are located via tokenizer character offset mappings.
2. Subwords are mean-pooled to yield contextual vector $h_{w, k}^{(l)}$ at layer $l$.
3. Self-similarity is computed across all occurrence pairs:
$$\text{SelfSim}(w, l) = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \cos(h_{w,i}^{(l)}, h_{w,j}^{(l)})$$
A low score indicates that the model assigns distinct, context-sensitive vectors to the word.

### 2.3 Layer Drift and Linear Centered Kernel Alignment (CKA)
To measure representation shift between consecutive layers $l$ and $l+1$, we compute:
1. **Cosine Drift**: $1 - \frac{1}{N}\sum_i \cos(x_i^{(l)}, x_i^{(l+1)})$.
2. **Linear CKA** (Kornblith et al., 2019):
$$\text{CKA}(X, Y) = \frac{\|Y_c^T X_c\|_F^2}{\|X_c^T X_c\|_F \|Y_c^T Y_c\|_F}$$
We identify the **CKA Elbow Layer** as the first layer where consecutive layer similarity reaches $\ge 0.85$.

---

## 3. Systematic Diagnostic Probing Suite

To assess linguistic usability, we train linear classifiers at every layer $l \in [0, L]$ using stratified train/test splits ($70/30$):

| Linguistic Dimension | Probing Task | Target Property | Metric |
|---|---|---|---|
| **Syntactic** | POS Tagging | Part-of-speech category (Noun, Verb, Adj, Adv) | Classification Accuracy |
| **Entity** | NER Classification | Named entity category (Person, Location, Org) | Classification Accuracy |
| **Semantic** | Topic Classification | 4-way balanced topic classification | Macro F1 |
| **Sentence Semantics**| STS Benchmark | Fine-grained semantic similarity | Spearman Rank Correlation ($\rho$) |

For each task $T$, we compute the **Plateau Layer**:
$$l_{\text{plateau}} = \min \{ l : \text{Score}(l) \ge 0.95 \cdot \max_{k} \text{Score}(k) \}$$

---

## 4. Empirical Results & Findings

### 4.1 Cross-Architecture Representation Geometry

| Model | Family | Layers | Avg Aniso | SVD$_1$ Peak | CKA Elbow | Max Drift | SOD ($\alpha=0.5$) | Avg Probe Plateau |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bert-base-uncased` | Encoder | 12 | 0.6471 | 0.7531 (L6) | Layer 1 | Layer 1 | Layer 5 | Layer 3.00 |
| `roberta-base` | Encoder | 12 | 0.9365 | 0.9972 (L12)| Layer 1 | Layer 1 | Layer 2 | Layer 0.25 |
| `gpt2` | Decoder | 12 | 0.7796 | 0.8149 (L0) | Layer 2 | Layer 1 | Layer 1 | Layer 0.75 |
| `distilgpt2` | Decoder | 6 | 0.8007 | 0.8351 (L0) | Layer 2 | Layer 1 | Layer 1 | Layer 0.75 |

### 4.2 Key Findings

#### 1. Geometric Stabilization Disconnects from Linguistic Plateau
In `bert-base-uncased`, topic classification reaches peak accuracy by Layer 3, and POS probe accuracy reaches 90% by Layer 2. However, anisotropy changes do not flatten until Layer 5 ($\text{SOD}=5$). The Mean Absolute Error (MAE) between SOD and probe plateaus is **3.00 layers**. In contrast, the CKA elbow and Max Drift layer occur at Layer 1, capturing the massive initial contextualization leap from static embeddings to contextual space.

#### 2. Severe Anisotropy in RoBERTa Does Not Impair Linear Separability
`roberta-base` exhibits extreme anisotropy (mean 0.9365; final layer SVD$_1 = 0.9972$). Nearly all representations point in a single narrow direction. Despite this severe directional collapse, linear probes achieve $1.0$ accuracy by Layer 0–1. This provides empirical confirmation that high cosine anisotropy does not imply loss of linear classification capacity; representations occupy a narrow cone, but remain linearly separable on hyperplanes slicing through that cone.

#### 3. Autoregressive Decoders are Directionally Polarized from the Outset
In `gpt2` and `distilgpt2`, the uncentered singular spectrum ratio $\text{SVD}_1$ is highest at Layer 0 ($\approx 0.81 - 0.83$). The causal mask and position embedding structure induce a strong directional bias prior to attention refinement. Probe plateaus occur almost immediately at Layer 0–1.

---

## 5. Sensitivity Analysis of Semantic Onset Depth

We conducted sensitivity analysis over the SOD threshold $\alpha \in [0.20, 0.95]$ and window size $w \in \{1, 2, 3\}$.
* For loose thresholds ($\alpha \ge 0.80$), SOD triggers prematurely at Layer 0 or 1 because almost every layer change is $< 0.80 \times \Delta_{\max}$.
* For strict thresholds ($\alpha \le 0.35$), SOD fluctuates widely or fails to trigger (returning $-1$).
* The metric is highly parameter-sensitive and lacks an intrinsic mathematical fixed point, making it unsuitable as an objective architectural constant.

---

## 6. Threats to Validity & Limitations
1. **Probe Complexity**: Linear probes measure linear decodability. Higher layers may encode complex non-linear relations that require multi-layer perceptron probes.
2. **Corpus Size**: Experiments were conducted on a curated benchmark of 40 multi-topic sentences and 32 diagnostic probing targets. While sufficient for statistical separation on foundational representations, scaling to full GLUE/SuperGLUE evaluation across hundreds of checkpoints remains future work.

---

## 7. Conclusion
We investigated whether the layer where representation geometry stabilizes corresponds to the layer where linguistic information becomes stable. Our findings indicate that **geometric anisotropy stabilization is largely decoupled from linguistic utility**. Basic syntactic and semantic features stabilize within the first 1–3 layers, whereas anisotropy trajectories continue shifting into deep layers due to residual stream mechanics. We advise researchers against interpreting geometric plateaus as "semantic commitment" without empirical probe validation.
