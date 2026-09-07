"""
Representation geometry metrics.

All functions operate on numpy arrays and are framework-agnostic.
Each function is pure: same input → same output, no side effects.

Metrics implemented
-------------------
- compute_anisotropy           : Average cosine similarity between representation pairs
                                 (exact for N <= threshold, sampled with CI for large N)
- compute_uncentered_svd_ratio : Fraction of energy along top singular value(s) of uncentered matrix
                                 (direct measure of directional cone concentration)
- compute_mev                  : Maximum Explainable Variance via centered PCA (Mu & Viswanath, 2018)
- compute_self_similarity      : Average cosine similarity between contextual representations of the same token
                                 (Ethayarajh, 2019)
- compute_linear_cka           : Linear Centered Kernel Alignment between two representation spaces
                                 (Kornblith et al., 2019)
- compute_layer_drift          : Layer-to-layer cosine distance and CKA similarity
- compute_cka_elbow            : Layer where consecutive CKA similarity plateaus
- compute_max_drift_layer      : Layer with maximum representation shift
- compute_silhouette           : Silhouette coefficient for labeled clusters
- compute_semantic_onset_depth : First layer where anisotropy delta plateaus (hypothesized onset)
- evaluate_sod_sensitivity     : Grid analysis of SOD across thresholds and window sizes
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. Shape: (n, d) → (n, d)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


# ---------------------------------------------------------------------------
# Core metrics: Anisotropy & Singular Spectrum
# ---------------------------------------------------------------------------


def compute_anisotropy(
    representations: np.ndarray,
    n_samples: int = 1000,
    random_state: int = 42,
    exact_threshold: int = 2000,
    return_ci: bool = False,
) -> float | tuple[float, tuple[float, float]]:
    """Compute anisotropy as average pairwise cosine similarity between representations.

    Isotropic representations → anisotropy ≈ 0.
    Degenerate / anisotropic cone representations → anisotropy → 1.

    Reference: Ethayarajh (2019), "How Contextual are Contextualized Word Representations?"

    Parameters
    ----------
    representations:
        Array of shape (n, hidden_size).
    n_samples:
        Number of random pairs to sample when n > exact_threshold.
    random_state:
        NumPy random seed for reproducibility.
    exact_threshold:
        If n <= exact_threshold, computes exact pairwise cosine across all n*(n-1)/2 pairs.
    return_ci:
        If True and sampled, returns (mean, (ci_lower, ci_upper)) using 95% normal CI.

    Returns
    -------
    float or tuple of (float, (float, float)):
        Average cosine similarity in [-1, 1], with optional 95% confidence interval.
    """
    n = len(representations)
    if n < 2:
        raise ValueError(f"Need at least 2 representations to compute anisotropy, got {n}.")

    normed = _l2_normalize(representations)

    # Exact computation for small to moderate datasets
    total_pairs = n * (n - 1) // 2
    if n <= exact_threshold or total_pairs <= n_samples:
        sim_matrix = normed @ normed.T
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_indices]
        mean_sim = float(np.mean(pairwise_sims))
        if return_ci:
            se = (
                float(np.std(pairwise_sims, ddof=1) / np.sqrt(len(pairwise_sims)))
                if len(pairwise_sims) > 1
                else 0.0
            )
            return mean_sim, (mean_sim - 1.96 * se, mean_sim + 1.96 * se)
        return mean_sim

    # Controlled sampling without replacement for large datasets
    rng = np.random.default_rng(random_state)
    sample_size = min(n_samples, total_pairs)

    idx_a = rng.integers(0, n, size=sample_size * 2)
    idx_b = rng.integers(0, n, size=sample_size * 2)
    valid_mask = idx_a != idx_b
    idx_a, idx_b = idx_a[valid_mask][:sample_size], idx_b[valid_mask][:sample_size]

    cos_sims = (normed[idx_a] * normed[idx_b]).sum(axis=1)
    mean_sim = float(cos_sims.mean())

    if return_ci:
        se = float(cos_sims.std(ddof=1) / np.sqrt(len(cos_sims)))
        return mean_sim, (mean_sim - 1.96 * se, mean_sim + 1.96 * se)

    return mean_sim


def compute_uncentered_svd_ratio(
    representations: np.ndarray,
    n_components: int = 1,
) -> float:
    """Compute the fraction of variance/energy captured by top uncentered singular value(s).

    Unlike centered PCA (which removes the global mean vector), the uncentered
    singular spectrum directly measures alignment with the dominant cone direction
    (i.e. directional collapse / anisotropy around the origin).

    Parameters
    ----------
    representations:
        Array of shape (n, hidden_size).
    n_components:
        Number of top singular values to consider.

    Returns
    -------
    float in [0, 1]. High value indicates representations lie in a narrow ray/cone.
    """
    n, d = representations.shape
    if n < 1 or d < 1:
        raise ValueError("Representations array must be non-empty.")

    # Compute singular values of X without centering
    s = np.linalg.svd(representations, compute_uv=False)
    total_energy = float(np.sum(s**2))
    if total_energy == 0.0:
        return 0.0

    k = min(n_components, len(s))
    top_energy = float(np.sum(s[:k] ** 2))
    return float(top_energy / total_energy)


def compute_mev(
    representations: np.ndarray,
    n_components: int = 1,
) -> float:
    """Maximum Explainable Variance: variance explained by top PCA component(s) after centering.

    Reference: Mu & Viswanath (2018), "All-but-the-Top"

    Parameters
    ----------
    representations:
        Array of shape (n, hidden_size).
    n_components:
        Number of top principal components to sum over.

    Returns
    -------
    float in [0, 1]. Higher means centered variations lie in a low-dimensional subspace.
    """
    n, d = representations.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples for PCA MEV, got {n}.")

    n_components = min(n_components, n, d)
    pca = PCA(n_components=n_components)
    pca.fit(representations)
    return float(pca.explained_variance_ratio_[:n_components].sum())


# ---------------------------------------------------------------------------
# Self-Similarity (Token-Level Contextualization)
# ---------------------------------------------------------------------------


def compute_self_similarity(
    token_representations: dict[str, np.ndarray],
) -> float:
    """Average intra-token cosine similarity across all evaluated tokens.

    Given target tokens and their contextual embeddings extracted across diverse sentences:
    computes how similar an isolated token's representations are across contexts.
    High value = context-invariant / static; Low value = highly contextualized.

    Reference: Ethayarajh (2019), Section 4.

    Parameters
    ----------
    token_representations:
        Dict mapping token string to array of shape (n_occurrences, hidden_size).
        Tokens with fewer than 2 occurrences are skipped.

    Returns
    -------
    float in [-1, 1].
    """
    per_token_sims: list[float] = []

    for token, reps in token_representations.items():
        if len(reps) < 2:
            continue
        normed = _l2_normalize(reps)
        sim_matrix = normed @ normed.T
        n = len(reps)
        upper_indices = np.triu_indices(n, k=1)
        per_token_sims.append(float(sim_matrix[upper_indices].mean()))

    if not per_token_sims:
        raise ValueError("No token had >= 2 occurrences. Cannot compute self-similarity.")

    return float(np.mean(per_token_sims))


# ---------------------------------------------------------------------------
# Representation Drift & Centered Kernel Alignment (CKA)
# ---------------------------------------------------------------------------


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment (CKA) between two representation matrices.

    Computes representational similarity invariant to orthogonal transformation and isotropic scaling.
    Reference: Kornblith et al. (ICML 2019), "Similarity of Neural Network Representations Revisited".

    Parameters
    ----------
    X: Shape (n, d1)
    Y: Shape (n, d2)

    Returns
    -------
    float in [0, 1]. 1 = identical representational geometries.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}.")

    # Center column-wise
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # Compute HSIC via inner products: tr(X_c X_c^T Y_c Y_c^T) = ||Y_c^T X_c||_F^2
    cross_term = np.linalg.norm(Y_c.T @ X_c, ord="fro") ** 2
    norm_x = np.linalg.norm(X_c.T @ X_c, ord="fro")
    norm_y = np.linalg.norm(Y_c.T @ Y_c, ord="fro")

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0

    return float(cross_term / (norm_x * norm_y))


def compute_layer_drift(
    layer_representations: np.ndarray,
) -> dict[str, list[float]]:
    """Compute layer-to-layer geometric shift across consecutive layers.

    Parameters
    ----------
    layer_representations:
        Shape (n_layers + 1, n_sentences, hidden_size).

    Returns
    -------
    dict with:
        "cosine_distance": Mean cosine distance between consecutive layers (1 - cos_sim)
        "linear_cka"     : Linear CKA similarity between consecutive layers
    """
    n_layers_total = layer_representations.shape[0]
    cosine_distances: list[float] = []
    cka_sims: list[float] = []

    for l in range(n_layers_total - 1):
        x1 = _l2_normalize(layer_representations[l])
        x2 = _l2_normalize(layer_representations[l + 1])
        cos_dist = float(1.0 - (x1 * x2).sum(axis=1).mean())
        cosine_distances.append(cos_dist)

        cka = compute_linear_cka(layer_representations[l], layer_representations[l + 1])
        cka_sims.append(cka)

    return {
        "cosine_distance": cosine_distances,
        "linear_cka": cka_sims,
    }


def compute_cka_elbow(layer_representations: np.ndarray, min_similarity: float = 0.85) -> int:
    """Find the layer where consecutive-layer CKA similarity first plateaus above min_similarity.

    Returns
    -------
    int: Layer index (0-indexed). Returns -1 if never reached.
    """
    drift = compute_layer_drift(layer_representations)
    ckas = drift["linear_cka"]
    for l, val in enumerate(ckas):
        if val >= min_similarity:
            return l + 1
    return int(np.argmax(ckas)) + 1 if len(ckas) > 0 else -1


def compute_max_drift_layer(layer_representations: np.ndarray) -> int:
    """Find the layer with the highest representation shift from its predecessor."""
    drift = compute_layer_drift(layer_representations)
    cos_dists = drift["cosine_distance"]
    if not cos_dists:
        return 0
    return int(np.argmax(cos_dists)) + 1


# ---------------------------------------------------------------------------
# Clustering & Semantic Metrics
# ---------------------------------------------------------------------------


def compute_silhouette(
    representations: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Silhouette coefficient for cluster cohesion of representations."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 unique labels for silhouette score.")

    return float(silhouette_score(representations, labels, metric=metric))


# ---------------------------------------------------------------------------
# Semantic Onset Depth (Hypothesized Metric & Sensitivity)
# ---------------------------------------------------------------------------


def compute_semantic_onset_depth(
    layer_representations: np.ndarray,
    threshold: float = 0.50,
    window: int = 2,
) -> int:
    """Compute the hypothesized Semantic Onset Depth (SOD).

    Defined as the shallowest layer L where the absolute change in anisotropy
    drops and stays below `threshold * max_delta` for `window` consecutive layers.

    NOTE: This is evaluated as a research hypothesis, not an established ground-truth metric.

    Parameters
    ----------
    layer_representations:
        Shape (n_layers + 1, n_sentences, hidden_size).
    threshold:
        Fraction of max delta below which layer changes are considered stabilized.
    window:
        Number of consecutive stabilized layer transitions required.

    Returns
    -------
    int: Layer index (0-indexed). Returns -1 if no stable plateau is observed.
    """
    n_layers = layer_representations.shape[0]

    anisotropy_per_layer = np.array(
        [compute_anisotropy(layer_representations[l]) for l in range(n_layers)]
    )

    deltas = np.abs(np.diff(anisotropy_per_layer))
    if len(deltas) == 0:
        return -1

    max_delta = float(deltas.max())
    if max_delta == 0.0:
        return 0

    stability_mask = deltas < (threshold * max_delta)

    count = 0
    for i, stable in enumerate(stability_mask):
        if stable:
            count += 1
            if count >= window:
                return int(i - window + 2)
        else:
            count = 0

    return -1


def evaluate_sod_sensitivity(
    layer_representations: np.ndarray,
    thresholds: list[float] | None = None,
    windows: list[int] | None = None,
) -> dict[str, dict[str, int]]:
    """Evaluate SOD stability across a grid of thresholds and window sizes."""
    if thresholds is None:
        thresholds = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    if windows is None:
        windows = [1, 2, 3]

    results: dict[str, dict[str, int]] = {}
    for w in windows:
        results[f"window_{w}"] = {}
        for t in thresholds:
            sod = compute_semantic_onset_depth(layer_representations, threshold=t, window=w)
            results[f"window_{w}"][f"thresh_{t:.2f}"] = sod

    return results


def compute_all_metrics(
    layer_representations: np.ndarray,
    token_representations: dict[str, np.ndarray] | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, object]:
    """Convenience function: compute core representation geometry metrics across all layers."""
    n_layers = layer_representations.shape[0]

    anisotropy = [compute_anisotropy(layer_representations[l]) for l in range(n_layers)]
    svd_ratio = [compute_uncentered_svd_ratio(layer_representations[l]) for l in range(n_layers)]
    mev = [compute_mev(layer_representations[l]) for l in range(n_layers)]
    drift = compute_layer_drift(layer_representations)

    results: dict[str, object] = {
        "anisotropy_per_layer": anisotropy,
        "uncentered_svd_ratio_per_layer": svd_ratio,
        "mev_per_layer": mev,
        "layer_drift_cosine": drift["cosine_distance"],
        "layer_drift_cka": drift["linear_cka"],
        "semantic_onset_depth": compute_semantic_onset_depth(layer_representations),
        "cka_elbow_layer": compute_cka_elbow(layer_representations),
        "max_drift_layer": compute_max_drift_layer(layer_representations),
    }

    if token_representations is not None:
        results["self_similarity"] = compute_self_similarity(token_representations)

    if labels is not None:
        results["silhouette_per_layer"] = [
            compute_silhouette(layer_representations[l], labels) for l in range(n_layers)
        ]

    return results
